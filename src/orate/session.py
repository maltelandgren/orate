"""Session runner — the Act-4 self-referential loop.

A Session wraps an engine with persistent KV, a registry of ``@program``
tools, and an outer grammar that admits free text interleaved with
``@name(args)`` tool invocations. The model samples one continuous token
stream; each registered program contributes a named sub-rule to the
outer grammar. When the model emits a call, the runtime dispatches:

* ``@make_new_program(name, task)`` — the bootstrap. Runtime triggers a
  grammar-switched sample under ``PROGRAM_SOURCE_GRAMMAR``, validates
  + compiles the emitted source, registers the new program, and
  rebuilds the outer grammar. From this point on, the model can emit
  the new program's name in any subsequent turn.
* ``@foo(args…)`` — any other registered program. Args are emitted
  under foo's body grammar (from ``derive_body_grammar``) and captured
  by the runtime. If foo has ``@program(ends_turn=True)``, the session
  hands control back to the client here with the args as the result.
  Otherwise the runtime appends a serialized confirmation to the KV
  and continues outer sampling.

No rewind, no re-ingest, no resets. The KV tape grows strictly
left-to-right through the whole conversation. Programs accumulate as
tools the model can invoke in later turns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from orate.body_grammar import (
    ArgType,
    BodyGrammarError,
    derive_body_grammar_rules,
    derive_call_arg_types,
)
from orate.gen import Bool, Choice, Gen, Int, String, ToolCall
from orate.meta import (
    PROGRAM_SOURCE_GRAMMAR,
    MetaProgramInvalid,
    compile_program_source,
    validate_program_source,
)
from orate.program import ProgramInvocation

# ---- events --------------------------------------------------------------


@dataclass(frozen=True)
class FreeText:
    text: str


@dataclass(frozen=True)
class ProgramInvoked:
    name: str
    args: tuple | dict  # tuple of positional emitted values, or dict for make_new_program
    result: Any = None
    source: str | None = None  # populated only for make_new_program


@dataclass(frozen=True)
class NewProgramRegistered:
    name: str
    source: str


@dataclass(frozen=True)
class TurnEnded:
    reason: str  # "ends_turn" | "eos" | "budget" | "no_progress"


Event = FreeText | ProgramInvoked | NewProgramRegistered | TurnEnded


# ---- session -------------------------------------------------------------


DEFAULT_MODE = "default"


@dataclass
class _RegistryEntry:
    name: str
    fn: Callable[..., ProgramInvocation]
    body_grammar_rules: dict[str, str]  # name → body-rule text (includes helpers)
    root_rule_name: str  # e.g. "foo_body"
    ends_turn: bool = False
    mode_transition: str | None = None
    # ``mode`` controls visibility: a program is callable only when the
    # session's active mode matches. ``None`` means "available in every
    # mode" — used for cross-cutting tools like make_new_program.
    mode: str | None = None
    # ``arg_types`` lets _parse_args coerce raw grammar fragments back
    # into native Python values. Empty list when typing isn't available
    # (e.g. make_new_program, which is hand-handled).
    arg_types: list[ArgType] = field(default_factory=list)


# One-unit outer grammar: either a free-text chunk (no '@') or exactly one
# @call. Session.advance() runs this in a loop, processing each chunk.
#
# Rationale: sampling the full (text | call)* in ONE matcher run requires
# introspecting grammar state mid-stream to detect call completions. One
# unit per sample is simpler and lets the runtime hook cleanly between
# units to register new programs, rebuild the grammar, and emit events.
_MAX_TEXT_CHUNK = 240


def _build_outer_grammar(
    registry: dict[str, _RegistryEntry],
    *,
    allow_free_text: bool = True,
) -> str:
    """One-unit outer grammar admitting text OR one @call.

    Every registered program contributes:
      1. A top-level alternative in `at_call` that invokes its body rule.
      2. Its body rule + helpers (namespaced under the program name).

    When ``allow_free_text=False``, the outer grammar is just ``at_call``
    — the model can ONLY emit tool calls. Useful for tool-strict sessions
    (e.g. legal-step demos) where any prose at all derails the trace.
    The Session driver still inserts a ``"\\n"`` separator between
    successive calls so they stay readable in the KV.

    The grammar is regenerated each time the registry or active mode
    changes.
    """
    at_call_alts: list[str] = []
    all_body_rules: list[str] = []

    seen_rules: set[str] = set()
    for name, entry in registry.items():
        at_call_alts.append(f'"@{name}(" {entry.root_rule_name} ")"')
        # derive_body_grammar_rules returns {rule_name: "rule_name ::= body"}
        # already in full GBNF form. Dedupe by rule name in case two programs
        # happen to share a helper rule (the per-program prefix usually makes
        # this rare, but integer helpers for equivalent ranges can collide).
        for rule_name, full_rule in entry.body_grammar_rules.items():
            if rule_name in seen_rules:
                continue
            seen_rules.add(rule_name)
            all_body_rules.append(full_rule)

    if not at_call_alts:
        raise RuntimeError("Session has empty registry — at least one program required")

    if not allow_free_text:
        header = [
            "root ::= at_call",
            f"at_call ::= {' | '.join(at_call_alts)}",
        ]
        return "\n".join(header + all_body_rules) + "\n"

    # text_chunk: 1..MAX printable non-'@' chars. Bounded so the grammar
    # yields control back per-unit. The Session driver loops to produce
    # longer texts.
    text_chunk_rule = (
        "text_chunk ::= text_char (text_char (text_char"
        + " text_char?" * (_MAX_TEXT_CHUNK - 3)
        + ")?)?"
    )
    # More practical unrolling: GBNF supports '*' via recursion; use that.
    text_chunk_rule = (
        "text_chunk ::= text_char text_chunk_rest\n"
        'text_chunk_rest ::= text_char text_chunk_rest | ""'
    )
    text_char_rule = "text_char ::= [ !#-?A-~\\t\\n]"  # printable ASCII minus '@'

    header = [
        "root ::= text_chunk | at_call",
        f"at_call ::= {' | '.join(at_call_alts)}",
        text_chunk_rule,
        text_char_rule,
    ]
    return "\n".join(header + all_body_rules) + "\n"


_CALL_RE = re.compile(r"^@([a-z_][a-z0-9_]*)\((.*)\)\s*$", re.DOTALL)


def _parse_at_call(raw: str) -> tuple[str, str]:
    """Split '@foo(args)' into (name, args_text)."""
    m = _CALL_RE.match(raw.strip())
    if not m:
        raise ValueError(f"malformed @call in session output: {raw!r}")
    return m.group(1), m.group(2)


def _serialize_result(value: Any) -> str:
    """Text-serialize a call result for back-appending into the KV.

    Short, marked so the model knows it was a tool output not its own text.
    Dicts / lists → JSON. Primitives → str().
    """
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _check_value_against_spec(spec: Gen, value: object, idx: int) -> str | None:
    """Verify a single parsed value against its Gen spec.

    Returns ``None`` on success, or a short error string identifying
    the failure. Used by Session._verify_program_emission to enforce
    predicates the call-site grammar can't capture (cross-yield
    closures, opaque ``where=`` lambdas).
    """
    if isinstance(spec, Choice):
        if value not in spec.options:
            return (
                f"yield #{idx}: {value!r} is not one of "
                f"{list(spec.options)!r}"
            )
        if spec.where is not None and not _safe_predicate(spec.where, value):
            return f"yield #{idx}: choice {value!r} failed where= predicate"
        return None
    if isinstance(spec, Int):
        if not isinstance(value, int) or isinstance(value, bool):
            return f"yield #{idx}: expected int, got {type(value).__name__}"
        if value < spec.min_val or value > spec.max_val:
            return (
                f"yield #{idx}: {value} outside "
                f"[{spec.min_val}, {spec.max_val}]"
            )
        if spec.where is not None and not _safe_predicate(spec.where, value):
            return f"yield #{idx}: integer {value} failed where= predicate"
        return None
    if isinstance(spec, Bool):
        if not isinstance(value, bool):
            return f"yield #{idx}: expected bool, got {type(value).__name__}"
        if spec.where is not None and not _safe_predicate(spec.where, value):
            return f"yield #{idx}: boolean {value} failed where= predicate"
        return None
    if isinstance(spec, String):
        if not isinstance(value, str):
            return f"yield #{idx}: expected str, got {type(value).__name__}"
        if spec.where is not None and not _safe_predicate(spec.where, value):
            return f"yield #{idx}: string {value!r} failed where= predicate"
        return None
    if isinstance(spec, ToolCall):
        # ToolCalls don't surface at the call-site grammar; skip.
        return None
    return None  # unknown Gen subclass — be lenient


def _safe_predicate(pred: Callable[[object], object], value: object) -> bool:
    """Run a where= predicate, returning False on any exception.

    Predicate authors aren't expected to defensively guard against
    parse failures (sympy.SympifyError, IndexError, …); the verifier
    should treat any raise as "rejected".
    """
    try:
        return bool(pred(value))
    except Exception:  # noqa: BLE001
        return False


def _scan_typed_args(text: str, arg_types: list[ArgType]) -> tuple:
    """Walk ``text`` left-to-right consuming one arg per ArgType.

    Driven by types: each kind has its own scanner that knows when its
    representation ends. ``", "`` separates args. We don't allow extra
    whitespace; the grammar emits exactly the shape we expect.

    Returns a tuple of native Python values matching ``arg_types`` in
    order. Raises ``ValueError`` on shape mismatch with a position hint.
    """
    pos = 0
    n = len(text)
    values: list[Any] = []
    for idx, t in enumerate(arg_types):
        if idx > 0:
            if text[pos : pos + 2] != ", ":
                raise ValueError(
                    f"expected ', ' before arg #{idx} at pos {pos} in {text!r}"
                )
            pos += 2

        if t.kind == "integer":
            start = pos
            if pos < n and text[pos] == "-":
                pos += 1
            digit_start = pos
            while pos < n and text[pos].isdigit():
                pos += 1
            if pos == digit_start:
                raise ValueError(f"expected digits for arg #{idx} at pos {start}")
            values.append(int(text[start:pos]))
        elif t.kind == "boolean":
            if text.startswith("true", pos):
                values.append(True)
                pos += 4
            elif text.startswith("false", pos):
                values.append(False)
                pos += 5
            else:
                raise ValueError(f"expected true/false for arg #{idx} at pos {pos}")
        elif t.kind == "choice":
            # Choices are emitted bare per the body grammar
            # ((alt1 | alt2 | ...)). Match the longest option that
            # starts at pos.
            matched: str | None = None
            for opt in sorted(t.options, key=len, reverse=True):
                if text.startswith(opt, pos):
                    matched = opt
                    break
            if matched is None:
                raise ValueError(
                    f"arg #{idx}: expected one of {list(t.options)} at pos {pos}"
                )
            values.append(matched)
            pos += len(matched)
        elif t.kind == "string":
            if pos >= n or text[pos] != '"':
                raise ValueError(
                    f'expected \'"\' opening quote for arg #{idx} at pos {pos}'
                )
            pos += 1  # consume opening quote
            content_start = pos
            buf: list[str] = []
            while pos < n and text[pos] != '"':
                if text[pos] == "\\" and pos + 1 < n:
                    nxt = text[pos + 1]
                    buf.append('"' if nxt == '"' else "\\" if nxt == "\\" else nxt)
                    pos += 2
                else:
                    buf.append(text[pos])
                    pos += 1
            if pos >= n:
                raise ValueError(
                    f"unterminated string for arg #{idx} starting at pos {content_start}"
                )
            pos += 1  # consume closing quote
            values.append("".join(buf))
        else:
            raise ValueError(f"unknown arg kind {t.kind!r} at index {idx}")

    if pos != n:
        raise ValueError(f"trailing unparsed content at pos {pos} in {text!r}")
    return tuple(values)


class Session:
    """Persistent-KV conversational session with tool accumulation.

    Modes
    -----
    A session has an active *mode* (default: ``"default"``). Each
    registered program is either *unscoped* (visible in every mode) or
    *mode-scoped* (visible only when the active mode matches). The
    outer grammar is rebuilt to expose only the visible subset whenever
    the mode changes.

    Mode transitions are program-driven: a program decorated with
    ``@program(mode_transition="combat")`` flips the session into the
    ``"combat"`` mode after a successful invocation, so the next sample
    is taken under a grammar built from the combat-mode programs only.
    Pair with ``mode_transition="default"`` (or whatever name) on an
    ``exit_*`` program to return.
    """

    def __init__(
        self,
        engine: Any,
        *,
        programs: dict[str, Callable[..., ProgramInvocation]] | None = None,
        system: str = "",
        max_turn_tokens: int = 1024,
        max_consecutive_synth_failures: int = 3,
        max_calls_per_turn: int = 4,
        allow_free_text: bool = True,
        call_separator: str = "\n",
    ) -> None:
        if not (hasattr(engine, "begin_session") and hasattr(engine, "sample_under")):
            raise TypeError(
                f"engine {type(engine).__name__} does not support session mode; "
                "need begin_session() and sample_under()"
            )
        self.engine = engine
        self.max_turn_tokens = max_turn_tokens
        self.max_consecutive_synth_failures = max_consecutive_synth_failures
        self._consecutive_synth_failures = 0
        self.max_calls_per_turn = max_calls_per_turn
        self.allow_free_text = allow_free_text
        self.call_separator = call_separator
        self.registry: dict[str, _RegistryEntry] = {}
        self._active_mode: str = DEFAULT_MODE

        # Bootstrap: register make_new_program built-in, plus any seeds.
        self._register_make_new_program()
        for name, fn in (programs or {}).items():
            if name == "make_new_program":
                continue  # already bootstrapped
            self.register(name, fn)

        self._rebuild_outer_grammar()
        self.engine.begin_session(system)
        # Pay XGrammar's first-compile JIT cost up front and pre-populate
        # the engine's grammar cache with the current outer grammar.
        # Without this, the first sample_under after the mode switch
        # eats a one-time ~10x penalty (see bench/results/legal_steps_*).
        # Engines that don't expose .warm() (e.g. MockEngine) fall through.
        self._engine_warm([self._outer_grammar])
        self.transcript: list[Event] = []

    def _engine_warm(self, grammars: list[str]) -> None:
        """Trigger the engine's grammar-compile cache for ``grammars``.

        No-op on engines that don't expose ``warm()``. Safe to call any
        time after construction.
        """
        if hasattr(self.engine, "warm"):
            self.engine.warm(grammars)

    # ---- registration ---------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[..., ProgramInvocation],
        *,
        mode: str | None = None,
    ) -> None:
        """Add a @program to the registry, rebuild the outer grammar.

        ``mode=None`` means the program is visible in every mode (the
        default — appropriate for cross-cutting tools). Pass an explicit
        mode name to scope the program: it becomes callable only when
        ``self._active_mode`` matches.
        """
        rules = derive_body_grammar_rules(fn)  # dict of {rule_name: rule_body}
        try:
            arg_types = derive_call_arg_types(fn)
        except BodyGrammarError:
            arg_types = []
        # Root rule is conventionally "<program_fn.__name__>_body".
        fn_name = getattr(fn, "__name__", name)
        root = f"{fn_name}_body"
        if root not in rules:
            raise BodyGrammarError(
                f"derive_body_grammar_rules for {name!r} did not produce expected root rule "
                f"{root!r} (got keys {sorted(rules.keys())})"
            )
        # Prefer attributes stashed on the wrapper at decoration time;
        # fall back to invoking once if needed (some legacy registrations
        # pass bare functions). The wrapper stash is cheaper and avoids
        # surprise side effects from a no-arg construction call.
        ends_turn = bool(getattr(fn, "__orate_ends_turn__", False))
        mode_transition = getattr(fn, "__orate_mode_transition__", None)
        if not ends_turn and callable(fn):
            try:
                inv = fn()
                ends_turn = bool(getattr(inv, "ends_turn", False))
                mode_transition = mode_transition or getattr(inv, "mode_transition", None)
            except TypeError:
                pass  # program takes args; nothing more to learn
        self.registry[name] = _RegistryEntry(
            name=name,
            fn=fn,
            body_grammar_rules=rules,
            root_rule_name=root,
            ends_turn=ends_turn,
            mode_transition=mode_transition,
            mode=mode,
            arg_types=arg_types,
        )
        self._rebuild_outer_grammar()
        # The outer grammar has changed; warm the cache so the next
        # sample_under doesn't pay compile time.
        self._engine_warm([self._outer_grammar])

    def set_mode(self, mode: str) -> None:
        """Switch the session's active mode and rebuild the outer grammar.

        Programs whose ``mode`` is ``None`` remain visible. Programs
        scoped to ``mode`` become visible; programs scoped to other
        modes drop out of the grammar until the session returns to them.
        """
        self._active_mode = mode
        self._rebuild_outer_grammar()
        self._engine_warm([self._outer_grammar])

    @property
    def active_mode(self) -> str:
        return self._active_mode

    def _visible_registry(self) -> dict[str, _RegistryEntry]:
        return {
            name: entry
            for name, entry in self.registry.items()
            if entry.mode is None or entry.mode == self._active_mode
        }

    def _rebuild_outer_grammar(self) -> None:
        self._outer_grammar = _build_outer_grammar(
            self._visible_registry(),
            allow_free_text=self.allow_free_text,
        )

    def _register_make_new_program(self) -> None:
        """Bootstrap: make_new_program is grammar-inlined but dispatched
        specially by the runtime (source synthesis happens in a sub-sample
        under PROGRAM_SOURCE_GRAMMAR, not via derive_body_grammar).
        """
        # make_new_program's call-site grammar: two string literals (name, task).
        # Rules are in the same "name ::= body" format derive_body_grammar_rules
        # returns, so _build_outer_grammar can consume them uniformly.
        rules = {
            "make_new_program_body": 'make_new_program_body ::= mnp_str ", " mnp_str',
            "mnp_str": 'mnp_str ::= "\\"" mnp_char+ "\\""',
            "mnp_char": "mnp_char ::= [a-zA-Z0-9 \\-_.]",
        }
        self.registry["make_new_program"] = _RegistryEntry(
            name="make_new_program",
            fn=lambda: None,  # never invoked via run()
            body_grammar_rules=rules,
            root_rule_name="make_new_program_body",
            ends_turn=False,
        )

    # ---- user messages --------------------------------------------------

    def user(self, text: str) -> None:
        """Append a user message to the session transcript + KV."""
        self.engine.append(f"\n<|user|>\n{text}\n<|assistant|>\n")

    # ---- the turn loop --------------------------------------------------

    def advance(self) -> Iterator[Event]:
        """Generate one assistant turn until an end condition fires.

        Yields events live — callers can act on them as they arrive.
        """
        tokens_used = 0
        no_progress_count = 0
        calls_this_turn = 0
        while tokens_used < self.max_turn_tokens:
            remaining = self.max_turn_tokens - tokens_used
            chunk = self.engine.sample_under(
                self._outer_grammar,
                max_tokens=min(remaining, 256),
            )
            if not chunk:
                no_progress_count += 1
                if no_progress_count >= 2:
                    ev = TurnEnded("no_progress")
                    self.transcript.append(ev)
                    yield ev
                    return
                continue
            no_progress_count = 0
            tokens_used += len(chunk) // 3  # rough byte→token estimate

            if chunk.lstrip().startswith("@"):
                ev_or_done = self._handle_call(chunk)
                for ev in ev_or_done:
                    yield ev
                    self.transcript.append(ev)
                    if isinstance(ev, TurnEnded):
                        return
                calls_this_turn += 1
                if calls_this_turn >= self.max_calls_per_turn:
                    ev = TurnEnded("max_calls")
                    self.transcript.append(ev)
                    yield ev
                    return
            else:
                ev = FreeText(chunk)
                self.transcript.append(ev)
                yield ev

        ev = TurnEnded("budget")
        self.transcript.append(ev)
        yield ev

    def _handle_call(self, raw: str) -> list[Event]:
        """Process a single @call chunk. Returns the events to yield."""
        try:
            name, args_text = _parse_at_call(raw)
        except ValueError as e:
            # Malformed — append a note to KV so the model knows, continue.
            self.engine.append(f"\n[session: ignored malformed call: {e}]\n")
            return []

        if name == "make_new_program":
            if self._consecutive_synth_failures >= self.max_consecutive_synth_failures:
                self.engine.append("\n[session: too many failed synthesis attempts; ending turn]\n")
                return [
                    ProgramInvoked(
                        name="make_new_program",
                        args={"raw": args_text},
                        result={"error": "synth_budget_exhausted"},
                    ),
                    TurnEnded("synth_budget_exhausted"),
                ]
            return self._handle_make_new_program(args_text)

        if name not in self.registry:
            self.engine.append(f"\n[session: unknown program {name!r}]\n")
            return []

        entry = self.registry[name]
        # Visibility check: a mode-scoped program shouldn't be in the
        # outer grammar at all when its mode isn't active, but we
        # double-check defensively in case the grammar leaked.
        if entry.mode is not None and entry.mode != self._active_mode:
            self.engine.append(
                f"\n[session: program {name!r} not callable in mode {self._active_mode!r}]\n"
            )
            return []

        try:
            args = self._parse_args(args_text, entry.arg_types)
        except ValueError as e:
            self.engine.append(f"\n[session: arg parse failed for {name}: {e}]\n")
            return []

        # Predicate verification: re-run the program body against the
        # parsed args, checking each yield's where=/options/range. If
        # the model emitted a syntactically-valid but semantically-bad
        # call (e.g. a "simplify" that isn't equivalent), reject and
        # let the model try again on the next sample.
        verify_error = self._verify_program_emission(entry.fn, args)
        if verify_error is not None:
            self.engine.append(
                f"\n[session: rejected — {verify_error}. Retry the call.]\n"
            )
            return [
                ProgramInvoked(
                    name=name,
                    args=args,
                    result={"rejected": True, "error": verify_error},
                )
            ]

        result = {"emitted_args": args_text, "parsed": args}
        self.engine.append(f" → {_serialize_result(result)}\n")

        events: list[Event] = [ProgramInvoked(name=name, args=args, result=result)]

        # Apply mode transition (if any) before deciding whether to end
        # the turn — a transition + ends_turn together mean "client gets
        # the args, and the next turn starts in the new mode."
        if entry.mode_transition is not None:
            self.set_mode(entry.mode_transition)
            self.engine.append(
                f"\n[session: mode → {entry.mode_transition!r}]\n"
            )

        if entry.ends_turn:
            events.append(TurnEnded("ends_turn"))
        return events

    def _verify_program_emission(
        self,
        fn: Callable[..., ProgramInvocation],
        parsed_args: tuple,
    ) -> str | None:
        """Drive the @program's body against ``parsed_args``, predicate-checking.

        Returns ``None`` on success, or a human-readable error describing
        which yield rejected which value. The body is run *as a generator*
        — at each yield we inspect the ``Gen`` spec, check the
        corresponding parsed arg against options/range/predicate, then
        send the parsed value back so subsequent yields' closures see it.

        Programs with parameters (no zero-arg invocation possible) are
        skipped — verification is opt-in via shape, and the demo
        programs are all parameterless.
        """
        try:
            invocation = fn()
        except TypeError:
            return None  # program has params; nothing to verify here

        body_iter = invocation.body(*invocation.args, **invocation.kwargs)
        sent: object = None
        idx = 0
        while True:
            try:
                spec = (
                    body_iter.send(sent) if sent is not None else next(body_iter)
                )
            except StopIteration:
                break
            if isinstance(spec, ProgramInvocation):
                # Flavor-B sub-program yields aren't materialised in the
                # outer call-site grammar; skip with an empty send.
                sent = None
                continue
            if not isinstance(spec, Gen):
                # Verifier yields, etc. — pass through.
                sent = None
                continue
            if idx >= len(parsed_args):
                return f"yield #{idx} has no parsed value to check"
            value = parsed_args[idx]
            err = _check_value_against_spec(spec, value, idx)
            if err is not None:
                return err
            sent = value
            idx += 1
        return None

    def _handle_make_new_program(self, args_text: str) -> list[Event]:
        """The bootstrap: switch grammars, sample source, register, continue."""
        # Sub-sample under PROGRAM_SOURCE_GRAMMAR on the same KV. The model
        # was just asked to call @make_new_program(name, task); it now
        # completes the call by emitting the program body. The emitted
        # source is appended to KV naturally since sample_under persists.
        self.engine.append("\n[session: synthesizing program…]\n")
        source = self.engine.sample_under(
            PROGRAM_SOURCE_GRAMMAR,
            max_tokens=2048,
        )
        errors = validate_program_source(source)
        if errors:
            msg = "; ".join(errors[:3])
            self._consecutive_synth_failures += 1
            self.engine.append(f"\n[session: synthesis rejected: {msg}]\n")
            return [
                ProgramInvoked(
                    name="make_new_program",
                    args={"raw": args_text},
                    result={"error": msg, "source": source},
                    source=source,
                )
            ]
        try:
            compiled = compile_program_source(source)
        except MetaProgramInvalid as e:
            self._consecutive_synth_failures += 1
            self.engine.append(f"\n[session: compile failed: {e}]\n")
            return [
                ProgramInvoked(
                    name="make_new_program",
                    args={"raw": args_text},
                    result={"error": str(e), "source": source},
                    source=source,
                )
            ]

        fn_name = compiled.__name__
        # Guard against collisions + the special name.
        if fn_name == "make_new_program" or fn_name in self.registry:
            self.engine.append(
                f"\n[session: program name {fn_name!r} is reserved or already registered]\n"
            )
            return [
                ProgramInvoked(
                    name="make_new_program",
                    args={"raw": args_text},
                    result={"error": "name collision", "name": fn_name},
                    source=source,
                )
            ]

        try:
            self.register(fn_name, compiled)
        except BodyGrammarError as e:
            self._consecutive_synth_failures += 1
            self.engine.append(f"\n[session: body grammar derivation failed: {e}]\n")
            return [
                ProgramInvoked(
                    name="make_new_program",
                    args={"raw": args_text},
                    result={"error": str(e), "source": source},
                    source=source,
                )
            ]

        self._consecutive_synth_failures = 0  # reset on successful registration
        self.engine.append(f"\n[session: registered @{fn_name}; grammar rebuilt]\n")
        return [
            NewProgramRegistered(name=fn_name, source=source),
            ProgramInvoked(
                name="make_new_program",
                args={"raw": args_text},
                result={"registered": fn_name},
                source=source,
            ),
        ]

    def _parse_args(self, args_text: str, arg_types: list[ArgType]) -> tuple:
        """Decode the emitted arg blob into a typed tuple.

        When ``arg_types`` is supplied (the common path: any program
        registered via ``Session.register`` carries them), each arg is
        scanned according to its kind:

        * ``integer`` — leading ``-?`` then digits, parsed via ``int``.
        * ``boolean`` — literal ``true``/``false``.
        * ``string`` / ``choice`` — JSON-style ``"..."`` with ``\\\\`` and
          ``\\"`` escapes.

        The body grammar guarantees that args are emitted in this exact
        shape, separated by ``", "``. With ``arg_types`` we can handle
        string content containing a literal ``", "`` correctly — naive
        splitting on ``", "`` would have torn it apart.

        When ``arg_types`` is empty (legacy path, or the special
        make_new_program entry), fall back to the shallow split.
        """
        if not args_text:
            return ()
        if not arg_types:
            return tuple(part.strip() for part in args_text.split(", "))
        return _scan_typed_args(args_text, arg_types)
