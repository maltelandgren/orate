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
    scan_typed_args,
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
    body_grammar: str = ""  # self-contained GBNF for this leaf's body alone
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


# Outer grammar is just a prefix-alternation: ``"@a(" | "@b(" | ...``,
# optionally with a free-text alternative. Each leaf's actual body
# grammar is compiled separately and stored on its _RegistryEntry; the
# Session driver invokes ``engine.sample_under(entry.body_grammar)`` as
# a *second* sample call after the outer matcher accepts on a prefix.
#
# This is how ``make_new_program`` already works (one sample for the
# args, then a separate sample under PROGRAM_SOURCE_GRAMMAR for the
# synthesised source). Generalising that pattern to every leaf gives us:
#   - small, self-contained grammars per leaf
#   - the outer grammar is a thin alternation of prefixes; cheap to
#     recompile when the registry mutates (e.g. make_new_program adds a
#     leaf, or a mode switch changes which subset is visible)
#   - no cross-leaf inlining; one leaf's grammar can't accidentally
#     interfere with another's
#
# The trade-off is a tiny bit of driver complexity in advance(): on a
# prefix accept, we read the program name from the chunk, look up its
# body grammar, sample it, then synthesise the closing ``)``. See
# ``Session.advance`` for the loop.


def _build_outer_grammar(
    registry: dict[str, _RegistryEntry],
    *,
    allow_free_text: bool = True,
) -> str:
    """Prefix-alternation outer grammar.

    For each visible leaf ``foo``, contributes a single alternative
    ``"@foo("`` to the root. The matcher accepts the moment a full
    prefix has been consumed; the body is sampled separately under
    ``entry.body_grammar``.

    When ``allow_free_text=False`` the outer admits only prefixes —
    the model's every emission is a tool call. Defaults to True for
    narrative-style sessions; legal-step / agent-strict demos pass
    False.
    """
    if not registry:
        raise RuntimeError("Session has empty registry — at least one program required")

    prefix_alts = " | ".join(f'"@{name}("' for name in registry)

    if not allow_free_text:
        return f"root ::= {prefix_alts}\n"

    text_rules = (
        "text_chunk ::= text_char text_chunk_rest\n"
        'text_chunk_rest ::= text_char text_chunk_rest | ""\n'
        "text_char ::= [ !#-?A-~\\t\\n]\n"  # printable ASCII minus '@'
    )
    return f"root ::= text_chunk | {prefix_alts}\n{text_rules}"


def _build_body_grammar(entry: _RegistryEntry) -> str:
    """Self-contained GBNF for one leaf's body.

    Adds a ``root ::= <name>_body`` line on top of the leaf's body
    rules so the result can be passed directly to
    ``engine.sample_under``. Each leaf's grammar is compiled once at
    registration time.
    """
    rules = list(entry.body_grammar_rules.values())
    return f"root ::= {entry.root_rule_name}\n" + "\n".join(rules) + "\n"


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


# Re-export the typed scanner for backwards compat with tests that
# imported the leading-underscore name. ``body_grammar.scan_typed_args``
# is the canonical home (it's the inverse of body grammar derivation).
_scan_typed_args = scan_typed_args


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
        """Add a leaf @program to the registry, rebuild the outer grammar.

        ``mode=None`` means the program is visible in every mode (the
        default — appropriate for cross-cutting tools). Pass an explicit
        mode name to scope the program: it becomes callable only when
        ``self._active_mode`` matches.

        The leaf's body grammar is compiled once here and stored on its
        registry entry. The outer grammar (rebuilt every call) is just
        the prefix-alternation; it never inlines body rules.
        """
        # Composers (@program(invocable=False)) don't have a call-site
        # grammar; they're not registerable. Catch this early with a
        # clear message rather than letting body_grammar_rules raise.
        if getattr(fn, "__orate_invocable__", True) is False:
            raise BodyGrammarError(
                f"register({name!r}): {getattr(fn, '__name__', fn)!r} is a composer "
                "(@program(invocable=False)). Composers orchestrate leaves and run "
                "directly via .run(engine=...); they are not registered into the "
                "session's outer grammar."
            )
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
        entry = _RegistryEntry(
            name=name,
            fn=fn,
            body_grammar_rules=rules,
            root_rule_name=root,
            ends_turn=ends_turn,
            mode_transition=mode_transition,
            mode=mode,
            arg_types=arg_types,
        )
        entry.body_grammar = _build_body_grammar(entry)
        self.registry[name] = entry
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
        """Bootstrap: make_new_program is just another leaf for the outer
        grammar (its prefix appears in the alternation), but its dispatch
        is special — after parsing the (name, task) body, the runtime
        runs a separate sample under PROGRAM_SOURCE_GRAMMAR to synthesise
        the new @program's source.

        From the outer driver's POV, this is identical to any other leaf:
        sample prefix, then sample body, then ``)``. The "different
        thing happens" is in ``_dispatch`` where we recognise the
        program name and run the synthesis path.
        """
        rules = {
            "make_new_program_body": 'make_new_program_body ::= mnp_str ", " mnp_str',
            "mnp_str": 'mnp_str ::= "\\"" mnp_char+ "\\""',
            "mnp_char": "mnp_char ::= [a-zA-Z0-9 \\-_.]",
        }
        entry = _RegistryEntry(
            name="make_new_program",
            fn=lambda: None,  # never invoked via run()
            body_grammar_rules=rules,
            root_rule_name="make_new_program_body",
            ends_turn=False,
        )
        entry.body_grammar = _build_body_grammar(entry)
        self.registry["make_new_program"] = entry

    # ---- user messages --------------------------------------------------

    def user(self, text: str) -> None:
        """Append a user message to the session transcript + KV."""
        self.engine.append(f"\n<|user|>\n{text}\n<|assistant|>\n")

    # ---- the turn loop --------------------------------------------------

    def advance(self) -> Iterator[Event]:
        """Generate one assistant turn until an end condition fires.

        Each loop iteration:

          1. Sample under the outer (prefix-only) grammar. Result is
             either a free-text chunk or a prefix ``"@<name>("``.
          2. If text: yield FreeText, continue.
          3. If prefix: look up the leaf, sample under its body grammar
             (a separate engine.sample_under call against its own
             matcher), append the closing ``)``, dispatch the call.

        This is the same shape ``make_new_program`` already used (sample
        prefix → sample body → run synthesis). Now every leaf works
        identically: the body grammar is per-leaf and self-contained,
        the outer grammar is just a thin alternation of prefixes that
        recompiles cheaply when the registry mutates.

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
            tokens_used += max(1, len(chunk) // 3)

            name = self._match_prefix(chunk)
            if name is None:
                # Free-text chunk.
                ev = FreeText(chunk)
                self.transcript.append(ev)
                yield ev
                continue

            # Prefix accepted. Sample the body under that leaf's own
            # grammar, then close the call with ``)``.
            entry = self.registry[name]
            body_text = self.engine.sample_under(
                entry.body_grammar,
                max_tokens=min(remaining, 512),
            )
            self.engine.append(")")
            tokens_used += max(1, len(body_text) // 3) + 1

            for ev in self._dispatch(name, body_text):
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

        ev = TurnEnded("budget")
        self.transcript.append(ev)
        yield ev

    def _match_prefix(self, chunk: str) -> str | None:
        """Return the leaf name if ``chunk`` is a recognised ``@name(`` prefix.

        With ``allow_free_text=True`` the outer grammar's root is
        ``text_chunk | prefix``; the matcher accepts either. We
        distinguish by structure: a prefix chunk is exactly
        ``@<name>(`` for a known leaf. Anything else is free text.
        """
        s = chunk.strip()
        if not (s.startswith("@") and s.endswith("(")):
            return None
        name = s[1:-1]
        if name not in self.registry:
            return None
        return name

    # ``_handle_call`` is preserved as a backwards-compat shim: tests
    # (and maybe external callers) feed in a fully-formed ``@name(args)``
    # string. We split it into (name, body_text) and delegate.
    def _handle_call(self, raw: str) -> list[Event]:
        """Process a fully-formed ``@name(args)`` chunk.

        Compatibility helper for tests that simulate model emissions.
        The advance() loop doesn't go through here — it samples the
        prefix and body separately and calls ``_dispatch`` directly.
        """
        try:
            name, body_text = _parse_at_call(raw)
        except ValueError as e:
            self.engine.append(f"\n[session: ignored malformed call: {e}]\n")
            return []
        return self._dispatch(name, body_text)

    def _dispatch(self, name: str, body_text: str) -> list[Event]:
        """Run the leaf's post-body logic: parse args, verify, emit events.

        Called after the engine has already produced the @-call's body
        (under that leaf's body grammar). ``body_text`` is what the
        body matcher emitted between the ``(`` and the ``)``.
        """
        if name == "make_new_program":
            if self._consecutive_synth_failures >= self.max_consecutive_synth_failures:
                self.engine.append("\n[session: too many failed synthesis attempts; ending turn]\n")
                return [
                    ProgramInvoked(
                        name="make_new_program",
                        args={"raw": body_text},
                        result={"error": "synth_budget_exhausted"},
                    ),
                    TurnEnded("synth_budget_exhausted"),
                ]
            return self._handle_make_new_program(body_text)

        if name not in self.registry:
            self.engine.append(f"\n[session: unknown program {name!r}]\n")
            return []

        entry = self.registry[name]
        # Visibility check: defensive — the grammar shouldn't expose a
        # mode-scoped program when its mode isn't active.
        if entry.mode is not None and entry.mode != self._active_mode:
            self.engine.append(
                f"\n[session: program {name!r} not callable in mode {self._active_mode!r}]\n"
            )
            return []

        try:
            args = self._parse_args(body_text, entry.arg_types)
        except ValueError as e:
            self.engine.append(f"\n[session: arg parse failed for {name}: {e}]\n")
            return []

        # Predicate verification: re-run the program body against the
        # parsed args, checking each yield's where=/options/range. Also
        # captures the body's ``return`` value — this is what client-
        # resolved tool calls (like ``@roll``) use to ship a result back
        # into the KV after predicate-checking the model's args.
        verify_error, returned = self._verify_program_emission(entry.fn, args)
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

        # If the program body returned a value, that's the resolved tool
        # result — surface it to the model and the client. Otherwise we
        # fall back to the (parsed-args, body-text) shape used by purely
        # declarative programs like @algebra_step.
        if returned is not None:
            result = returned
        else:
            result = {"emitted_args": body_text, "parsed": args}
        self.engine.append(f" → {_serialize_result(result)}\n")

        events: list[Event] = [ProgramInvoked(name=name, args=args, result=result)]

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
    ) -> tuple[str | None, Any]:
        """Drive the @program's body against ``parsed_args``, predicate-checking.

        Returns ``(error, return_value)``:

          * ``error`` — None on success, else a human-readable string
            describing which yield rejected which value.
          * ``return_value`` — the body's ``return`` value (captured from
            ``StopIteration.value``). This is the result the runtime
            injects back into the KV as the call's resolved tool result.
            Programs that don't return anything (or are skipped because
            of zero-arg-invocation TypeError) yield ``None`` here, in
            which case the caller falls back to the parsed-args result.

        The body is run *as a generator* — at each yield we inspect the
        ``Gen`` spec, check the corresponding parsed arg against
        options/range/predicate, then send the parsed value back so
        subsequent yields' closures see it. The body can keep running
        Python after its last yield; whatever it returns is the
        resolver's result. This is the load-bearing piece for true
        client-resolved tool calls — e.g. ``@roll`` predicate-checks
        the (skill, dc) args via yields, then computes ``random.randint``
        and returns ``{success: ..., d20: ...}``, which the model sees
        on the next sample.
        """
        try:
            invocation = fn()
        except TypeError:
            return None, None  # program has params; nothing to verify

        body_iter = invocation.body(*invocation.args, **invocation.kwargs)
        sent: object = None
        idx = 0
        return_value: Any = None
        while True:
            try:
                spec = (
                    body_iter.send(sent) if sent is not None else next(body_iter)
                )
            except StopIteration as stop:
                return_value = stop.value
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
                return f"yield #{idx} has no parsed value to check", None
            value = parsed_args[idx]
            err = _check_value_against_spec(spec, value, idx)
            if err is not None:
                return err, None
            sent = value
            idx += 1
        return None, return_value

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
