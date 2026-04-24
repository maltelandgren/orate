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
from dataclasses import dataclass
from typing import Any

from orate.body_grammar import BodyGrammarError, derive_body_grammar_rules
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


@dataclass
class _RegistryEntry:
    name: str
    fn: Callable[..., ProgramInvocation]
    body_grammar_rules: dict[str, str]  # name → body-rule text (includes helpers)
    root_rule_name: str  # e.g. "foo_body"
    ends_turn: bool = False


# One-unit outer grammar: either a free-text chunk (no '@') or exactly one
# @call. Session.advance() runs this in a loop, processing each chunk.
#
# Rationale: sampling the full (text | call)* in ONE matcher run requires
# introspecting grammar state mid-stream to detect call completions. One
# unit per sample is simpler and lets the runtime hook cleanly between
# units to register new programs, rebuild the grammar, and emit events.
_MAX_TEXT_CHUNK = 240


def _build_outer_grammar(registry: dict[str, _RegistryEntry]) -> str:
    """One-unit outer grammar admitting text OR one @call.

    Every registered program contributes:
      1. A top-level alternative in `at_call` that invokes its body rule.
      2. Its body rule + helpers (namespaced under the program name).

    The grammar is regenerated each time the registry changes.
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


class Session:
    """Persistent-KV conversational session with tool accumulation."""

    def __init__(
        self,
        engine: Any,
        *,
        programs: dict[str, Callable[..., ProgramInvocation]] | None = None,
        system: str = "",
        max_turn_tokens: int = 1024,
        max_consecutive_synth_failures: int = 3,
        max_calls_per_turn: int = 4,
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
        self.registry: dict[str, _RegistryEntry] = {}

        # Bootstrap: register make_new_program built-in, plus any seeds.
        self._register_make_new_program()
        for name, fn in (programs or {}).items():
            if name == "make_new_program":
                continue  # already bootstrapped
            self.register(name, fn)

        self._outer_grammar = _build_outer_grammar(self.registry)
        self.engine.begin_session(system)
        self.transcript: list[Event] = []

    # ---- registration ---------------------------------------------------

    def register(self, name: str, fn: Callable[..., ProgramInvocation]) -> None:
        """Add a @program to the registry, rebuild the outer grammar."""
        rules = derive_body_grammar_rules(fn)  # dict of {rule_name: rule_body}
        # Root rule is conventionally "<program_fn.__name__>_body".
        fn_name = getattr(fn, "__name__", name)
        root = f"{fn_name}_body"
        if root not in rules:
            raise BodyGrammarError(
                f"derive_body_grammar_rules for {name!r} did not produce expected root rule "
                f"{root!r} (got keys {sorted(rules.keys())})"
            )
        ends_turn = getattr(fn(), "ends_turn", False) if callable(fn) else False
        self.registry[name] = _RegistryEntry(
            name=name,
            fn=fn,
            body_grammar_rules=rules,
            root_rule_name=root,
            ends_turn=ends_turn,
        )
        self._outer_grammar = _build_outer_grammar(self.registry)

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
        # Args come pre-parsed by the body grammar — for simple cases we
        # just echo them as a tuple. Richer arg deserialization (typed
        # tuples, dicts) is a v2 extension.
        args = self._parse_args(args_text)
        result = {"emitted_args": args_text, "parsed": args}
        self.engine.append(f" → {_serialize_result(result)}\n")

        events: list[Event] = [ProgramInvoked(name=name, args=args, result=result)]
        if entry.ends_turn:
            events.append(TurnEnded("ends_turn"))
        return events

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

    def _parse_args(self, args_text: str) -> tuple:
        """Split the emitted arg blob into a tuple of positional values.

        The body grammar emits fields separated by ``, ``. This is a
        shallow split; a v2 parser would honor the per-field types
        (int vs str vs bool) but here we keep all fields as strings.
        """
        if not args_text:
            return ()
        return tuple(part.strip() for part in args_text.split(", "))
