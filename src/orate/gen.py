from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from orate.engine.protocol import Engine


class GrammarExhausted(RuntimeError):
    """Raised when tightening has excluded every candidate in the accept set."""


class Gen:
    """Base for a yieldable generation spec.

    A @program body yields Gen instances at decision points. The runner
    calls dispatch(engine) on each yield; the returned value is sent
    back to the generator as the yield's result.

    Subclasses implement tightening-on-reject: when `where=` fails on a
    sample, the accept set is narrowed and the engine is re-queried.
    The narrowing is deterministic — no dice rolling to reach correctness.

    For finite domains (Choice, Int in practical ranges, Bool), the
    compile step in ``orate.compile`` pre-computes the exact accept set
    via witness enumeration before the engine sees the yield. The
    rejection loop is the honest fallback for when enumeration isn't
    feasible (unbounded strings, opaque predicates on huge ranges).
    """

    description: str | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 16

    def dispatch(self, engine: Engine) -> Any:
        raise NotImplementedError

    def _format_reject(self, value: Any) -> str | None:
        if self.reject_message is None:
            return None
        if callable(self.reject_message):
            return self.reject_message(value)
        return self.reject_message

    def _notify_reject(self, engine: Any, value: Any) -> None:
        """Phase-B: inject a natural-language reject hint into the engine's session."""
        msg = self._format_reject(value)
        if msg and hasattr(engine, "inject_context"):
            engine.inject_context(f"(note: {msg})")


@dataclass
class Choice(Gen):
    """Pick one of a fixed set of string options."""

    options: Sequence[str] = ()
    where: Callable[[str], bool] | None = None
    description: str | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 16

    def dispatch(self, engine: Engine) -> str:
        # Layer 1: witness-enumerate the accept set. For Choice this is
        # always enumerable; rejection-sampling never fires here.
        from orate.compile import enumerate_choice  # noqa: PLC0415

        accept = enumerate_choice(self.options, self.where)
        if not accept:
            raise GrammarExhausted(
                f"gen.choice: no option in {list(self.options)!r} satisfies predicate"
            )
        if len(accept) == 1:
            return accept[0]
        return engine.sample_choice(accept)


@dataclass
class Int(Gen):
    """Pick an integer in [min_val, max_val]."""

    min_val: int = 0
    max_val: int = 0
    where: Callable[[int], bool] | None = None
    description: str | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 16

    def dispatch(self, engine: Engine) -> int:
        if self.min_val > self.max_val:
            raise ValueError(f"gen.int: min={self.min_val} > max={self.max_val}")

        # Layer 1: witness-enumerate the accept set when the range is
        # small enough (default budget: 10k). For the common case this
        # eliminates the retry loop entirely — the engine sees only the
        # pre-filtered values and cannot emit a rejection.
        from orate.compile import enumerate_int  # noqa: PLC0415

        accept = enumerate_int(self.min_val, self.max_val, self.where)
        if accept is not None:
            if not accept:
                raise GrammarExhausted(
                    f"gen.int[{self.min_val},{self.max_val}]: no value satisfies predicate"
                )
            if len(accept) == 1:
                return accept[0]
            pick_str = engine.sample_choice([str(v) for v in accept])
            return int(pick_str)

        # Fallback: domain too large for enumeration. Use today's
        # rejection-sampling + tightening loop.
        excluded: set[int] = set()
        attempts = 0
        while attempts < self.max_retries:
            if len(excluded) >= self.max_val - self.min_val + 1:
                raise GrammarExhausted(
                    f"gen.int[{self.min_val},{self.max_val}]: all values excluded"
                )
            pick = engine.sample_int(self.min_val, self.max_val, excluded=excluded)
            if self.where is None or self.where(pick):
                return pick
            self._notify_reject(engine, pick)
            excluded.add(pick)
            attempts += 1
        raise GrammarExhausted(
            f"gen.int[{self.min_val},{self.max_val}]: max_retries={self.max_retries} exceeded"
        )


@dataclass
class String(Gen):
    """Free string with optional regex constraint and length cap.

    String domains are unbounded, so witness enumeration doesn't apply.
    The ``pattern=`` kwarg compiles directly to an XGrammar regex
    constraint (that's the "pattern → grammar" path from the plan —
    users ask for it explicitly rather than having it inferred from a
    ``where=`` lambda). For arbitrary ``where=`` predicates we fall back
    to rejection sampling with tightening and Phase-B context injection.
    """

    max_len: int = 256
    pattern: str | None = None
    where: Callable[[str], bool] | None = None
    description: str | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 8

    def __post_init__(self) -> None:
        if self.pattern is not None:
            re.compile(self.pattern)

    def dispatch(self, engine: Engine) -> str:
        seen: set[str] = set()
        attempts = 0
        while attempts < self.max_retries:
            pick = engine.sample_string(
                max_len=self.max_len,
                pattern=self.pattern,
                excluded=seen,
            )
            if self.where is None or self.where(pick):
                return pick
            self._notify_reject(engine, pick)
            seen.add(pick)
            attempts += 1
        raise GrammarExhausted(f"gen.string: max_retries={self.max_retries} exceeded")


@dataclass
class Bool(Gen):
    """Sample True or False."""

    where: Callable[[bool], bool] | None = None
    description: str | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 4

    def dispatch(self, engine: Engine) -> bool:
        # Layer 1: the domain has two values; always enumerable.
        from orate.compile import enumerate_bool  # noqa: PLC0415

        accept = enumerate_bool(self.where)
        if not accept:
            raise GrammarExhausted("gen.bool: no value satisfies predicate")
        if len(accept) == 1:
            return accept[0]
        # Both values are accepted; let the model's argmax pick.
        return engine.sample_bool()


@dataclass
class Struct(Gen):
    """Compound sugar: yield a dict of typed fields in one logical step.

    With a cross-field ``where=``, Struct applies forward-checking: as
    each field is bound, the remaining fields' domains are recompiled
    with the cross-field predicate closed over the bound values. This
    is the witness-enumeration primitive applied sequentially; no CSP
    solver required.
    """

    fields: dict[str, Gen] = field(default_factory=dict)
    where: Callable[[dict], bool] | None = None
    description: str | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 8

    def dispatch(self, engine: Engine) -> dict:
        # Fast path: no cross-field predicate → dispatch each field
        # independently (each field's own where= still tightens).
        if self.where is None:
            if hasattr(engine, "sample_struct"):
                return engine.sample_struct(self.fields)
            return {name: child.dispatch(engine) for name, child in self.fields.items()}

        # With a cross-field predicate, forward-check each field against
        # the bindings accumulated so far. For fields whose domain
        # can't be enumerated (e.g. String), the field dispatches
        # natively and the cross-field predicate is enforced at the end
        # via the struct-level rejection loop (backup behavior).
        from orate.compile import compile_struct_field  # noqa: PLC0415

        attempts = 0
        while attempts < self.max_retries:
            bound: dict[str, Any] = {}
            failed = False
            for name, spec in self.fields.items():
                # Attach field name for compile_struct_field's hint helper.
                import contextlib  # noqa: PLC0415

                with contextlib.suppress(Exception):
                    object.__setattr__(spec, "_field_name", name)
                accept = compile_struct_field(spec, bound, self.where)
                if accept is not None:
                    if not accept:
                        # No value for this field can satisfy the cross
                        # predicate given already-bound siblings. Treat
                        # as a struct-level rejection.
                        failed = True
                        break
                    if len(accept) == 1:
                        bound[name] = accept[0]
                    else:
                        # Let the engine pick among the feasible values.
                        pick = engine.sample_choice([str(v) for v in accept])
                        bound[name] = _coerce_to_field_type(pick, spec)
                else:
                    # Field isn't enumerable: dispatch natively, then
                    # the cross predicate will be re-checked below.
                    bound[name] = spec.dispatch(engine)
            if failed:
                self._notify_reject(engine, bound)
                attempts += 1
                continue
            if self.where is None or self.where(bound):
                return bound
            self._notify_reject(engine, bound)
            attempts += 1
        raise GrammarExhausted("gen.struct: max_retries exceeded")


def _coerce_to_field_type(pick: str, spec: Gen) -> Any:
    """When Struct forward-checking routes an Int/Bool through sample_choice,
    the engine returns the stringified value; convert back to the native type."""
    if isinstance(spec, Int):
        return int(pick)
    if isinstance(spec, Bool):
        return pick == "True"
    return pick


@dataclass
class ToolCall(Gen):
    """Tool call as a yield: unifies structured-output / tool-use / agent APIs.

    The Act-3 punchline: there's no separate "tool-use API." A tool call
    is just another decision point in the program.
    """

    tool: Callable[..., Any] = field(default=lambda: None)
    args: dict = field(default_factory=dict)
    where: Callable[[Any], bool] | None = None
    description: str | None = None
    reject_message: Callable[[Any], str] | str | None = None

    def dispatch(self, engine: Engine) -> Any:
        result = self.tool(**self.args)
        if self.where is not None and not self.where(result):
            raise GrammarExhausted(f"tool {self.tool.__name__} output failed predicate")
        return result


# Public constructors — lower-case, keyword-friendly.


def choice(
    options: Sequence[str],
    *,
    where: Callable[[str], bool] | None = None,
    description: str | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 16,
) -> Choice:
    return Choice(
        options=list(options),
        where=where,
        description=description,
        reject_message=reject_message,
        max_retries=max_retries,
    )


def integer(
    min_val: int,
    max_val: int,
    *,
    where: Callable[[int], bool] | None = None,
    description: str | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 16,
) -> Int:
    return Int(
        min_val=min_val,
        max_val=max_val,
        where=where,
        description=description,
        reject_message=reject_message,
        max_retries=max_retries,
    )


int_ = integer


def string(
    *,
    max_len: int = 256,
    pattern: str | None = None,
    where: Callable[[str], bool] | None = None,
    description: str | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 8,
) -> String:
    return String(
        max_len=max_len,
        pattern=pattern,
        where=where,
        description=description,
        reject_message=reject_message,
        max_retries=max_retries,
    )


def boolean(
    *,
    where: Callable[[bool], bool] | None = None,
    description: str | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 4,
) -> Bool:
    return Bool(
        where=where,
        description=description,
        reject_message=reject_message,
        max_retries=max_retries,
    )


bool_ = boolean


def struct(
    *,
    where: Callable[[dict], bool] | None = None,
    description: str | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 8,
    **fields: Gen,
) -> Struct:
    """Compound sugar: every kwarg is a field name -> Gen spec.

    A cross-field ``where=`` enables forward-checking (Layer 3): as each
    field is bound, remaining fields' domains are re-compiled with the
    predicate closed over bound values.

    Usage::

        yield gen.struct(x=gen.integer(0, 10), y=gen.integer(0, 10),
                         where=lambda d: d["x"] + d["y"] == 10)
    """
    return Struct(
        fields=fields,
        where=where,
        description=description,
        reject_message=reject_message,
        max_retries=max_retries,
    )


def tool(
    fn: Callable[..., Any],
    /,
    *,
    description: str | None = None,
    **args: Any,
) -> ToolCall:
    return ToolCall(tool=fn, args=args, description=description)
