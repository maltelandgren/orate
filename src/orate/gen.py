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
    The narrowing is deterministic — this is the ADR-0014 stance: no
    dice rolling to reach correctness.
    """

    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 16

    def dispatch(self, engine: Engine) -> Any:
        raise NotImplementedError

    def _format_reject(self, value: Any) -> str | None:
        """Phase-B hook: produce a natural-language reject message on predicate fail.

        The runner can use this to append context to the session before
        the next retry. Engine-agnostic; the injection is the runner's job.
        """
        if self.reject_message is None:
            return None
        if callable(self.reject_message):
            return self.reject_message(value)
        return self.reject_message

    def _notify_reject(self, engine: Any, value: Any) -> None:
        """Inject a natural-language reject hint into the engine's session.

        No-op unless (a) the Gen has a reject_message and (b) the engine
        implements inject_context. This is Phase-B: the grammar still
        tightens, but the model *also* sees why the last sample failed,
        so its next argmax moves to a different region of the accept set.
        """
        msg = self._format_reject(value)
        if msg and hasattr(engine, "inject_context"):
            engine.inject_context(f"(note: {msg})")


@dataclass
class Choice(Gen):
    """Pick one of a fixed set of string options."""

    options: Sequence[str] = ()
    where: Callable[[str], bool] | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 16

    def dispatch(self, engine: Engine) -> str:
        remaining = list(self.options)
        if not remaining:
            raise ValueError("gen.choice requires at least one option")
        attempts = 0
        while remaining and attempts < self.max_retries:
            pick = engine.sample_choice(remaining)
            if self.where is None or self.where(pick):
                return pick
            self._notify_reject(engine, pick)
            remaining = [o for o in remaining if o != pick]
            attempts += 1
        raise GrammarExhausted(
            f"gen.choice: no option in {list(self.options)!r} satisfies predicate"
        )


@dataclass
class Int(Gen):
    """Pick an integer in [min_val, max_val]."""

    min_val: int = 0
    max_val: int = 0
    where: Callable[[int], bool] | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 16

    def dispatch(self, engine: Engine) -> int:
        if self.min_val > self.max_val:
            raise ValueError(f"gen.int: min={self.min_val} > max={self.max_val}")
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

    Tightening on reject is weak here (no natural "exclude one string"
    narrowing). We rely on max_retries and on `reject_message` carrying
    the steering signal into the next attempt's context.
    """

    max_len: int = 256
    pattern: str | None = None
    where: Callable[[str], bool] | None = None
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
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 4

    def dispatch(self, engine: Engine) -> bool:
        candidates = [True, False]
        for _ in range(self.max_retries):
            pick = engine.sample_bool()
            if self.where is None or self.where(pick):
                return pick
            self._notify_reject(engine, pick)
            candidates = [c for c in candidates if c != pick]
            if not candidates:
                break
        raise GrammarExhausted("gen.bool: no value satisfies predicate")


@dataclass
class Struct(Gen):
    """Compound sugar: yield a dict of typed fields in one logical step.

    Semantically equivalent to yielding each field in sequence and
    packaging into a dict. Engines that support compound grammar
    lowering (XGrammar) can fuse the fields into a single sampling
    call; engines that don't (Mock) fall back to sequential dispatch.
    """

    fields: dict[str, Gen] = field(default_factory=dict)
    where: Callable[[dict], bool] | None = None
    reject_message: Callable[[Any], str] | str | None = None
    max_retries: int = 8

    def dispatch(self, engine: Engine) -> dict:
        attempts = 0
        while attempts < self.max_retries:
            if hasattr(engine, "sample_struct"):
                result = engine.sample_struct(self.fields)
            else:
                result = {name: child.dispatch(engine) for name, child in self.fields.items()}
            if self.where is None or self.where(result):
                return result
            self._notify_reject(engine, result)
            attempts += 1
        raise GrammarExhausted("gen.struct: max_retries exceeded")


@dataclass
class ToolCall(Gen):
    """Tool call as a yield: unifies structured-output / tool-use / agent APIs.

    The Act-3 punchline: there's no separate "tool-use API." A tool call
    is just another decision point in the program. The runner dispatches
    it against whichever engine is active.
    """

    tool: Callable[..., Any] = field(default=lambda: None)
    args: dict = field(default_factory=dict)
    where: Callable[[Any], bool] | None = None
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
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 16,
) -> Choice:
    return Choice(
        options=list(options),
        where=where,
        reject_message=reject_message,
        max_retries=max_retries,
    )


def integer(
    min_val: int,
    max_val: int,
    *,
    where: Callable[[int], bool] | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 16,
) -> Int:
    return Int(
        min_val=min_val,
        max_val=max_val,
        where=where,
        reject_message=reject_message,
        max_retries=max_retries,
    )


# Alias: `int` shadows the builtin, so we expose both. Users write `gen.integer(...)`
# or `gen.int_(...)`; the former is preferred.
int_ = integer


def string(
    *,
    max_len: int = 256,
    pattern: str | None = None,
    where: Callable[[str], bool] | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 8,
) -> String:
    return String(
        max_len=max_len,
        pattern=pattern,
        where=where,
        reject_message=reject_message,
        max_retries=max_retries,
    )


def boolean(
    *,
    where: Callable[[bool], bool] | None = None,
    reject_message: Callable[[Any], str] | str | None = None,
    max_retries: int = 4,
) -> Bool:
    return Bool(where=where, reject_message=reject_message, max_retries=max_retries)


bool_ = boolean


def struct(
    **fields: Gen,
) -> Struct:
    """Compound sugar: every kwarg is a field name -> Gen spec.

    Usage: yield gen.struct(name=gen.string(max_len=20), age=gen.integer(0, 120))
    """
    return Struct(fields=fields)


def tool(
    fn: Callable[..., Any],
    /,
    **args: Any,
) -> ToolCall:
    """Yield a tool call. Runs the function now; returned value is the yield result.

    Under the orate thesis this is not a special "tool-use API" — it's
    just another yield. The engine doesn't need to know about tools at
    all; the runner handles ToolCall locally.
    """
    return ToolCall(tool=fn, args=args)
