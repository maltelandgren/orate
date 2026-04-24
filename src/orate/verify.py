"""First-class verifiers.

A ``Verifier`` is a named callable that inspects a value and returns
either ``Accept()`` or ``Reject(message)``. Yielded inside a
``@program``, a Verifier's reject raises ``ProgramRejected`` — which
the Phase-C retry loop catches, rewinds on, and injects the message
into the engine session for the next attempt.

Semantically a Verifier plays the same role for validation that
``gen.*`` plays for generation: a named, composable, introspectable
unit. You can build libraries of them, reuse them across programs,
and test them in isolation without an engine.

Example::

    from orate import program, gen
    from orate.verify import Accept, Reject, verifier

    @verifier
    def has_at_most(n, *, limit):
        if len(n) <= limit:
            return Accept()
        return Reject(f"length {len(n)} exceeds limit {limit}")

    @program(whole_program_retries=5)
    def short_word():
        w = yield gen.string(max_len=50)
        yield has_at_most(w, limit=5)
        return w
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from orate.gen import Gen


@dataclass(frozen=True)
class Accept:
    """A verifier's success result. Yield value is `None`."""

    __slots__ = ()


@dataclass(frozen=True)
class Reject:
    """A verifier's failure result. The message is injected as Phase-C context."""

    message: str


VerifierResult = Accept | Reject


@dataclass
class VerifierCall(Gen):
    """Runtime yield produced by calling a @verifier-decorated function.

    A user writes ``yield my_verifier(value, ...)``; that call returns
    a VerifierCall bound to the value. The runner dispatches it like
    any other Gen yield: on dispatch we run the check function and
    either return (Accept) or raise ProgramRejected (Reject).
    """

    check: Callable[..., VerifierResult] = field(default=lambda *a, **k: Accept())
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    name: str = "verifier"
    description: str | None = None

    def dispatch(self, engine: Any) -> None:  # noqa: ARG002 - engine not used
        # Local import avoids a program ↔ verify cycle.
        from orate.program import ProgramRejected  # noqa: PLC0415

        result = self.check(*self.args, **self.kwargs)
        if isinstance(result, Accept):
            return None
        if isinstance(result, Reject):
            raise ProgramRejected(f"{self.name}: {result.message}")
        raise TypeError(
            f"@verifier {self.name!r} must return Accept() or Reject(...); got {result!r}"
        )


def verifier(
    fn: Callable[..., VerifierResult] | None = None,
    /,
    *,
    description: str | None = None,
) -> Any:
    """Decorate a function as a first-class Verifier.

    The decorated function should accept a candidate value and any
    user-supplied context, returning ``Accept()`` or ``Reject(msg)``.
    Calling the decorated function returns a ``VerifierCall`` which
    you ``yield`` inside a ``@program``.

    Supports both bare and parametrized forms::

        @verifier
        def v1(x): ...

        @verifier(description="must be short")
        def v2(x): ...
    """

    def decorate(fn_: Callable[..., VerifierResult]) -> Callable[..., VerifierCall]:
        name = getattr(fn_, "__name__", "verifier")

        def factory(*args: Any, **kwargs: Any) -> VerifierCall:
            return VerifierCall(
                check=fn_,
                args=args,
                kwargs=kwargs,
                name=name,
                description=description,
            )

        factory.__wrapped__ = fn_  # type: ignore[attr-defined]
        factory.__name__ = name
        return factory

    if fn is not None and callable(fn):
        return decorate(fn)
    return decorate
