from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from orate.engine.protocol import Engine
from orate.gen import Gen, GrammarExhausted


class ProgramRejected(RuntimeError):
    """Raised when a @program body signals its own rejection.

    Caught by the Phase-C whole-program retry loop; the program is
    rewound, a reject message is optionally injected into the session,
    and the body runs from the top.
    """


def reject_program(message: str | None = None) -> None:
    """Signal that the current @program invocation should be rewound.

    Usage inside a @program body::

        if not task_validator(result):
            reject_program("the result failed task validation")

    Causes the Phase-C retry path to rewind and re-invoke the body.
    """
    raise ProgramRejected(message or "program rejected by body")


@dataclass
class ProgramInvocation:
    """A bound call to a @program body. run(engine) executes the coroutine.

    On each yielded Gen, the runner calls Gen.dispatch(engine); Gen
    internally handles predicate rejection and grammar tightening. The
    ProgramInvocation is the outer unit for Phase-C whole-program retry.
    """

    body: Callable[..., Iterator[Gen]]
    args: tuple
    kwargs: dict
    whole_program_retries: int = 0
    reject_message: Callable[[int, BaseException], str] | str | None = None
    ends_turn: bool = False
    trace: list[dict] = field(default_factory=list)

    def _run_once(self, engine: Engine) -> Any:
        gen_iter = self.body(*self.args, **self.kwargs)
        sent: Any = None
        while True:
            try:
                spec = gen_iter.send(sent) if sent is not None else next(gen_iter)
            except StopIteration as stop:
                return stop.value
            if isinstance(spec, ProgramInvocation):
                # Flavor B (minimal): recursively run the sub-program on the
                # same engine. The inner invocation uses its own
                # whole_program_retries; its ProgramRejected / GrammarExhausted
                # propagate up to the outer's Phase-C loop only if the inner
                # exhausts its own retries.
                sent = spec.run(engine=engine)
            elif isinstance(spec, Gen):
                sent = spec.dispatch(engine)
            else:
                raise TypeError(
                    f"@program body yielded non-Gen, non-ProgramInvocation "
                    f"value {spec!r}; yield a Gen "
                    f"(gen.choice(...) / gen.integer(...) / gen.tool(...) / etc.) "
                    f"or another @program invocation."
                )

    def run(self, *, engine: Engine) -> Any:
        """Run the program. On Phase-C-eligible failure, rewind and retry."""
        last_exc: BaseException | None = None
        for attempt in range(self.whole_program_retries + 1):
            try:
                result = self._run_once(engine)
                self.trace.append({"attempt": attempt, "status": "ok"})
                return result
            except (ProgramRejected, GrammarExhausted) as exc:
                last_exc = exc
                self.trace.append({"attempt": attempt, "status": "rejected", "reason": str(exc)})
                if attempt >= self.whole_program_retries:
                    break
                self._inject_program_level_reject(engine, attempt, exc)
        assert last_exc is not None
        raise last_exc

    def _inject_program_level_reject(
        self, engine: Engine, attempt: int, exc: BaseException
    ) -> None:
        if self.reject_message is None:
            if hasattr(engine, "inject_context"):
                engine.inject_context(
                    f"(previous attempt #{attempt} was rejected: {exc}. Try a different approach.)"
                )
            return
        if callable(self.reject_message):
            msg = self.reject_message(attempt, exc)
        else:
            msg = self.reject_message
        if hasattr(engine, "inject_context"):
            engine.inject_context(f"(note: {msg})")


def program(
    fn: Callable[..., Iterator[Gen]] | None = None,
    /,
    *,
    whole_program_retries: int = 0,
    reject_message: Callable[[int, BaseException], str] | str | None = None,
    ends_turn: bool = False,
) -> Any:
    """Decorator: turn a generator function into a runnable program.

    Two forms::

        @program
        def f(): ...                             # no Phase-C retry

        @program(whole_program_retries=3)        # Phase-C retry on
        def g(): ...                             # ProgramRejected / GrammarExhausted

    The decorated function returns a ProgramInvocation; call `.run(engine=...)`
    to execute. The generator is driven one yield at a time against the engine.

    ``ends_turn`` is metadata only at the @program runner level — it is not
    read or acted on here. A future Session runner inspects
    ``invocation.ends_turn`` to decide whether completing this invocation
    should end the agent turn.
    """

    def decorate(fn_: Callable[..., Iterator[Gen]]) -> Callable[..., ProgramInvocation]:
        def wrapper(*args: Any, **kwargs: Any) -> ProgramInvocation:
            return ProgramInvocation(
                body=fn_,
                args=args,
                kwargs=kwargs,
                whole_program_retries=whole_program_retries,
                reject_message=reject_message,
                ends_turn=ends_turn,
            )

        wrapper.__wrapped__ = fn_  # type: ignore[attr-defined]
        wrapper.__name__ = getattr(fn_, "__name__", "program")
        return wrapper

    if fn is not None and callable(fn):
        return decorate(fn)
    return decorate
