from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from orate.engine.protocol import Engine
from orate.gen import Gen


@dataclass
class ProgramInvocation:
    """A bound call to a @program body. run(engine) executes the coroutine."""

    body: Callable[..., Iterator[Gen]]
    args: tuple
    kwargs: dict

    def run(self, *, engine: Engine) -> Any:
        gen_iter = self.body(*self.args, **self.kwargs)
        sent: Any = None
        while True:
            try:
                spec = gen_iter.send(sent) if sent is not None else next(gen_iter)
            except StopIteration as stop:
                return stop.value
            if not isinstance(spec, Gen):
                raise TypeError(
                    f"@program body yielded non-Gen value {spec!r}; "
                    f"use gen.choice(...) / gen.int(...) / etc."
                )
            sent = spec.dispatch(engine)


def program(fn: Callable[..., Iterator[Gen]]) -> Callable[..., ProgramInvocation]:
    """Decorator: turn a generator function into a runnable program.

    The decorated function returns a ProgramInvocation; call .run(engine=...)
    to execute. The generator is driven one yield at a time against the engine.
    """

    def wrapper(*args: Any, **kwargs: Any) -> ProgramInvocation:
        return ProgramInvocation(body=fn, args=args, kwargs=kwargs)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(fn, "__name__", "program")
    return wrapper
