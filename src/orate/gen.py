from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from orate.engine.protocol import Engine


class GrammarExhausted(RuntimeError):
    """Raised when tightening has excluded every candidate."""


class Gen:
    """Base for a yieldable generation spec.

    A @program's body yields Gen instances at decision points. The
    runner calls dispatch(engine) on each yield; the returned value is
    sent back to the generator as the yield's result.
    """

    def dispatch(self, engine: Engine) -> Any:
        raise NotImplementedError


@dataclass
class Choice(Gen):
    """Pick one of a fixed set of string options.

    `where=` is a predicate over the sampled value. Rejection tightens
    the accepted set by removing the offending option; the loop
    re-samples until the predicate passes or the set is empty. This is
    the toy version of the ADR-0014 mechanism — the real backend does
    this at the grammar/FSM level.
    """

    options: Sequence[str]
    where: Callable[[str], bool] | None = None

    def dispatch(self, engine: Engine) -> str:
        remaining = list(self.options)
        if not remaining:
            raise ValueError("gen.choice requires at least one option")
        while remaining:
            pick = engine.sample_choice(remaining)
            if self.where is None or self.where(pick):
                return pick
            remaining = [o for o in remaining if o != pick]
        raise GrammarExhausted(
            f"gen.choice: no option in {list(self.options)!r} satisfies predicate"
        )


def choice(
    options: Sequence[str],
    *,
    where: Callable[[str], bool] | None = None,
) -> Choice:
    return Choice(options=list(options), where=where)
