from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class MockEngine:
    """Picks uniformly at random from the offered options, seeded.

    Useful for exercising the runner without a model. The `where=`
    tightening loop and compound lowering are observable against this
    engine; only the "good proposer" behavior is not.
    """

    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample_choice(self, options: Sequence[str]) -> str:
        if not options:
            raise ValueError("MockEngine.sample_choice called with empty options")
        return self._rng.choice(list(options))
