from __future__ import annotations

import random
import string as _string
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockEngine:
    """Random sampler, seeded. For tests and offline dry-runs.

    Exercises the runner and the tightening-on-reject loop but cannot
    produce meaningful content — the output of MockEngine is noise.
    Useful to confirm wiring without touching a model.
    """

    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)
    _context: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample_choice(self, options: Sequence[str]) -> str:
        if not options:
            raise ValueError("MockEngine.sample_choice called with empty options")
        return self._rng.choice(list(options))

    def sample_int(
        self,
        min_val: int,
        max_val: int,
        *,
        excluded: set[int] | None = None,
    ) -> int:
        excluded = excluded or set()
        allowed = [i for i in range(min_val, max_val + 1) if i not in excluded]
        if not allowed:
            raise ValueError("MockEngine.sample_int: all values excluded")
        return self._rng.choice(allowed)

    def sample_string(
        self,
        *,
        max_len: int,
        pattern: str | None = None,
        excluded: set[str] | None = None,
    ) -> str:
        # Pattern is honored only trivially; this is a mock.
        excluded = excluded or set()
        for _ in range(32):
            length = self._rng.randint(1, max(1, min(max_len, 16)))
            candidate = "".join(
                self._rng.choice(_string.ascii_lowercase) for _ in range(length)
            )
            if candidate not in excluded:
                return candidate
        raise RuntimeError("MockEngine.sample_string: could not escape excluded set")

    def sample_bool(self) -> bool:
        return self._rng.choice([True, False])

    def sample_struct(self, fields: dict[str, Any]) -> dict:
        # MockEngine lowers to per-field dispatch; no real compound behavior.
        return {name: spec.dispatch(self) for name, spec in fields.items()}

    def inject_context(self, text: str) -> None:
        self._context.append(text)
