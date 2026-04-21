from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    """Minimal engine surface.

    Grows as new Gen types land (int ranges, free strings with regex,
    compound grammars). For now `sample_choice` is enough to exercise
    the @program runner and the tightening-on-reject loop.
    """

    def sample_choice(self, options: Sequence[str]) -> str: ...
