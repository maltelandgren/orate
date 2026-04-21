from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    """What an engine backend must implement.

    One method per primitive gen type. Engines that support compound
    lowering can also implement `sample_struct` (optional) to fuse a
    dict of fields into a single sampling call; without it, the runner
    falls back to sequential dispatch.

    The `excluded` arg on integer/string samplers is the tightening
    handle (ADR-0014): the engine must not return a value the runner
    has already rejected. For XGrammar this is enforced at the grammar
    mask level; for Mock it's a post-sample filter.
    """

    def sample_choice(self, options: Sequence[str]) -> str: ...

    def sample_int(
        self,
        min_val: int,
        max_val: int,
        *,
        excluded: set[int] | None = None,
    ) -> int: ...

    def sample_string(
        self,
        *,
        max_len: int,
        pattern: str | None = None,
        excluded: set[str] | None = None,
    ) -> str: ...

    def sample_bool(self) -> bool: ...


@runtime_checkable
class SupportsStruct(Protocol):
    """Optional capability: fused struct sampling."""

    def sample_struct(self, fields: dict[str, Any]) -> dict: ...


@runtime_checkable
class SupportsContext(Protocol):
    """Optional capability: inject text into the session between yields.

    Phase-B hook — the runner calls this after a predicate rejection to
    give the next sample a natural-language steering signal. Engines
    without a session (e.g. MockEngine) are exempt.
    """

    def inject_context(self, text: str) -> None: ...
