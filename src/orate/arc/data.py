"""ARC-AGI-2 task loading.

Tasks live in ``arc-data/ARC-AGI-2/data/{split}/{task_id}.json`` relative to the
repository root. Grids are stored as nested tuples so they are hashable and
usable as dict / set keys for caches later in the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Grid = tuple[tuple[int, ...], ...]


# Repo root = two parents up from this file (src/orate/arc/data.py → repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _REPO_ROOT / "arc-data" / "ARC-AGI-2" / "data"


@dataclass(frozen=True)
class ArcTask:
    """A single ARC-AGI-2 task: train demos + test queries."""

    task_id: str
    train: tuple[tuple[Grid, Grid], ...]
    test: tuple[tuple[Grid, Grid | None], ...]


def _to_grid(raw: list[list[int]]) -> Grid:
    return tuple(tuple(int(c) for c in row) for row in raw)


def _split_dir(split: str) -> Path:
    if split not in {"training", "evaluation"}:
        raise ValueError(f"unknown split {split!r}; expected 'training' or 'evaluation'")
    return _DATA_ROOT / split


def load_task(task_id: str, split: str = "training") -> ArcTask:
    """Load a task by ID from the given split."""
    path = _split_dir(split) / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"task {task_id!r} not found at {path}")
    with path.open() as f:
        raw = json.load(f)

    train = tuple((_to_grid(pair["input"]), _to_grid(pair["output"])) for pair in raw["train"])
    test: tuple[tuple[Grid, Grid | None], ...] = tuple(
        (
            _to_grid(pair["input"]),
            _to_grid(pair["output"]) if pair.get("output") is not None else None,
        )
        for pair in raw["test"]
    )
    return ArcTask(task_id=task_id, train=train, test=test)


def list_tasks(split: str = "training") -> list[str]:
    """Return sorted task IDs (filename stems) for the given split."""
    d = _split_dir(split)
    if not d.exists():
        raise FileNotFoundError(
            f"ARC data directory missing: {d}. "
            "Clone with: git clone https://github.com/arcprize/ARC-AGI-2 arc-data/ARC-AGI-2"
        )
    return sorted(p.stem for p in d.glob("*.json"))


def grids_equal(a: Grid, b: Grid) -> bool:
    """Deep structural equality between two grids."""
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b, strict=True):
        if len(row_a) != len(row_b):
            return False
        if any(x != y for x, y in zip(row_a, row_b, strict=True)):
            return False
    return True


def grid_shape(g: Grid) -> tuple[int, int]:
    """Return (rows, cols). Cols is 0 for an empty grid."""
    rows = len(g)
    cols = len(g[0]) if rows else 0
    return rows, cols
