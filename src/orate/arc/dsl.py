"""ARC transformation DSL and executor.

A ``Program`` is an immutable sequence of typed ``Step`` primitives. Each step
names a single grid-to-grid operation from ``OPS`` plus a tuple of positional
arguments. Programs are hashable (ergo usable as dedup keys in the Phase-B
"don't re-propose this" set) and execute deterministically against an input
``Grid``.

The primitives are intentionally small and compositional so an LLM can reason
about them in natural language during proposal. See ``OPS`` below for the full
menu.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

# Import from the leaf module rather than the package to avoid depending on
# other modules (``render.py``) being present in the parallel workstream.
from orate.arc.data import Grid, grid_shape

OpKind = Literal[
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "recolor",
    "replace_color",
    "crop_to_bbox",
    "tile_horizontal",
    "tile_vertical",
    "pad",
    "fill_background",
]


class ExecutionError(RuntimeError):
    """Raised when a Step cannot execute against a grid (bad args, shape, etc.)."""


@dataclass(frozen=True)
class Step:
    """A single op + positional args. Immutable and hashable."""

    kind: str
    args: tuple = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if not self.args:
            return f"Step({self.kind!r})"
        return f"Step({self.kind!r}, args={self.args!r})"


@dataclass(frozen=True)
class Program:
    """An ordered list of Steps. Hashable so callers can dedup proposals."""

    steps: tuple[Step, ...] = ()

    def __repr__(self) -> str:
        if not self.steps:
            return "Program([])"
        lines = ["Program(["]
        lines.extend(f"  {s!r}," for s in self.steps)
        lines.append("])")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primitive implementations. Each takes a Grid (tuple-of-tuples) plus op-specific
# positional args and returns a new Grid. Pure functions.
# ---------------------------------------------------------------------------


def _op_identity(g: Grid) -> Grid:
    return g


def _op_rotate90(g: Grid) -> Grid:
    # 90deg clockwise: new[r][c] = old[rows-1-c][r]
    rows, cols = grid_shape(g)
    if rows == 0 or cols == 0:
        return g
    return tuple(tuple(g[rows - 1 - c][r] for c in range(rows)) for r in range(cols))


def _op_rotate180(g: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in reversed(g))


def _op_rotate270(g: Grid) -> Grid:
    # 270deg clockwise == 90deg counter-clockwise.
    rows, cols = grid_shape(g)
    if rows == 0 or cols == 0:
        return g
    return tuple(tuple(g[c][cols - 1 - r] for c in range(rows)) for r in range(cols))


def _op_flip_horizontal(g: Grid) -> Grid:
    """Mirror across the vertical axis (swap left/right)."""
    return tuple(tuple(reversed(row)) for row in g)


def _op_flip_vertical(g: Grid) -> Grid:
    """Mirror across the horizontal axis (swap top/bottom)."""
    return tuple(reversed(g))


def _op_transpose(g: Grid) -> Grid:
    rows, cols = grid_shape(g)
    if rows == 0 or cols == 0:
        return g
    return tuple(tuple(g[r][c] for r in range(rows)) for c in range(cols))


def _validate_color(c: int, *, field: str) -> None:
    if not isinstance(c, int) or isinstance(c, bool):
        raise ExecutionError(f"{field} must be int, got {type(c).__name__}")
    if c < 0 or c > 9:
        raise ExecutionError(f"{field}={c} out of ARC color range [0, 9]")


def _op_recolor(g: Grid, mapping: tuple[tuple[int, int], ...]) -> Grid:
    """Apply a color remapping. ``mapping`` is a tuple of (old, new) pairs."""
    if not isinstance(mapping, tuple):
        raise ExecutionError(f"recolor mapping must be tuple, got {type(mapping).__name__}")
    table: dict[int, int] = {}
    for pair in mapping:
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ExecutionError(f"recolor mapping entries must be (old, new) pairs; got {pair!r}")
        old, new = pair
        _validate_color(old, field="recolor old")
        _validate_color(new, field="recolor new")
        table[old] = new
    return tuple(tuple(table.get(v, v) for v in row) for row in g)


def _op_replace_color(g: Grid, old: int, new: int) -> Grid:
    _validate_color(old, field="replace_color old")
    _validate_color(new, field="replace_color new")
    return tuple(tuple(new if v == old else v for v in row) for row in g)


def _op_crop_to_bbox(g: Grid, background: int = 0) -> Grid:
    _validate_color(background, field="crop_to_bbox background")
    rows, cols = grid_shape(g)
    if rows == 0 or cols == 0:
        return g
    top, left = rows, cols
    bottom, right = -1, -1
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != background:
                if r < top:
                    top = r
                if r > bottom:
                    bottom = r
                if c < left:
                    left = c
                if c > right:
                    right = c
    if bottom < 0:
        # No non-background cells — return the original grid unchanged.
        return g
    return tuple(tuple(g[r][c] for c in range(left, right + 1)) for r in range(top, bottom + 1))


def _op_tile_horizontal(g: Grid, times: int) -> Grid:
    if not isinstance(times, int) or times < 1:
        raise ExecutionError(f"tile_horizontal times must be positive int, got {times!r}")
    return tuple(row * times for row in g)


def _op_tile_vertical(g: Grid, times: int) -> Grid:
    if not isinstance(times, int) or times < 1:
        raise ExecutionError(f"tile_vertical times must be positive int, got {times!r}")
    return g * times


def _op_pad(g: Grid, top: int, bottom: int, left: int, right: int, fill: int = 0) -> Grid:
    for name, v in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        if not isinstance(v, int) or v < 0:
            raise ExecutionError(f"pad {name} must be non-negative int, got {v!r}")
    _validate_color(fill, field="pad fill")
    rows, cols = grid_shape(g)
    new_cols = cols + left + right
    top_row = tuple(fill for _ in range(new_cols))
    padded_rows = []
    for _ in range(top):
        padded_rows.append(top_row)
    for r in range(rows):
        padded_rows.append(
            tuple(fill for _ in range(left)) + tuple(g[r]) + tuple(fill for _ in range(right))
        )
    for _ in range(bottom):
        padded_rows.append(top_row)
    return tuple(padded_rows)


def _op_fill_background(g: Grid, old_fill: int, new_fill: int) -> Grid:
    """Swap one solid background color for another. Alias of replace_color but
    semantically distinct when reasoning about ARC tasks (LLMs propose it when
    they specifically want to change the background rather than any color)."""
    return _op_replace_color(g, old_fill, new_fill)


OPS: dict[str, Callable[..., Grid]] = {
    "identity": _op_identity,
    "rotate90": _op_rotate90,
    "rotate180": _op_rotate180,
    "rotate270": _op_rotate270,
    "flip_horizontal": _op_flip_horizontal,
    "flip_vertical": _op_flip_vertical,
    "transpose": _op_transpose,
    "recolor": _op_recolor,
    "replace_color": _op_replace_color,
    "crop_to_bbox": _op_crop_to_bbox,
    "tile_horizontal": _op_tile_horizontal,
    "tile_vertical": _op_tile_vertical,
    "pad": _op_pad,
    "fill_background": _op_fill_background,
}


def execute(program: Program, grid: Grid) -> Grid:
    """Run ``program`` against ``grid`` and return the resulting grid.

    Raises ``ExecutionError`` for unknown ops, bad arg shapes, or invalid colors.
    """
    current = grid
    for i, step in enumerate(program.steps):
        impl = OPS.get(step.kind)
        if impl is None:
            raise ExecutionError(f"step {i}: unknown op {step.kind!r}")
        try:
            current = impl(current, *step.args)
        except ExecutionError:
            raise
        except TypeError as e:
            raise ExecutionError(f"step {i} ({step.kind}): bad args {step.args!r}: {e}") from e
    return current
