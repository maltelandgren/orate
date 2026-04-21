"""Verify a Program against the training demonstrations of an ARC task.

Two responsibilities:

* ``verify_on_train`` — pure consistency check. Returns the list of train-pair
  indices where executing the program on the input does not yield the expected
  output. Empty list means "consistent with every demo".
* ``describe_mismatch`` — builds a short natural-language steering string for
  Phase-B retry. The LLM is shown this description so it can correct the
  program on its next proposal.

The task parameter is typed structurally: anything with a ``train`` attribute
that is an iterable of ``(input_grid, output_grid)`` pairs works. This keeps
``verify`` decoupled from the exact ``ArcTask`` dataclass shape if the data
layer evolves.
"""

from __future__ import annotations

from typing import Any

from orate.arc.data import Grid, grid_shape, grids_equal
from orate.arc.dsl import ExecutionError, Program, execute

_MAX_COORDS_IN_DESCRIPTION = 5


def _train_pairs(task: Any) -> list[tuple[Grid, Grid]]:
    train = getattr(task, "train", None)
    if train is None:
        raise TypeError("task has no 'train' attribute")
    return [(inp, out) for inp, out in train]


def verify_on_train(program: Program, task: Any) -> list[int]:
    """Return indices of train pairs where ``program`` disagrees with the demo.

    Pairs that raise ``ExecutionError`` during ``execute`` count as mismatches
    (the program is malformed or shape-incompatible with that demo).
    """
    mismatches: list[int] = []
    for i, (inp, expected) in enumerate(_train_pairs(task)):
        try:
            produced = execute(program, inp)
        except ExecutionError:
            mismatches.append(i)
            continue
        if not grids_equal(produced, expected):
            mismatches.append(i)
    return mismatches


def _diff_coords(a: Grid, b: Grid, limit: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    rows = min(len(a), len(b))
    for r in range(rows):
        row_a = a[r]
        row_b = b[r]
        cols = min(len(row_a), len(row_b))
        for c in range(cols):
            if row_a[c] != row_b[c]:
                coords.append((r, c))
                if len(coords) >= limit:
                    return coords
    return coords


def describe_mismatch(program: Program, task: Any, train_idx: int) -> str:
    """Build a concise (<300 char) description of what went wrong on train[idx].

    The string is injected into the LLM context on a Phase-B retry so it can
    self-correct. It covers three cases: execution error, shape mismatch, and
    cell-level mismatch (listing up to a handful of offending coordinates).
    """
    pairs = _train_pairs(task)
    if not (0 <= train_idx < len(pairs)):
        raise IndexError(f"train_idx {train_idx} out of range [0, {len(pairs)})")
    inp, expected = pairs[train_idx]
    exp_shape = grid_shape(expected)
    try:
        produced = execute(program, inp)
    except ExecutionError as e:
        return (
            f"Applied to training example {train_idx}, your program failed with an "
            f"execution error: {e}. Expected output shape {exp_shape}."
        )[:299]
    prod_shape = grid_shape(produced)
    if prod_shape != exp_shape:
        return (
            f"Applied to training example {train_idx}, your program produced a "
            f"{prod_shape[0]}x{prod_shape[1]} grid; the expected output is "
            f"{exp_shape[0]}x{exp_shape[1]}."
        )[:299]
    if grids_equal(produced, expected):
        return f"Applied to training example {train_idx}, the output matches."
    coords = _diff_coords(produced, expected, _MAX_COORDS_IN_DESCRIPTION)
    total = sum(
        1
        for r in range(exp_shape[0])
        for c in range(exp_shape[1])
        if produced[r][c] != expected[r][c]
    )
    shown = ", ".join(f"({r},{c})" for r, c in coords)
    suffix = f" (+{total - len(coords)} more)" if total > len(coords) else ""
    return (
        f"Applied to training example {train_idx}, your program produced a "
        f"{prod_shape[0]}x{prod_shape[1]} grid matching the expected shape but "
        f"differing in cells {shown}{suffix}."
    )[:299]
