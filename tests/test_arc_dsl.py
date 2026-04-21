"""Tests for the ARC transformation DSL, executor, and verifier."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from orate.arc.data import Grid
from orate.arc.dsl import (
    OPS,
    ExecutionError,
    Program,
    Step,
    execute,
)
from orate.arc.verify import describe_mismatch, verify_on_train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeTask:
    """Structural stand-in for ArcTask — verify_on_train only needs .train."""

    train: tuple[tuple[Grid, Grid], ...]


def _g(rows: list[list[int]]) -> Grid:
    return tuple(tuple(r) for r in rows)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_rotate90_four_times_is_identity_on_nonsquare():
    grid = _g([[1, 2, 3], [4, 5, 6]])  # 2x3
    prog = Program((Step("rotate90"), Step("rotate90"), Step("rotate90"), Step("rotate90")))
    assert execute(prog, grid) == grid


def test_rotate90_once_swaps_dims_and_rotates_values():
    grid = _g([[1, 2, 3], [4, 5, 6]])  # 2x3 → 3x2
    # 90deg CW: [[4,1],[5,2],[6,3]]
    assert execute(Program((Step("rotate90"),)), grid) == _g([[4, 1], [5, 2], [6, 3]])


def test_rotate270_is_inverse_of_rotate90():
    grid = _g([[1, 2, 3], [4, 5, 6]])
    p = Program((Step("rotate90"), Step("rotate270")))
    assert execute(p, grid) == grid


def test_rotate180_reverses_in_both_axes():
    grid = _g([[1, 2, 3], [4, 5, 6]])
    assert execute(Program((Step("rotate180"),)), grid) == _g([[6, 5, 4], [3, 2, 1]])


def test_flip_horizontal_twice_is_identity():
    grid = _g([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    prog = Program((Step("flip_horizontal"), Step("flip_horizontal")))
    assert execute(prog, grid) == grid


def test_flip_vertical_twice_is_identity():
    grid = _g([[1, 2, 3], [4, 5, 6]])
    prog = Program((Step("flip_vertical"), Step("flip_vertical")))
    assert execute(prog, grid) == grid


def test_transpose_2x3_becomes_3x2_with_correct_values():
    grid = _g([[1, 2, 3], [4, 5, 6]])
    out = execute(Program((Step("transpose"),)), grid)
    assert out == _g([[1, 4], [2, 5], [3, 6]])


def test_recolor_swaps_colors():
    grid = _g([[0, 1, 0], [1, 0, 1]])
    mapping = ((0, 1), (1, 0))
    out = execute(Program((Step("recolor", (mapping,)),)), grid)
    assert out == _g([[1, 0, 1], [0, 1, 0]])


def test_replace_color_only_touches_matching_cells():
    grid = _g([[0, 1, 2], [1, 2, 0]])
    out = execute(Program((Step("replace_color", (1, 9)),)), grid)
    assert out == _g([[0, 9, 2], [9, 2, 0]])


def test_recolor_rejects_negative_color():
    grid = _g([[0, 1], [1, 0]])
    bad = ((0, -1),)
    with pytest.raises(ExecutionError):
        execute(Program((Step("recolor", (bad,)),)), grid)


def test_crop_to_bbox_single_nonzero_pixel_returns_1x1():
    grid = _g(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 0],
        ]
    )
    out = execute(Program((Step("crop_to_bbox", (0,)),)), grid)
    assert out == _g([[5]])


def test_crop_to_bbox_default_background_is_zero():
    grid = _g([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    # No args → background defaults to 0.
    out = execute(Program((Step("crop_to_bbox"),)), grid)
    assert out == _g([[3]])


def test_tile_horizontal_triples_width():
    grid = _g([[1, 2], [3, 4]])
    out = execute(Program((Step("tile_horizontal", (3,)),)), grid)
    assert out == _g([[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]])


def test_tile_vertical_doubles_height():
    grid = _g([[1, 2], [3, 4]])
    out = execute(Program((Step("tile_vertical", (2,)),)), grid)
    assert out == _g([[1, 2], [3, 4], [1, 2], [3, 4]])


def test_pad_adds_border_with_fill():
    grid = _g([[1, 2], [3, 4]])
    out = execute(Program((Step("pad", (1, 1, 1, 1, 0)),)), grid)
    assert out == _g(
        [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
    )


def test_fill_background_swaps_only_background():
    grid = _g([[0, 1, 0], [1, 0, 1]])
    out = execute(Program((Step("fill_background", (0, 8)),)), grid)
    assert out == _g([[8, 1, 8], [1, 8, 1]])


def test_identity_returns_input_unchanged():
    grid = _g([[1, 2], [3, 4]])
    assert execute(Program((Step("identity"),)), grid) == grid


def test_unknown_op_raises_execution_error():
    grid = _g([[0]])
    with pytest.raises(ExecutionError):
        execute(Program((Step("does_not_exist"),)), grid)


def test_composition_order_matters():
    # rotate90 then flip_horizontal ≠ flip_horizontal then rotate90 in general.
    grid = _g([[1, 2, 3], [4, 5, 6]])
    a = execute(Program((Step("rotate90"), Step("flip_horizontal"))), grid)
    b = execute(Program((Step("flip_horizontal"), Step("rotate90"))), grid)
    assert a != b


def test_ops_registry_has_every_declared_op():
    expected = {
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
    }
    assert expected <= set(OPS.keys())


# ---------------------------------------------------------------------------
# Program identity / hashing
# ---------------------------------------------------------------------------


def test_program_is_hashable_and_set_dedups():
    p1 = Program((Step("rotate90"),))
    p2 = Program((Step("rotate90"),))
    p3 = Program((Step("rotate180"),))
    s = {p1, p2, p3}
    assert len(s) == 2


def test_program_repr_prints_one_step_per_line():
    p = Program((Step("rotate90"), Step("flip_horizontal")))
    r = repr(p)
    assert r.count("\n") >= 2
    assert "rotate90" in r
    assert "flip_horizontal" in r


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_verify_on_train_identity_program_on_identity_task_returns_empty():
    train = (
        (_g([[1, 2], [3, 4]]), _g([[1, 2], [3, 4]])),
        (_g([[0]]), _g([[0]])),
        (_g([[5, 5, 5]]), _g([[5, 5, 5]])),
    )
    task = _FakeTask(train=train)
    assert verify_on_train(Program((Step("identity"),)), task) == []


def test_verify_on_train_returns_mismatching_indices():
    train = (
        (_g([[1, 2]]), _g([[1, 2]])),  # identity matches
        (_g([[3, 4]]), _g([[9, 9]])),  # identity does NOT match
        (_g([[5, 6]]), _g([[7, 8]])),  # identity does NOT match
    )
    task = _FakeTask(train=train)
    assert verify_on_train(Program((Step("identity"),)), task) == [1, 2]


def test_verify_on_train_treats_execution_error_as_mismatch():
    train = ((_g([[0, 1]]), _g([[0, 1]])),)
    task = _FakeTask(train=train)
    # Unknown op → ExecutionError → counted as mismatch, not raised.
    assert verify_on_train(Program((Step("does_not_exist"),)), task) == [0]


def test_describe_mismatch_mentions_shapes_when_shapes_differ():
    # Identity on a 2x3 input vs a 1x1 expected output → shape mismatch.
    train = ((_g([[1, 2, 3], [4, 5, 6]]), _g([[7]])),)
    task = _FakeTask(train=train)
    msg = describe_mismatch(Program((Step("identity"),)), task, 0)
    assert "2x3" in msg
    assert "1x1" in msg
    assert len(msg) < 300


def test_describe_mismatch_reports_cells_on_same_shape():
    # Shapes match but values differ in one cell.
    train = ((_g([[0, 0], [0, 0]]), _g([[0, 9], [0, 0]])),)
    task = _FakeTask(train=train)
    msg = describe_mismatch(Program((Step("identity"),)), task, 0)
    assert "(0,1)" in msg
    assert len(msg) < 300


def test_describe_mismatch_reports_execution_error():
    train = ((_g([[0, 1]]), _g([[0, 1]])),)
    task = _FakeTask(train=train)
    msg = describe_mismatch(Program((Step("does_not_exist"),)), task, 0)
    assert "execution error" in msg.lower() or "error" in msg.lower()
    assert len(msg) < 300


def test_describe_mismatch_reports_match_when_correct():
    train = ((_g([[1, 2]]), _g([[1, 2]])),)
    task = _FakeTask(train=train)
    msg = describe_mismatch(Program((Step("identity"),)), task, 0)
    assert "match" in msg.lower()


def test_describe_mismatch_index_out_of_range_raises():
    train = ((_g([[1]]), _g([[1]])),)
    task = _FakeTask(train=train)
    with pytest.raises(IndexError):
        describe_mismatch(Program((Step("identity"),)), task, 5)
