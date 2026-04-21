"""Tests for ``orate.arc`` data loading and rendering."""

from __future__ import annotations

import pytest

from orate.arc import (
    ArcTask,
    grid_shape,
    grid_to_ascii,
    grids_equal,
    list_tasks,
    load_task,
    render_task_to_ascii,
)


def _first_task_id() -> str:
    ids = list_tasks("training")
    assert ids, "no training tasks found — is arc-data/ARC-AGI-2 cloned?"
    return ids[0]


def test_list_tasks_training_has_1000():
    ids = list_tasks("training")
    assert len(ids) == 1000
    assert ids == sorted(ids)


def test_list_tasks_evaluation_has_120():
    ids = list_tasks("evaluation")
    assert len(ids) == 120


def test_list_tasks_bad_split():
    with pytest.raises(ValueError):
        list_tasks("nope")


def test_load_task_structure():
    tid = _first_task_id()
    task = load_task(tid)
    assert isinstance(task, ArcTask)
    assert task.task_id == tid
    assert len(task.train) >= 1
    assert len(task.test) >= 1
    # grids are tuples-of-tuples
    inp, out = task.train[0]
    assert isinstance(inp, tuple)
    assert isinstance(inp[0], tuple)
    assert isinstance(out, tuple)
    # ARC-AGI-2 training pairs have outputs provided
    assert out is not None
    # frozen dataclass → hashable
    hash(task)


def test_load_task_missing():
    with pytest.raises(FileNotFoundError):
        load_task("does_not_exist_xxxx")


def test_grid_shape():
    g = ((1, 2, 3), (4, 5, 6))
    assert grid_shape(g) == (2, 3)
    assert grid_shape(()) == (0, 0)


def test_grids_equal_positive():
    a = ((1, 2), (3, 4))
    b = ((1, 2), (3, 4))
    assert grids_equal(a, b)


def test_grids_equal_negative():
    a = ((1, 2), (3, 4))
    assert not grids_equal(a, ((1, 2), (3, 5)))
    assert not grids_equal(a, ((1, 2),))
    assert not grids_equal(a, ((1, 2, 0), (3, 4, 0)))


def test_grid_to_ascii_contains_digits():
    g = ((1, 2, 3), (4, 5, 0))
    s = grid_to_ascii(g)
    for ch in "12345":
        assert ch in s
    # two lines for two rows
    assert s.count("\n") == 1


def test_render_task_to_ascii_real_task():
    task = load_task(_first_task_id())
    s = render_task_to_ascii(task)
    assert task.task_id in s
    assert "train[0]" in s
    assert "test[0]" in s


def test_save_grid_png(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    del plt  # only used to trigger skip when missing
    from orate.arc import save_grid_png

    g = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    out = tmp_path / "grid.png"
    save_grid_png(g, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_task_png(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    del plt
    from orate.arc import save_task_png

    task = load_task(_first_task_id())
    out = tmp_path / "task.png"
    save_task_png(task, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
