"""ASCII and PNG rendering for ARC grids.

ASCII uses a 2-char-wide cell so grids line up in monospace terminals. Each
cell prints the digit followed by a space; blank (0) cells use "``. ``" to give
a visual "background" feel while staying ANSI-free.

PNG rendering uses the canonical ARC 10-color palette via matplotlib. The
matplotlib import is deferred so the ``orate.arc`` module is usable even when
matplotlib is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orate.arc.data import ArcTask, Grid, grid_shape

if TYPE_CHECKING:
    pass

# Canonical ARC palette (fchollet/ARC): indices 0-9.
ARC_PALETTE: tuple[str, ...] = (
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 cyan
    "#870C25",  # 9 brown
)


def _cell(v: int) -> str:
    """2-char cell: digit + space; 0 rendered as dot for readability."""
    if v == 0:
        return ". "
    return f"{v} "


def grid_to_ascii(g: Grid) -> str:
    """Monospace-friendly ASCII rendering. Each cell is 2 chars wide."""
    if not g:
        return ""
    return "\n".join("".join(_cell(v) for v in row).rstrip() for row in g)


def _ascii_lines(g: Grid) -> list[str]:
    rows, cols = grid_shape(g)
    width = cols * 2
    if rows == 0:
        return [""]
    return [("".join(_cell(v) for v in row)).ljust(width) for row in g]


def _side_by_side(left: Grid, right: Grid, sep: str = "  ->  ") -> str:
    l_lines = _ascii_lines(left)
    r_lines = _ascii_lines(right)
    h = max(len(l_lines), len(r_lines))
    l_width = max((len(x) for x in l_lines), default=0)
    r_width = max((len(x) for x in r_lines), default=0)
    l_lines += [" " * l_width] * (h - len(l_lines))
    r_lines += [" " * r_width] * (h - len(r_lines))
    mid = len(l_lines) // 2
    out = []
    for i, (a, b) in enumerate(zip(l_lines, r_lines, strict=True)):
        joiner = sep if i == mid else " " * len(sep)
        out.append(f"{a.ljust(l_width)}{joiner}{b.ljust(r_width)}")
    return "\n".join(out)


def render_task_to_ascii(task: ArcTask) -> str:
    """Render all train pairs and the test input(s) to a single string."""
    parts: list[str] = [f"task {task.task_id}"]
    for i, (inp, out) in enumerate(task.train):
        parts.append(f"\ntrain[{i}]  input -> output")
        parts.append(_side_by_side(inp, out))
    for i, (inp, out) in enumerate(task.test):
        parts.append(f"\ntest[{i}]  input")
        parts.append(grid_to_ascii(inp))
        if out is not None:
            parts.append(f"test[{i}]  expected output")
            parts.append(grid_to_ascii(out))
    return "\n".join(parts)


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for PNG rendering. Install with: pip install 'orate[arc]'"
        ) from e
    return plt, ListedColormap


def _draw_grid(ax, g: Grid, cmap, title: str | None = None) -> None:
    rows, cols = grid_shape(g)
    data = [list(row) for row in g] if rows else [[0]]
    ax.imshow(data, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    # Thin grid overlay.
    ax.set_xticks([x - 0.5 for x in range(1, cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, rows)], minor=True)
    ax.grid(which="minor", color="#333333", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    if title:
        ax.set_title(title, fontsize=9)


def save_grid_png(g: Grid, path: str) -> None:
    """Save a single grid to PNG using the canonical ARC palette."""
    plt, ListedColormap = _require_matplotlib()
    cmap = ListedColormap(list(ARC_PALETTE))
    rows, cols = grid_shape(g)
    fig, ax = plt.subplots(figsize=(max(1.0, cols * 0.3), max(1.0, rows * 0.3)))
    _draw_grid(ax, g, cmap)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_task_png(task: ArcTask, path: str) -> None:
    """Render all train pairs + test input(s) as a single PNG."""
    plt, ListedColormap = _require_matplotlib()
    cmap = ListedColormap(list(ARC_PALETTE))

    n_train = len(task.train)
    n_test = len(task.test)
    # Layout: one row per train pair (input, output), plus rows for test inputs.
    n_rows = n_train + n_test
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.2),
        squeeze=False,
    )
    for i, (inp, out) in enumerate(task.train):
        _draw_grid(axes[i][0], inp, cmap, title=f"train[{i}] in")
        _draw_grid(axes[i][1], out, cmap, title=f"train[{i}] out")
    for j, (inp, out) in enumerate(task.test):
        row = n_train + j
        _draw_grid(axes[row][0], inp, cmap, title=f"test[{j}] in")
        if out is not None:
            _draw_grid(axes[row][1], out, cmap, title=f"test[{j}] out")
        else:
            axes[row][1].axis("off")

    fig.suptitle(f"task {task.task_id}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
