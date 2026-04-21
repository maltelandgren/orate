"""ARC-AGI-2 task loading, DSL, and program synthesis."""

from orate.arc.data import (
    ArcTask,
    Grid,
    grid_shape,
    grids_equal,
    list_tasks,
    load_task,
)
from orate.arc.dsl import (
    OPS,
    ExecutionError,
    OpKind,
    Program,
    Step,
    execute,
)
from orate.arc.render import (
    grid_to_ascii,
    render_task_to_ascii,
    save_grid_png,
    save_task_png,
)
from orate.arc.solve import (
    SolveResult,
    apply_solution_to_test,
    make_propose_program,
    solve_task,
)
from orate.arc.verify import describe_mismatch, verify_on_train

__all__ = [
    "OPS",
    "ArcTask",
    "ExecutionError",
    "Grid",
    "OpKind",
    "Program",
    "SolveResult",
    "Step",
    "apply_solution_to_test",
    "describe_mismatch",
    "execute",
    "grid_shape",
    "grid_to_ascii",
    "grids_equal",
    "list_tasks",
    "load_task",
    "make_propose_program",
    "render_task_to_ascii",
    "save_grid_png",
    "save_task_png",
    "solve_task",
    "verify_on_train",
]
