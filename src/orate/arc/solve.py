"""Act-4: the LLM writes its own @program to solve an ARC-AGI-2 task.

A ``solve_task(task)`` call runs ``propose_program`` — itself a
``@program`` — which yields DSL op choices and arguments. The result
is a ``Program`` AST. Phase-C whole-program retry wraps the proposer:
on any demo-mismatch, the program is rewound and re-invoked with a
``describe_mismatch`` string injected into the engine's context so
the next argmax moves to a different region of program-space.

This is the uppercut:

- The *inner* ``@program`` is authored at library-design time by us.
  Its yields emit ops and args; the engine's grammar-constrained
  decoding guarantees each Step is well-formed.
- The *output* of the inner ``@program`` is a ``Program`` AST — a
  *new* program, authored at runtime by the LLM, that transforms
  ARC grids.
- The outer retry loop rejects entire ASTs that don't match the
  training demonstrations, the same mechanism that filters scalar
  values in Act-2.

Programs that pass on the train pairs are consistent with the
demonstrations. Whether they generalize to the held-out test input is
what ARC-AGI measures — and that's the frontier of the demo.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from orate import gen, program, reject_program
from orate.arc.data import ArcTask
from orate.arc.dsl import OPS, Program, Step
from orate.arc.verify import describe_mismatch, verify_on_train
from orate.gen import Gen

# Ops are grouped by their arg-shape so the proposer can dispatch uniformly.
_ZERO_ARG_OPS = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
)
_TWO_COLOR_OPS = ("replace_color", "fill_background")
_TILE_OPS = ("tile_horizontal", "tile_vertical")


def _yield_no_args() -> Iterator[Gen]:
    """Zero-arg ops. Must be a generator for uniform ``yield from`` dispatch."""
    if False:  # pragma: no cover
        yield
    return ()


def _yield_two_color_args() -> Iterator[Gen]:
    old = yield gen.integer(
        0, 9, reject_message=lambda v: f"color {v} did not help for the 'from' side"
    )
    new = yield gen.integer(
        0, 9, reject_message=lambda v: f"color {v} did not help for the 'to' side"
    )
    return (old, new)


def _yield_tile_args() -> Iterator[Gen]:
    times = yield gen.integer(2, 4)
    return (times,)


def _yield_crop_args() -> Iterator[Gen]:
    background = yield gen.integer(0, 9)
    return (background,)


def _yield_recolor_args() -> Iterator[Gen]:
    old = yield gen.integer(0, 9)
    new = yield gen.integer(0, 9)
    # DSL expects tuple-of-(old, new) pairs; one pair is a reasonable first cut.
    return (((old, new),),)


def _yield_pad_args() -> Iterator[Gen]:
    top = yield gen.integer(0, 3)
    bottom = yield gen.integer(0, 3)
    left = yield gen.integer(0, 3)
    right = yield gen.integer(0, 3)
    fill = yield gen.integer(0, 9)
    return (top, bottom, left, right, fill)


def _yield_args_for(op_kind: str) -> Iterator[Gen]:
    """Dispatch a sub-coroutine that yields the right number of arg-gens."""
    if op_kind in _ZERO_ARG_OPS:
        return (yield from _yield_no_args())
    if op_kind in _TWO_COLOR_OPS:
        return (yield from _yield_two_color_args())
    if op_kind in _TILE_OPS:
        return (yield from _yield_tile_args())
    if op_kind == "crop_to_bbox":
        return (yield from _yield_crop_args())
    if op_kind == "recolor":
        return (yield from _yield_recolor_args())
    if op_kind == "pad":
        return (yield from _yield_pad_args())
    raise ValueError(f"unknown op kind: {op_kind!r}")


def make_propose_program(task: ArcTask, *, max_steps: int = 4, whole_program_retries: int = 20):
    """Build the proposer @program for a specific ARC task.

    The proposer is a fresh closure per task so it can reference the
    task's train demonstrations inside its body (for in-program
    verification before ``reject_program`` fires).
    """
    op_names = tuple(OPS.keys())

    @program(whole_program_retries=whole_program_retries)
    def propose() -> Iterator[Gen]:
        n_steps = yield gen.integer(
            1,
            max_steps,
            reject_message=lambda v: f"a {v}-step program was rejected",
        )
        steps: list[Step] = []
        for _ in range(n_steps):
            op_kind = yield gen.choice(list(op_names))
            args = yield from _yield_args_for(op_kind)
            steps.append(Step(kind=op_kind, args=args))
        candidate = Program(steps=tuple(steps))
        mismatches = verify_on_train(candidate, task)
        if mismatches:
            reject_program(describe_mismatch(candidate, task, mismatches[0]))
        return candidate

    return propose


@dataclass
class SolveResult:
    """Outcome of a ``solve_task`` call."""

    task_id: str
    solved: bool
    program: Program | None = None
    attempts: int = 0
    trace: list[dict] = field(default_factory=list)
    mismatches_at_exit: list[int] | None = None


def solve_task(
    task: ArcTask,
    *,
    engine: Any,
    max_steps: int = 4,
    whole_program_retries: int = 20,
) -> SolveResult:
    """Solve an ARC task by authoring a ``Program`` that matches all train demos.

    The engine must implement the orate ``Engine`` protocol; engines
    that also support ``inject_context`` get Phase-C context-injection
    between retries (strongly recommended for this task shape).

    Returns ``SolveResult`` with ``solved=True`` if a consistent program
    was found; on exhaustion, returns ``solved=False`` with the last
    attempt's trace so the caller can diagnose.
    """
    propose = make_propose_program(
        task,
        max_steps=max_steps,
        whole_program_retries=whole_program_retries,
    )
    invocation = propose()
    try:
        candidate = invocation.run(engine=engine)
    except Exception:
        return SolveResult(
            task_id=task.task_id,
            solved=False,
            attempts=len(invocation.trace),
            trace=invocation.trace,
        )
    return SolveResult(
        task_id=task.task_id,
        solved=True,
        program=candidate,
        attempts=len(invocation.trace),
        trace=invocation.trace,
        mismatches_at_exit=[],
    )


def apply_solution_to_test(program: Program, task: ArcTask) -> list:
    """Run the solved program on each test input, returning predicted grids."""
    from orate.arc.dsl import execute

    return [execute(program, inp) for inp, _ in task.test]
