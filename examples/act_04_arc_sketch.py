"""Act 4 — The LLM writes its own @program.

For a new task, the model authors its own transformation rule at
runtime: a Program AST over a small grid DSL, grammar-constrained by
the same machinery that filtered scalar values in Act 2. The program
is executed on the training demonstrations; mismatches tighten the
program-level grammar and inject the grid-diff as natural-language
context. The model's next argmax moves to a different region of
*program*-space.

This script runs the uppercut against two tasks:

1. A synthetic "flip horizontal" task — designed so a one-op program
   (`flip_horizontal`) solves it. MockEngine may or may not find it in
   the retry budget; the *mechanism* is what the demo showcases.
2. An optional real ARC-AGI-2 task loaded from `arc-data/` (requires
   the ARC-AGI-2 repo cloned there; see README).

Under MockEngine this will not reliably solve tasks — it's a random
sampler. The win-condition is demonstrated visibly in the retry trace:
proposer proposes, verifier rejects, context note appears, next
attempt is different. Swap in the local XGrammarEngine with Qwen2.5
(or the OpenRouter engine with Opus 4.7) to see actual convergence.
"""

from __future__ import annotations

import sys
from pathlib import Path

from orate.arc.data import ArcTask, load_task
from orate.arc.render import grid_to_ascii
from orate.arc.solve import apply_solution_to_test, solve_task
from orate.engine.mock import MockEngine


def _flip_horizontal_task() -> ArcTask:
    inp = ((1, 2, 3), (4, 5, 6))
    out = ((3, 2, 1), (6, 5, 4))
    return ArcTask(
        task_id="synth-flip-h",
        train=((inp, out), (inp, out)),
        test=((inp, out),),
    )


def _print_trace(trace: list[dict]) -> None:
    """Short human-readable rendering of the retry trace."""
    for i, entry in enumerate(trace):
        if entry["status"] == "ok":
            print(f"  [{i}] ✓ accepted")
        else:
            reason = entry.get("reason", "").strip()
            if len(reason) > 80:
                reason = reason[:77] + "..."
            print(f"  [{i}] ✗ rejected — {reason}")


def _run_on(task: ArcTask, seed: int = 0, budget: int = 30) -> None:
    print(f"Task: {task.task_id}")
    print()
    print("Training demonstrations:")
    for i, (inp, out) in enumerate(task.train):
        print(f"  demo {i} input:")
        print("    " + grid_to_ascii(inp).replace("\n", "\n    "))
        print(f"  demo {i} expected output:")
        print("    " + grid_to_ascii(out).replace("\n", "\n    "))
        print()

    engine = MockEngine(seed=seed)
    result = solve_task(
        task,
        engine=engine,
        whole_program_retries=budget,
        max_steps=3,
    )

    print(f"Attempts: {result.attempts}  (budget: {budget + 1})")
    print(f"Solved: {result.solved}")
    if result.solved and result.program is not None:
        print(f"Found program: {result.program!r}")
        preds = apply_solution_to_test(result.program, task)
        for i, pred in enumerate(preds):
            print(f"  test {i} prediction:")
            print("    " + grid_to_ascii(pred).replace("\n", "\n    "))
    print()
    print("Retry trace (first 10):")
    _print_trace(result.trace[:10])
    if len(result.trace) > 10:
        print(f"  ... ({len(result.trace) - 10} more)")

    # Show any Phase-C notes injected into the session. With a real
    # engine this is the steering signal the model sees.
    if engine._context:
        print()
        print("Phase-C context injected (first 3):")
        for note in engine._context[:3]:
            print(f"  {note}")


def main() -> None:
    print("Act 4 — the LLM writes its own @program to solve an ARC task.")
    print()
    print("=" * 70)

    # Task 1: synthetic, designed to be easy. MockEngine should find it
    # within a modest retry budget — *sometimes*. The mechanism is what
    # the demo showcases.
    _run_on(_flip_horizontal_task(), seed=0, budget=40)

    # Task 2: real ARC-AGI-2 task. Only runs if data is cloned locally.
    arc_root = Path("arc-data/ARC-AGI-2/data")
    if not arc_root.exists():
        print()
        print("(Skipping real ARC task — arc-data/ARC-AGI-2 not cloned.)")
        print("Clone with: git clone https://github.com/arcprize/ARC-AGI-2 arc-data/ARC-AGI-2")
        return

    print()
    print("=" * 70)
    arg_task_id = sys.argv[1] if len(sys.argv) > 1 else None
    if arg_task_id is None:
        # Pick the first evaluation task deterministically for the demo.
        eval_tasks = sorted(p.stem for p in (arc_root / "evaluation").glob("*.json"))
        if not eval_tasks:
            print("(No evaluation tasks found.)")
            return
        arg_task_id = eval_tasks[0]

    task = load_task(arg_task_id, split="evaluation")
    _run_on(task, seed=0, budget=25)

    print()
    print("Under MockEngine, real ARC tasks won't reliably solve — it's random.")
    print("Swap engine=XGrammarEngine(...) or engine=OpenRouterEngine(...) for real runs.")


if __name__ == "__main__":
    main()
