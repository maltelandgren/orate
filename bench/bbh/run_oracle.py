"""Deterministic-solver upper bound for BBH logical_deduction.

No LLM. Just the regex-based extraction (extractor + constraint_parser)
+ the brute-force permutation solver. We grade by checking each option
text in turn; the option whose Fact is forced under the extracted
premises is our answer.

This is the oracle / upper bound for "Path A" (extract → encode →
enforce). It doesn't measure the model — it measures how complete the
deterministic English parser is. We confirmed earlier that this gets
100% on all 750 problems, so the oracle is included in the report as
a context anchor (NOT as a model-vs-orate comparison).

Run:
    .venv/bin/python bench/bbh/run_oracle.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bench"))

from bbh.constraint_parser import extract_premises  # noqa: E402
from bbh.extractor import extract_problem  # noqa: E402
from bbh.loader import SUBTASK_OBJECT_COUNT, SUBTASKS, load_subtask  # noqa: E402
from bbh.ordering import Fact, derivable, parse_option  # noqa: E402


@dataclass
class OracleRun:
    subtask: str
    index: int
    target: str
    extracted: str | None
    correct: bool
    n_premises: int


def solve_problem(question: str, items: list[str], options: dict[str, str], n_items: int) -> tuple[str | None, list[Fact]]:
    """Return (best_letter, premises) for ``question``.

    Best letter: the option whose parsed Fact is forced by the extracted
    premises. If no option is forced (extraction incomplete) or several
    are forced (which shouldn't happen given puzzles have a unique
    solution), returns None.
    """
    premises = extract_premises(question, n_items)
    forced: list[str] = []
    for letter, text in options.items():
        f = parse_option(text, n_items)
        if f is None:
            continue
        if derivable(items, premises, f):
            forced.append(letter)
    if len(forced) == 1:
        return forced[0], premises
    return None, premises


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=list(SUBTASKS) + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=str(_REPO / "bench" / "results"))
    parser.add_argument("--stamp", type=str, default=None)
    args = parser.parse_args()

    subtasks = list(SUBTASKS) if args.subtask == "all" else [args.subtask]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or datetime.now().strftime("%Y-%m-%d_%H%M")

    for subtask in subtasks:
        problems = load_subtask(subtask, limit=args.limit)
        n_items = SUBTASK_OBJECT_COUNT[subtask]
        runs: list[OracleRun] = []
        n_correct = 0
        t0 = time.perf_counter()
        for p in problems:
            ex = extract_problem(p.question)
            letter_only, premises = solve_problem(p.question, ex.items, ex.options, n_items)
            # ``letter_only`` is already in the "(X)" form (the keys of
            # ex.options). No extra wrapping.
            extracted = letter_only.upper() if letter_only else None
            correct = extracted == p.target.upper() if extracted else False
            if correct:
                n_correct += 1
            runs.append(OracleRun(
                subtask=p.subtask, index=p.index, target=p.target,
                extracted=extracted, correct=correct,
                n_premises=len(premises),
            ))
        elapsed = time.perf_counter() - t0
        print(f"{subtask}: {n_correct}/{len(problems)} = "
              f"{100 * n_correct / max(1, len(problems)):.1f}% "
              f"({elapsed:.2f}s total)")
        out_path = out_dir / f"bbh_oracle_{subtask}_{stamp}.json"
        out_path.write_text(json.dumps(
            {
                "subtask": subtask, "stamp": stamp,
                "n_problems": len(problems), "n_correct": n_correct,
                "accuracy": n_correct / max(1, len(problems)),
                "wall_time_s": elapsed,
                "runs": [asdict(r) for r in runs],
            },
            indent=2,
        ))


if __name__ == "__main__":
    main()
