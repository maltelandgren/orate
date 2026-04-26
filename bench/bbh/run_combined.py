"""Combined baseline + orate runner — load Qwen-7B once, run both modes.

Saves ~30s per subtask vs invoking the baseline and orate scripts
separately (each engine load takes that long). More importantly, lets
us cap problem counts per-mode-per-subtask flexibly so we can ship
results within the time budget when other GPU users contend on the
same Apple Silicon box.

For each (subtask, problem):
  1. Run free-text Qwen-7B chain-of-thought (`run_baseline.run_problem`).
  2. Run orate-augmented (`run_orate.run_problem`).
  3. Append both to per-subtask traces.

Saves ``bench/results/bbh_baseline_<subtask>_<stamp>.json`` and the
matching ``bbh_orate_<subtask>_<stamp>.json`` so the report renderer
(`bbh.report`) joins them transparently.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bench"))

from bbh.loader import SUBTASKS, load_subtask  # noqa: E402
from bbh.run_baseline import (  # noqa: E402
    run_problem as run_baseline_problem,
)
from bbh.run_orate import run_problem as run_orate_problem  # noqa: E402

from orate.engine.xgrammar import XGrammarEngine  # noqa: E402


def _pick_model() -> str:
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("no local Qwen2.5 GGUF found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=list(SUBTASKS) + ["all"], default="all")
    parser.add_argument(
        "--limits", type=str, default="50,50,20",
        help="Comma-separated per-subtask limits, in canonical order "
             "(three / five / seven). Default: 50,50,20.",
    )
    parser.add_argument("--max-tokens-baseline", type=int, default=384)
    parser.add_argument("--max-turn-tokens-orate", type=int, default=1024)
    parser.add_argument("--max-calls-orate", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--out-dir", type=str, default=str(_REPO / "bench" / "results"))
    parser.add_argument("--stamp", type=str, default=None)
    parser.add_argument(
        "--only", choices=["baseline", "orate", "both"], default="both",
        help="Skip a mode if you already have its trace from a prior run.",
    )
    parser.add_argument(
        "--answer-only", action="store_true",
        help="Use the answer-only orate variant (predicate forces the unique correct letter).",
    )
    args = parser.parse_args()

    limits = [int(x) for x in args.limits.split(",")]
    while len(limits) < len(SUBTASKS):
        limits.append(50)

    subtasks = list(SUBTASKS) if args.subtask == "all" else [args.subtask]

    model = _pick_model()
    print(f"=== Loading {Path(model).name} ===", flush=True)
    engine = XGrammarEngine(
        model_path=model,
        max_tokens_per_sample=max(args.max_tokens_baseline, args.max_turn_tokens_orate),
        n_ctx=args.n_ctx,
    )
    engine.load()
    engine.warm()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or datetime.now().strftime("%Y-%m-%d_%H%M")

    for subtask in subtasks:
        idx = SUBTASKS.index(subtask)
        limit = limits[idx]
        print(f"\n=== {subtask} (limit={limit}) ===", flush=True)
        problems = load_subtask(subtask, limit=limit)

        baseline_runs = []
        orate_runs = []
        n_b_correct = 0
        n_o_correct = 0
        t_subtask = time.perf_counter()

        for i, p in enumerate(problems):
            line = f"  [{i + 1:>3}/{len(problems)}] target={p.target}"

            if args.only in ("baseline", "both"):
                rb = run_baseline_problem(engine, p, max_tokens=args.max_tokens_baseline)
                baseline_runs.append(rb)
                if rb.correct:
                    n_b_correct += 1
                line += f"  base:{'✓' if rb.correct else '✗'} {rb.extracted} ({rb.wall_time_s:.0f}s)"

            if args.only in ("orate", "both"):
                ro = run_orate_problem(
                    engine, p,
                    max_turn_tokens=args.max_turn_tokens_orate,
                    max_calls=args.max_calls_orate,
                    answer_only=args.answer_only,
                )
                orate_runs.append(ro)
                if ro.correct:
                    n_o_correct += 1
                line += (
                    f"  orate:{'✓' if ro.correct else '✗'} {ro.extracted} "
                    f"({ro.wall_time_s:.0f}s, "
                    f"p={ro.n_premise_ok}/{ro.n_premise_ok + ro.n_premise_rejected}, "
                    f"d={ro.n_deduce_ok}/{ro.n_deduce_ok + ro.n_deduce_rejected}, "
                    f"end={ro.turn_end_reason})"
                )

            print(line, flush=True)

        elapsed = time.perf_counter() - t_subtask
        if baseline_runs:
            print(
                f"  baseline: {n_b_correct}/{len(baseline_runs)} = "
                f"{100 * n_b_correct / max(1, len(baseline_runs)):.1f}%",
                flush=True,
            )
        if orate_runs:
            print(
                f"  orate:    {n_o_correct}/{len(orate_runs)} = "
                f"{100 * n_o_correct / max(1, len(orate_runs)):.1f}%",
                flush=True,
            )
        print(f"  total wall: {elapsed:.0f}s", flush=True)

        if baseline_runs:
            path = out_dir / f"bbh_baseline_{subtask}_{stamp}.json"
            path.write_text(json.dumps(
                {
                    "model": model, "subtask": subtask, "stamp": stamp,
                    "n_problems": len(baseline_runs),
                    "n_correct": n_b_correct,
                    "accuracy": n_b_correct / max(1, len(baseline_runs)),
                    "wall_time_s": elapsed,
                    "max_tokens": args.max_tokens_baseline,
                    "runs": [asdict(r) for r in baseline_runs],
                },
                indent=2,
            ))
            print(f"  wrote {path}", flush=True)
        if orate_runs:
            path = out_dir / f"bbh_orate_{subtask}_{stamp}.json"
            path.write_text(json.dumps(
                {
                    "model": model, "subtask": subtask, "stamp": stamp,
                    "n_problems": len(orate_runs),
                    "n_correct": n_o_correct,
                    "accuracy": n_o_correct / max(1, len(orate_runs)),
                    "wall_time_s": elapsed,
                    "max_turn_tokens": args.max_turn_tokens_orate,
                    "max_calls": args.max_calls_orate,
                    "runs": [asdict(r) for r in orate_runs],
                },
                indent=2,
            ))
            print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
