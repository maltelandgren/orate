"""Free-text Qwen-7B chain-of-thought baseline on BBH logical_deduction.

Pipeline per problem:

  1. Wrap the BBH 3-shot CoT prompt + the new question in Qwen 2.5's
     chat template (``<|im_start|>system ... <|im_end|>``...) so the
     model sees a normal conversation instead of orate's terse
     ``<|user|>`` markers.
  2. Prime the engine with the templated text.
  3. Sample under a permissive printable-ASCII + newline grammar up to
     ``--max-tokens`` tokens. Argmax decoding. ``<|im_end|>`` is *not*
     in the grammar, so the model will run to ``max_tokens`` — we trim
     after extracting the answer.
  4. Grade by ``extract_answer_letter`` (last ``(X)`` token).

Saves a JSON trace per subtask: ``bench/results/bbh_baseline_<subtask>_<stamp>.json``
with one row per problem (input / response / extracted / target / correct
/ wall_time_s). The orate runner consumes these as fixtures so the two
modes are graded against the same problem set.
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

from bbh.loader import (  # noqa: E402
    SUBTASKS,
    BBHProblem,
    extract_answer_letter,
    load_subtask,
)
from bbh.prompts import build_bbh_user_message  # noqa: E402

from orate.engine.xgrammar import XGrammarEngine  # noqa: E402

# Permissive printable-ASCII + newline grammar. Same shape as
# ``measure_legal_steps._FREE_TEXT_GRAMMAR``.
_FREE_TEXT_GRAMMAR = """\
root ::= chunk+
chunk ::= [ !-~] | "\\n"
"""


@dataclass
class BaselineRun:
    subtask: str
    index: int
    target: str
    response: str
    extracted: str | None
    correct: bool
    wall_time_s: float
    output_chars: int


def _qwen_chat_prompt(user_message: str, system: str = "") -> str:
    """Render a Qwen 2.5 chat-template prompt that ends ready for the model."""
    sys_block = (
        f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    )
    return (
        f"{sys_block}"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _pick_model() -> str:
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("no local Qwen2.5 GGUF found")


def run_problem(
    engine: XGrammarEngine,
    problem: BBHProblem,
    *,
    max_tokens: int,
) -> BaselineRun:
    """Prime + sample one BBH problem under free-text decoding."""
    user_msg = build_bbh_user_message(problem.question)
    prompt = _qwen_chat_prompt(user_msg)
    engine.prime(prompt)
    t0 = time.perf_counter()
    response = engine.sample_grammar(_FREE_TEXT_GRAMMAR, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0
    extracted = extract_answer_letter(response)
    return BaselineRun(
        subtask=problem.subtask,
        index=problem.index,
        target=problem.target,
        response=response,
        extracted=extracted,
        correct=extracted is not None and extracted == problem.target.upper(),
        wall_time_s=elapsed,
        output_chars=len(response),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subtask",
        choices=list(SUBTASKS) + ["all"],
        default="all",
        help="Which BBH logical_deduction subtask to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N problems")
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max free-text tokens per problem (default 512)",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=8192,
        help="Context window for the engine (default 8192)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(_REPO / "bench" / "results"),
        help="Where to write the JSON trace",
    )
    parser.add_argument(
        "--stamp", type=str, default=None,
        help="Override the output filename stamp (default: now)",
    )
    args = parser.parse_args()

    subtasks = list(SUBTASKS) if args.subtask == "all" else [args.subtask]

    model = _pick_model()
    print(f"=== Loading {Path(model).name} ===", flush=True)
    engine = XGrammarEngine(
        model_path=model,
        max_tokens_per_sample=args.max_tokens,
        n_ctx=args.n_ctx,
    )
    engine.load()
    engine.warm()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or datetime.now().strftime("%Y-%m-%d_%H%M")

    for subtask in subtasks:
        print(f"\n=== {subtask} ===", flush=True)
        problems = load_subtask(subtask, limit=args.limit)
        runs: list[BaselineRun] = []
        n_correct = 0
        t0 = time.perf_counter()

        for i, p in enumerate(problems):
            r = run_problem(engine, p, max_tokens=args.max_tokens)
            runs.append(r)
            if r.correct:
                n_correct += 1
            mark = "✓" if r.correct else "✗"
            print(
                f"  [{i + 1:>3}/{len(problems)}] {mark} target={p.target} "
                f"got={r.extracted} ({r.wall_time_s:.1f}s)",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        print(
            f"  -> {n_correct}/{len(problems)} = "
            f"{100 * n_correct / max(1, len(problems)):.1f}%  "
            f"({elapsed:.0f}s total)",
            flush=True,
        )

        out_path = out_dir / f"bbh_baseline_{subtask}_{stamp}.json"
        out_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "subtask": subtask,
                    "stamp": stamp,
                    "n_problems": len(problems),
                    "n_correct": n_correct,
                    "accuracy": n_correct / max(1, len(problems)),
                    "wall_time_s": elapsed,
                    "max_tokens": args.max_tokens,
                    "runs": [asdict(r) for r in runs],
                },
                indent=2,
            )
        )
        print(f"  -> wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
