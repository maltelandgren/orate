"""Report rendering for the BBH benchmark — joins baseline + orate JSONs.

The two runners drop sibling JSONs:

    bench/results/bbh_baseline_<subtask>_<stamp>.json
    bench/results/bbh_orate_<subtask>_<stamp>.json

Each file has a ``runs`` list keyed by ``index`` (the BBH dataset row).
We join on ``(subtask, index)`` and emit a single markdown report
suitable for ``bench/results/bbh_logical_deduction_<stamp>.md``.

Run as a CLI:

    .venv/bin/python bench/bbh/report.py --stamp 2026-04-25_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from bench.bbh.loader import SUBTASKS  # noqa: E402


@dataclass
class JoinedRow:
    subtask: str
    index: int
    target: str
    baseline_extracted: str | None
    baseline_correct: bool
    baseline_time_s: float
    baseline_response: str
    orate_extracted: str | None
    orate_correct: bool
    orate_time_s: float
    orate_premise_ok: int
    orate_premise_rejected: int
    orate_deduce_ok: int
    orate_deduce_rejected: int
    orate_answer_attempts: int
    orate_turn_end: str | None
    orate_trace: list


def _load(stamp: str, subtask: str, kind: str, results_dir: Path) -> dict:
    path = results_dir / f"bbh_{kind}_{subtask}_{stamp}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing trace: {path}")
    return json.loads(path.read_text())


def _maybe_load_oracle(subtask: str, stamp: str, results_dir: Path) -> dict | None:
    """Try to load the oracle trace using a couple of likely stamps.

    The oracle is fast (~8s for the whole dataset) and stamped
    independently. We try the same stamp first, then fall back to
    any oracle file matching the subtask.
    """
    candidates = [
        results_dir / f"bbh_oracle_{subtask}_{stamp}.json",
    ]
    candidates += sorted(results_dir.glob(f"bbh_oracle_{subtask}_*.json"), reverse=True)
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def join_subtask(stamp: str, subtask: str, results_dir: Path) -> tuple[dict, dict, list[JoinedRow]]:
    base = _load(stamp, subtask, "baseline", results_dir)
    orate = _load(stamp, subtask, "orate", results_dir)
    by_idx_o = {r["index"]: r for r in orate["runs"]}
    rows: list[JoinedRow] = []
    for b in base["runs"]:
        idx = b["index"]
        o = by_idx_o.get(idx)
        if o is None:
            continue
        rows.append(
            JoinedRow(
                subtask=subtask,
                index=idx,
                target=b["target"],
                baseline_extracted=b.get("extracted"),
                baseline_correct=bool(b.get("correct")),
                baseline_time_s=float(b.get("wall_time_s", 0.0)),
                baseline_response=b.get("response", ""),
                orate_extracted=o.get("extracted"),
                orate_correct=bool(o.get("correct")),
                orate_time_s=float(o.get("wall_time_s", 0.0)),
                orate_premise_ok=int(o.get("n_premise_ok", 0)),
                orate_premise_rejected=int(o.get("n_premise_rejected", 0)),
                orate_deduce_ok=int(o.get("n_deduce_ok", 0)),
                orate_deduce_rejected=int(o.get("n_deduce_rejected", 0)),
                orate_answer_attempts=int(o.get("n_answer_attempts", 0)),
                orate_turn_end=o.get("turn_end_reason"),
                orate_trace=o.get("trace", []),
            )
        )
    return base, orate, rows


def _summary_line(label: str, n_correct: int, n: int, time_s: float) -> str:
    pct = 100 * n_correct / max(1, n)
    return f"- **{label}:** {n_correct}/{n} = {pct:.1f}% ({time_s:.0f}s wall, {time_s / max(1, n):.1f}s/problem)"


def render_report(stamp: str, results_dir: Path, model_name: str) -> str:
    lines: list[str] = []
    lines.append("# BBH logical_deduction — free-text vs orate-augmented Qwen-7B")
    lines.append("")
    lines.append(f"- **Model:** `{model_name}`")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Decoding:** deterministic argmax under masked logits (XGrammar) for both modes.")
    lines.append("- **Stamp:** `" + stamp + "`")
    lines.append("")
    lines.append(
        "Two runs over the same problem set per subtask: free-text "
        "chain-of-thought (the official BBH 3-shot prompt) and orate-"
        "augmented (predicate-bound `@premise` / `@deduce` / `@answer` "
        "calls; see `bench/bbh/programs.py`)."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    headline_rows: list[str] = []
    aggregate_b = 0
    aggregate_o = 0
    aggregate_oracle = 0
    aggregate_n = 0
    for subtask in SUBTASKS:
        try:
            base, orate, rows = join_subtask(stamp, subtask, results_dir)
        except FileNotFoundError:
            continue
        n = len(rows)
        n_b = sum(1 for r in rows if r.baseline_correct)
        n_o = sum(1 for r in rows if r.orate_correct)
        b_pct = 100 * n_b / max(1, n)
        o_pct = 100 * n_o / max(1, n)
        delta = o_pct - b_pct
        sign = "+" if delta >= 0 else ""

        # Oracle: deterministic constraint-extractor + solver. We
        # restrict to the same problem indices for an apples-to-apples
        # cell. (If the oracle file is missing we just leave a dash.)
        oracle = _maybe_load_oracle(subtask, stamp, results_dir)
        oracle_cell = "—"
        oracle_n = 0
        if oracle is not None:
            idxs = {r.index for r in rows}
            oracle_runs = [r for r in oracle["runs"] if r["index"] in idxs]
            oracle_n = sum(1 for r in oracle_runs if r["correct"])
            if oracle_runs:
                oracle_cell = (
                    f"{oracle_n}/{len(oracle_runs)} "
                    f"({100 * oracle_n / len(oracle_runs):.1f}%)"
                )
                aggregate_oracle += oracle_n

        headline_rows.append(
            f"| `{subtask}` | {n} | {n_b} ({b_pct:.1f}%) | "
            f"{n_o} ({o_pct:.1f}%) | {sign}{delta:.1f} pts | {oracle_cell} |"
        )
        aggregate_b += n_b
        aggregate_o += n_o
        aggregate_n += n
    lines.append("| subtask | N | free-text CoT | orate | Δ | oracle (det.) |")
    lines.append("|---|---|---|---|---|---|")
    for r in headline_rows:
        lines.append(r)
    if aggregate_n > 0:
        agg_b_pct = 100 * aggregate_b / aggregate_n
        agg_o_pct = 100 * aggregate_o / aggregate_n
        delta = agg_o_pct - agg_b_pct
        sign = "+" if delta >= 0 else ""
        oracle_cell_total = (
            f"{aggregate_oracle}/{aggregate_n} "
            f"({100 * aggregate_oracle / aggregate_n:.1f}%)"
            if aggregate_oracle > 0 else "—"
        )
        lines.append(
            f"| **all** | **{aggregate_n}** | "
            f"**{aggregate_b} ({agg_b_pct:.1f}%)** | "
            f"**{aggregate_o} ({agg_o_pct:.1f}%)** | "
            f"**{sign}{delta:.1f} pts** | **{oracle_cell_total}** |"
        )
    lines.append("")
    lines.append(
        "*Oracle = the deterministic English-constraint parser + "
        "permutation solver in `bench/bbh/constraint_parser.py`. It "
        "doesn't run the LLM; the column is included as the upper bound "
        "achievable when every constraint and option is parseable.*"
    )
    lines.append("")

    lines.append("## Per-subtask detail")
    for subtask in SUBTASKS:
        try:
            base, orate, rows = join_subtask(stamp, subtask, results_dir)
        except FileNotFoundError:
            continue
        n = len(rows)
        n_b = sum(1 for r in rows if r.baseline_correct)
        n_o = sum(1 for r in rows if r.orate_correct)
        b_time = sum(r.baseline_time_s for r in rows)
        o_time = sum(r.orate_time_s for r in rows)
        prem_ok = sum(r.orate_premise_ok for r in rows)
        prem_rej = sum(r.orate_premise_rejected for r in rows)
        ded_ok = sum(r.orate_deduce_ok for r in rows)
        ded_rej = sum(r.orate_deduce_rejected for r in rows)
        ans_att = sum(r.orate_answer_attempts for r in rows)
        n_unanswered = sum(1 for r in rows if r.orate_extracted is None)

        lines.append("")
        lines.append(f"### `{subtask}`")
        lines.append("")
        lines.append(_summary_line("Free-text CoT", n_b, n, b_time))
        lines.append(_summary_line("Orate-augmented", n_o, n, o_time))
        lines.append(
            f"- **Orate predicate trace:** "
            f"{prem_ok} premises accepted ({prem_rej} rejected), "
            f"{ded_ok} deductions accepted ({ded_rej} rejected), "
            f"{ans_att} answer attempts, "
            f"{n_unanswered} problems left unanswered."
        )
        lines.append("")

        # Examples: pick a win, a loss, an interesting case
        wins = [r for r in rows if r.orate_correct and not r.baseline_correct]
        losses = [r for r in rows if r.baseline_correct and not r.orate_correct]
        unanswered = [r for r in rows if r.orate_extracted is None]
        if wins:
            r = wins[0]
            lines.append(f"**Example: orate wins where baseline lost (idx {r.index}, target {r.target})**")
            lines.append("")
            lines.append("- Baseline: `{}` ✗".format(r.baseline_extracted or "(no parse)"))
            lines.append(f"- Orate: `{r.orate_extracted}` ✓")
            lines.append(f"- Trace: {_summarise_trace(r.orate_trace)}")
            lines.append("")
        if losses:
            r = losses[0]
            lines.append(f"**Example: baseline wins where orate lost (idx {r.index}, target {r.target})**")
            lines.append("")
            lines.append("- Baseline: `{}` ✓".format(r.baseline_extracted or "(no parse)"))
            lines.append("- Orate: `{}` ✗".format(r.orate_extracted or "(unanswered)"))
            lines.append(f"- Trace: {_summarise_trace(r.orate_trace)}")
            lines.append(f"- End reason: `{r.orate_turn_end}`")
            lines.append("")
        if unanswered:
            r = unanswered[0]
            lines.append(f"**Example: orate failed to answer (idx {r.index}, target {r.target})**")
            lines.append("")
            lines.append(f"- Trace: {_summarise_trace(r.orate_trace)}")
            lines.append(f"- End reason: `{r.orate_turn_end}`")
            lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Orate intervention shape (Path C).** Three predicate-bound "
        "leaves: `@premise(predicate, args)` enforces that the asserted "
        "fact parses and is consistent with prior premises; "
        "`@deduce(predicate, args)` enforces that the new fact is *forced* "
        "by the union of premises + previous deductions; `@answer(letter)` "
        "enforces that the chosen option is entailed. The `derivable` "
        "check is a brute-force permutation enumerator (≤ 7! = 5040), "
        "which is the analogue of `derivable_under` from the legal-step "
        "demos."
    )
    lines.append(
        "- **Extraction.** Items and option list are extracted "
        "deterministically by regex (`bench/bbh/extractor.py`) — 100% "
        "coverage on the observed BBH problems. The model must still "
        "translate the *constraints* themselves into machine-readable "
        "facts via `@premise` calls; that is where the orate enforcement "
        "bites."
    )
    lines.append(
        "- **Grading.** Free-text uses the standard BBH grader (last "
        "`(X)` letter in the response). Orate uses the `@answer` call's "
        "`letter` arg directly. Both modes are compared on the same "
        "problem indices."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _summarise_trace(trace: list) -> str:
    """One-line summary of a trace: counts of premise/deduce/answer ok+rej."""
    counts: dict[str, int] = {}
    for t in trace:
        if "name" not in t:
            continue
        key = f"{t['name']}{'-rej' if t.get('rejected') else ''}"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--results-dir", default=str(_REPO / "bench" / "results"))
    ap.add_argument(
        "--model-name", default="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        help="Model name to embed in the report (just for display)",
    )
    ap.add_argument("--out", default=None, help="Override output path; default uses stamp")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    text = render_report(args.stamp, results_dir, args.model_name)
    out = Path(args.out) if args.out else results_dir / f"bbh_logical_deduction_{args.stamp}.md"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
