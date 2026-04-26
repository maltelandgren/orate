"""Benchmark: free-text vs constrained Qwen-7B on legal-step problems.

For each problem in the suite, runs the same model under two modes:

- **Free text.** Prompt the model with the problem and an instruction
  to end with ``ANSWER: <var> = <int>``. Sample under a permissive
  printable-ASCII grammar up to ``--max-tokens`` tokens. Parse the
  final answer with a regex; correctness is exact-match against the
  problem's expected answer.

- **Constrained.** Run a Session with ``algebra_step`` + ``done``
  registered. The model can only emit grammar-valid @-calls;
  ``equivalent_under`` rejects any call where the (rule, after) pair
  isn't algebraically equivalent to ``before``. The final answer
  comes from the ``@done`` payload.

Both modes are deterministic argmax decoding under the same model.

Outputs ``bench/results/legal_steps_<date>.{json,md}``: a JSON dump
of every run's transcript + timing, and a markdown table summary
fit for pasting into the video script or production status doc.

Run:
    .venv/bin/python bench/measure_legal_steps.py            # full suite
    .venv/bin/python bench/measure_legal_steps.py --quick    # just 2 problems
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Make examples/ importable so we can pick up algebra_step + done.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "examples"))

from legal_steps.algebra import algebra_step, done  # type: ignore[import-not-found]  # noqa: E402

from orate import (  # noqa: E402
    FreeText,
    NewProgramRegistered,
    ProgramInvoked,
    Session,
    TurnEnded,
)
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402

# ---- problem suite ----------------------------------------------------


@dataclass
class Problem:
    """One algebra problem.

    ``expr`` is the equation as the user states it (also embedded in the
    prompts). ``var`` is the variable to solve for. ``expected`` is the
    integer answer we exact-match on. ``difficulty`` is editorial — for
    the markdown table.
    """

    name: str
    expr: str
    var: str
    expected: int
    difficulty: str  # "easy" | "medium" | "hard"


SUITE: list[Problem] = [
    Problem("eq_3x_plus_5", "3x + 5 = 14", "x", 3, "easy"),
    Problem("eq_2x_eq_6", "2x = 6", "x", 3, "easy"),
    Problem("eq_x_plus_7", "x + 7 = 12", "x", 5, "easy"),
    Problem("eq_both_sides", "2x + 3 = x + 9", "x", 6, "medium"),
    Problem("eq_distribute", "3(x + 1) = 12", "x", 3, "medium"),
    Problem("eq_negative", "5 - 2x = 1", "x", 2, "medium"),
    Problem("eq_two_step", "4(x - 2) + 1 = 13", "x", 5, "hard"),
    # Round it out to a clean 10. Each new entry is linear (within
    # the algebra_step rule set: simplify / combine_like /
    # isolate_var / evaluate) and chosen to either differentiate
    # free-text (sign/distribute slips) or test the boost in retry
    # budget.
    Problem("eq_7x_minus_5", "7x - 5 = 16", "x", 3, "easy"),
    Problem("eq_both_sides_b", "4x + 7 = 2x + 13", "x", 3, "medium"),
    Problem("eq_distribute_b", "2(x - 3) = 8", "x", 7, "medium"),
]


# ---- free-text ---------------------------------------------------------


_FREE_TEXT_GRAMMAR = """\
root ::= chunk+
chunk ::= [ !-~] | "\\n"
"""


def _free_text_prompt(problem: Problem) -> str:
    return (
        "You are a careful arithmetic assistant. Solve the equation step "
        f"by step. End with a single line in the exact form 'ANSWER: "
        f"{problem.var} = <integer>'.\n\n"
        f"Equation: {problem.expr}\n\nWork:\n"
    )


_ANSWER_RE = re.compile(r"ANSWER:\s*([a-z])\s*=\s*(-?\d+)", re.IGNORECASE)


def _parse_free_text_answer(output: str, var: str) -> int | None:
    """Extract the final integer answer from a free-text response.

    Looks for the LAST ``ANSWER: <var> = <int>`` line (the model is
    instructed to put it last, but might write multiple). Falls back to
    the last bare ``<var> = <int>`` we find anywhere in the text.
    """
    matches = list(_ANSWER_RE.finditer(output))
    if matches:
        last = matches[-1]
        if last.group(1).lower() == var.lower():
            try:
                return int(last.group(2))
            except ValueError:
                return None
    bare = re.findall(rf"(?:^|\W){re.escape(var)}\s*=\s*(-?\d+)", output)
    if bare:
        try:
            return int(bare[-1])
        except ValueError:
            return None
    return None


@dataclass
class FreeTextRun:
    problem: str
    output: str
    parsed_answer: int | None
    correct: bool
    wall_time_s: float
    output_tokens: int  # rough: bytes // 3


def run_free_text(engine: XGrammarEngine, problem: Problem, max_tokens: int) -> FreeTextRun:
    """Prime the engine with a free-text prompt; sample under a permissive grammar."""
    engine.prime(_free_text_prompt(problem))
    t0 = time.perf_counter()
    output = engine.sample_grammar(_FREE_TEXT_GRAMMAR, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0
    parsed = _parse_free_text_answer(output, problem.var)
    return FreeTextRun(
        problem=problem.name,
        output=output,
        parsed_answer=parsed,
        correct=parsed == problem.expected,
        wall_time_s=elapsed,
        output_tokens=max(1, len(output) // 3),
    )


# ---- constrained -------------------------------------------------------


_CONSTRAINED_SYSTEM = """\
You output ONLY @-calls. No markdown, no prose, no commentary.

Available calls:
  @algebra_step("before", rule, "after")
  @done(<integer>)

The runtime mathematically verifies that `after` equals `before`
under `rule`. If not, the call is rejected.

Rules and what each one MEANS — pick precisely:
  simplify     — algebraic equivalence; e.g. distribute, move constants,
                 cancel terms. RHS or LHS may still have coefficients.
  combine_like — collect like terms together (e.g. `2x + 3x` → `5x`).
  isolate_var  — produce an `after` whose LHS is JUST the variable,
                 with no coefficient. e.g. `4x = 20` → `x = 5`. NOT
                 for `4x - 8 = 12` → `4x = 20` (that's `simplify`).
  evaluate     — reduce a numeric expression. e.g. `x = 9 / 3` → `x = 3`.

The `before` and `after` strings must contain ONLY the equation
itself — no parenthetical comments, no descriptions.

Worked example — pretend the equation is `7a + 5 = 26`:

@algebra_step("7a + 5 = 26", simplify, "7a = 21")
@algebra_step("7a = 21", isolate_var, "a = 3")
@done(3)

Always show your work step by step. Do not skip directly to @done.
"""


def _constrained_user_prompt(problem: Problem) -> str:
    return (
        "Solve for the unknown. End with @done. Output only @-calls.\n\n"
        f"  {problem.expr}\n"
    )


@dataclass
class ConstrainedRun:
    problem: str
    answer: object | None  # int from typed @done(int), or legacy string
    parsed_answer: int | None
    correct: bool
    wall_time_s: float
    n_steps_ok: int
    n_steps_rejected: int
    turn_end_reason: str | None
    trace: list[dict] = field(default_factory=list)


def _parse_done_answer(answer: object, var: str) -> int | None:
    """``done`` now takes a typed integer arg, so the answer comes
    through as an int (or as a tuple/dict containing one). Fall back
    to the old regex path for backwards compat with older traces."""
    if answer is None:
        return None
    if isinstance(answer, int) and not isinstance(answer, bool):
        return answer
    if isinstance(answer, str):
        m = re.search(rf"{re.escape(var)}\s*=\s*(-?\d+)", answer)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        # Bare integer fallback.
        try:
            return int(answer.strip())
        except (ValueError, AttributeError):
            return None
    return None


def run_constrained(
    engine: XGrammarEngine,
    problem: Problem,
    max_turn_tokens: int,
    max_calls_per_turn: int,
) -> ConstrainedRun:
    """Run the algebra_step + done Session against ``problem``."""
    session = Session(
        engine=engine,
        programs={"algebra_step": algebra_step, "done": done},
        system=_CONSTRAINED_SYSTEM,
        max_turn_tokens=max_turn_tokens,
        max_calls_per_turn=max_calls_per_turn,
        allow_free_text=False,
    )
    session.user(_constrained_user_prompt(problem))

    t0 = time.perf_counter()
    answer: str | None = None
    n_ok = 0
    n_rejected = 0
    turn_end_reason: str | None = None
    trace: list[dict] = []

    for event in session.advance():
        if isinstance(event, ProgramInvoked):
            entry = {
                "name": event.name,
                "args": event.args,
                "rejected": event.result.get("rejected", False),
            }
            trace.append(entry)
            if event.result.get("rejected"):
                n_rejected += 1
                continue
            n_ok += 1
            if event.name == "done" and event.args:
                answer = event.args[0] if isinstance(event.args, tuple) else event.args.get(
                    "answer"
                )
        elif isinstance(event, TurnEnded):
            turn_end_reason = event.reason
            break
        elif isinstance(event, NewProgramRegistered | FreeText):
            trace.append({"event": type(event).__name__})

    elapsed = time.perf_counter() - t0
    parsed = _parse_done_answer(answer, problem.var)
    return ConstrainedRun(
        problem=problem.name,
        answer=answer,
        parsed_answer=parsed,
        correct=parsed == problem.expected,
        wall_time_s=elapsed,
        n_steps_ok=n_ok,
        n_steps_rejected=n_rejected,
        turn_end_reason=turn_end_reason,
        trace=trace,
    )


# ---- driver -----------------------------------------------------------


def _pick_model() -> str:
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("no local Qwen2.5 GGUF found")


def _estimate_constrained_tokens(run: ConstrainedRun) -> int:
    """Rough token count for a constrained run, summed over its trace.

    Each ``@-call`` emits the prefix ``@<name>(``, the body grammar's
    args (sum of arg-string lengths), and a closing ``)``. We add a
    ~20-char framing fudge per call. This isn't precise but is a
    reasonable proxy for "how many model tokens did the engine
    actually decode" so we can compute tok/s.
    """
    body_chars = 0
    for t in run.trace:
        if "args" not in t:
            continue
        args = t["args"]
        if isinstance(args, list | tuple):
            body_chars += sum(len(str(a)) for a in args) + 20
        else:
            body_chars += 30
    return max(1, body_chars // 3)


def _format_markdown(
    free_runs: list[FreeTextRun],
    constrained_runs: list[ConstrainedRun],
    problems: list[Problem],
    model: str,
) -> str:
    by_name_p = {p.name: p for p in problems}
    by_name_f = {r.problem: r for r in free_runs}
    by_name_c = {r.problem: r for r in constrained_runs}

    lines: list[str] = []
    lines.append("# Legal-step benchmark — free text vs constrained")
    lines.append("")
    lines.append(f"- **Model:** `{Path(model).name}`")
    lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(
        "- **Decoding:** deterministic argmax under masked logits (XGrammar)."
    )
    lines.append("")
    free_correct = sum(1 for r in free_runs if r.correct)
    cons_correct = sum(1 for r in constrained_runs if r.correct)
    free_time = sum(r.wall_time_s for r in free_runs)
    cons_time = sum(r.wall_time_s for r in constrained_runs)
    cons_rejects = sum(r.n_steps_rejected for r in constrained_runs)
    free_tokens = sum(r.output_tokens for r in free_runs)
    cons_tokens = sum(_estimate_constrained_tokens(r) for r in constrained_runs)
    lines.append(
        f"**Aggregate.** free-text {free_correct}/{len(free_runs)} correct in "
        f"{free_time:.1f}s ({free_tokens / max(free_time, 0.001):.1f} tok/s avg); "
        f"constrained {cons_correct}/{len(constrained_runs)} correct in "
        f"{cons_time:.1f}s ({cons_tokens / max(cons_time, 0.001):.1f} tok/s avg), "
        f"with {cons_rejects} predicate-rejected steps en route."
    )
    lines.append("")
    lines.append("## Per-problem results")
    lines.append("")
    lines.append(
        "| problem | difficulty | expected | free-text | constrained | "
        "free tok/s | cons tok/s | rejects |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )
    for p in problems:
        f = by_name_f.get(p.name)
        c = by_name_c.get(p.name)
        if f is None or c is None:
            continue
        free_cell = (
            f"**{f.parsed_answer}** ✓" if f.correct
            else f"{f.parsed_answer} ✗" if f.parsed_answer is not None
            else "_(no parse)_ ✗"
        )
        cons_cell = (
            f"**{c.parsed_answer}** ✓" if c.correct
            else f"{c.parsed_answer} ✗" if c.parsed_answer is not None
            else f"_(reason: {c.turn_end_reason})_ ✗"
        )
        free_tps = f.output_tokens / max(f.wall_time_s, 0.001)
        cons_toks = _estimate_constrained_tokens(c)
        cons_tps = cons_toks / max(c.wall_time_s, 0.001)
        lines.append(
            f"| `{p.name}` | {p.difficulty} | {p.var} = {p.expected} | "
            f"{free_cell} ({f.wall_time_s:.0f}s) | "
            f"{cons_cell} ({c.wall_time_s:.0f}s) | "
            f"{free_tps:.1f} | {cons_tps:.1f} | {c.n_steps_rejected} |"
        )
        _ = by_name_p
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Free-text parsing.** Free-text outputs are parsed for the "
        "LAST ``ANSWER: <var> = <int>`` line, falling back to the last "
        "bare ``<var> = <int>`` anywhere in the text. Models that don't "
        "follow the requested format produce ``_(no parse)_`` and count "
        "as wrong."
    )
    lines.append(
        "- **Constrained tok/s** is a rough estimate: per-call body "
        "characters + 20-char framing, divided by 3. Predicate "
        "rejections cost tokens too and are included."
    )
    lines.append(
        "- **Cold-start (historical).** Before the engine grew a "
        "compiled-grammar cache, the very first constrained call after "
        "a mode switch paid a ~10× warm-up. The cache (compile by GBNF "
        "source, warm during ``Session.__init__``) eliminated it; "
        "constrained calls now run at the steady-state rate from the "
        "first sample. The 2026-04-25_1132 run shows the old behaviour; "
        "the 1200 run and later show the cache fix."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the first 2 problems (~1 minute total)",
    )
    parser.add_argument(
        "--max-tokens-free", type=int, default=512,
        help="Max tokens for free-text answer (default 512)",
    )
    parser.add_argument(
        "--max-turn-tokens", type=int, default=2048,
        help="Max tokens per constrained turn (default 2048)",
    )
    parser.add_argument(
        "--max-calls", type=int, default=30,
        help="Max @-calls per constrained turn (default 30). Bumped "
             "from 10 because eq_distribute previously hit the wall "
             "after 9 rejections; the headroom lets the model recover.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(_REPO / "bench" / "results"),
        help="Where to write JSON + Markdown results",
    )
    args = parser.parse_args()

    problems = SUITE[:2] if args.quick else SUITE

    model = _pick_model()
    print(f"=== Loading {Path(model).name} ===", flush=True)
    engine = XGrammarEngine(
        model_path=model,
        max_tokens_per_sample=args.max_tokens_free,
        n_ctx=16384,
    )

    print(f"=== Benchmarking {len(problems)} problems × 2 modes ===", flush=True)

    free_runs: list[FreeTextRun] = []
    constrained_runs: list[ConstrainedRun] = []

    for p in problems:
        print(f"  [{p.name}] free-text…", end=" ", flush=True)
        f = run_free_text(engine, p, max_tokens=args.max_tokens_free)
        free_runs.append(f)
        print(f"{f.parsed_answer} ({'✓' if f.correct else '✗'}) {f.wall_time_s:.1f}s")

        print(f"  [{p.name}] constrained…", end=" ", flush=True)
        c = run_constrained(
            engine, p,
            max_turn_tokens=args.max_turn_tokens,
            max_calls_per_turn=args.max_calls,
        )
        constrained_runs.append(c)
        print(
            f"{c.parsed_answer} ({'✓' if c.correct else '✗'}) "
            f"{c.wall_time_s:.1f}s ({c.n_steps_ok} ok, {c.n_steps_rejected} rej)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    json_path = out_dir / f"legal_steps_{stamp}.json"
    md_path = out_dir / f"legal_steps_{stamp}.md"

    json_path.write_text(
        json.dumps(
            {
                "model": model,
                "timestamp": stamp,
                "problems": [asdict(p) for p in problems],
                "free_runs": [asdict(r) for r in free_runs],
                "constrained_runs": [asdict(r) for r in constrained_runs],
            },
            indent=2,
            default=str,
        )
    )
    md_path.write_text(_format_markdown(free_runs, constrained_runs, problems, model))

    print()
    print(f"=== Results: {json_path}")
    print(f"=== Summary: {md_path}")


if __name__ == "__main__":
    main()
