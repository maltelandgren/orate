"""Orate-augmented Qwen-7B run on BBH logical_deduction.

Pipeline per problem (Path C — domain-specific predicate-bound primitives):

  1. Deterministically extract ``items`` and ``options`` from the BBH
     problem text (regex parser; see :mod:`bench.bbh.extractor`).
  2. Build a fresh :class:`bench.bbh.programs.Knowledge` instance and
     ``set_active`` it. The @-call predicates close over this instance.
  3. Run an orate :class:`Session` with three programs registered:
     ``@premise`` / ``@deduce`` / ``@answer``. The system prompt teaches
     the model the call shape; few-shot exemplars show one worked
     extraction → derivation → answer chain.
  4. Drive the Session to a TurnEnded event. The ``@answer`` call's
     ``letter`` is the final prediction. We grade against the BBH
     target.
  5. Capture the trace: every ProgramInvoked event (including rejected
     ones), wall-time, fact-chain at termination.

Saves a JSON trace per subtask: ``bench/results/bbh_orate_<subtask>_<stamp>.json``
shaped identically to the baseline traces (so the writeup can join on
problem index).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bench"))

from bbh.extractor import extract_problem  # noqa: E402
from bbh.loader import SUBTASKS, BBHProblem, load_subtask  # noqa: E402
from bbh.programs import (  # noqa: E402
    Knowledge,
    answer,
    deduce,
    premise,
    record_invocation,
    set_active,
)

from orate import (  # noqa: E402
    FreeText,
    NewProgramRegistered,
    ProgramInvoked,
    Session,
    TurnEnded,
)
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402


SYSTEM_ANSWER_ONLY = """\
You solve ordering puzzles by emitting ONE @answer(letter) call.
No prose, no markdown, no commentary. The puzzle's constraints have
been extracted into a KNOWN FACTS list. Pick the option that follows.

  @answer(letter)

The runtime VERIFIES the choice: the option is rejected if it isn't
forced by the known facts. Only one option is forced. Pick it.

Worked example — three birds. KNOWN FACTS:
right_of(falcon, blue jay); right_of(blue jay, quail). Question:
which is second from the left? Options: (A) blue jay (B) quail
(C) falcon.

@answer(A)
"""


SYSTEM = """\
You solve ordering puzzles by emitting ONLY @-calls. No prose, no
markdown, no commentary. The puzzle's constraints have ALREADY been
extracted into a fact list shown below; you reason FROM that list to
the answer. Two calls available:

  @deduce(predicate, "args")
  @answer(letter)

Predicates — all positions 1-based, left-to-right. For age/price:
oldest=position 1, newest=position N; cheapest=position 1, most
expensive=position N.

  left_of            args = "a, b"           → a is to the left of b
  right_of           args = "a, b"           → a is to the right of b
  leftmost           args = "a"              → a is the leftmost
  rightmost          args = "a"              → a is the rightmost
  position           args = "a, k"           → a is at position k
  between            args = "a, b, c"        → a is between b and c
  immediately_left_of   args = "a, b"        → a is immediately left of b
  immediately_right_of  args = "a, b"        → a is immediately right of b

The runtime VERIFIES every @-call. A @deduce is rejected if its
fact isn't FORCED by the known facts (i.e. true under every valid
arrangement). An @answer is rejected if the chosen option isn't
forced by the known facts. Rejected calls are noted; retry with
the corrected call.

Strategy:
  1. The known facts plus your @deduce-derived facts form your
     working set.
  2. Combine known facts with @deduce calls until the answer is
     forced. Use the canonical item names from the prompt
     (lowercase, no articles).
  3. Emit @answer(letter) for the entailed option.

Worked example — three birds (blue jay, quail, falcon). KNOWN FACTS:
right_of(falcon, blue jay); right_of(blue jay, quail). Question:
which is second from the left? Options: (A) blue jay (B) quail
(C) falcon.

@deduce(position, "blue jay, 2")
@answer(A)
"""


def _user_prompt(
    problem: BBHProblem,
    items: list[str],
    options: dict[str, str],
    *,
    known_facts: list | None = None,
) -> str:
    """Render the per-problem instruction block.

    We surface the parsed items + options as machine-readable lists so
    the model uses the canonical lowercased item names — that's what
    the predicate parser expects (see ``ordering._norm``).

    When ``known_facts`` is supplied (Path A; the default for the
    headline run), they're rendered as a `KNOWN FACTS:` block so the
    model treats them as ground truth and only needs to deduce the
    answer.
    """
    items_line = ", ".join(items) if items else "(unparsed)"
    options_lines = "\n".join(f"  {k} {v}" for k, v in options.items())
    facts_block = ""
    if known_facts:
        facts_block = (
            "KNOWN FACTS (extracted from the puzzle, treat as ground truth):\n"
            + "\n".join(f"  {f.render()}" for f in known_facts)
            + "\n\n"
        )
    return (
        f"{problem.question}\n\n"
        f"Items (use exactly these names): {items_line}\n\n"
        f"{facts_block}"
        f"Options:\n{options_lines}\n\n"
        f"Emit @-calls only. End with @answer(letter)."
    )


@dataclass
class OrateRun:
    subtask: str
    index: int
    target: str
    extracted: str | None  # the @answer letter ("(A)" form), or None
    correct: bool
    wall_time_s: float
    n_premise_ok: int
    n_premise_rejected: int
    n_deduce_ok: int
    n_deduce_rejected: int
    n_answer_attempts: int
    turn_end_reason: str | None
    items: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)


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
    max_turn_tokens: int,
    max_calls: int,
    preload_premises: bool = True,
    answer_only: bool = False,
) -> OrateRun:
    """Run one BBH problem under the orate constrained chain.

    When ``preload_premises=True`` (the default and the path the
    headline run uses), we deterministically extract the puzzle
    constraints via :mod:`bench.bbh.constraint_parser` and pre-populate
    the :class:`Knowledge` instance. This isolates the contribution of
    orate's @-call predicates to the *deduction* + *answer* phases —
    the model doesn't have to translate English constraints, it just
    has to combine the pre-loaded facts and pick the entailed option.

    With ``preload_premises=False`` we take the harder path: the model
    must author @premise calls itself. We tried this first; Qwen-7B's
    English-to-fact translation isn't reliable enough at this scale
    (e.g. it confuses "X to the right of Y" with ``left_of(x, y)``),
    which propagates through the deduction chain and tanks accuracy
    even when the predicates are correct.
    """
    from bbh.constraint_parser import extract_premises  # noqa: PLC0415
    from bbh.loader import SUBTASK_OBJECT_COUNT  # noqa: PLC0415

    ex = extract_problem(problem.question)
    knowledge = Knowledge(items=list(ex.items), options=dict(ex.options))
    if preload_premises:
        n_items = SUBTASK_OBJECT_COUNT[problem.subtask]
        knowledge.premises = list(extract_premises(problem.question, n_items))
    set_active(knowledge)

    # Path A-min: only @answer registered (premises pre-loaded; no
    # intermediate deductions allowed). The predicate forces the
    # unique correct letter — the model can't escape via reasoning,
    # right or wrong.
    # Path A:    @deduce + @answer (model derives intermediate facts).
    # Path C:    @premise + @deduce + @answer (model authors premises too).
    if answer_only:
        programs = {"answer": answer}
    elif preload_premises:
        programs = {"deduce": deduce, "answer": answer}
    else:
        programs = {"premise": premise, "deduce": deduce, "answer": answer}

    session = Session(
        engine=engine,
        programs=programs,
        system=SYSTEM_ANSWER_ONLY if answer_only else SYSTEM,
        max_turn_tokens=max_turn_tokens,
        max_calls_per_turn=max_calls,
        allow_free_text=False,
    )
    session.user(_user_prompt(
        problem, ex.items, ex.options,
        known_facts=knowledge.premises if preload_premises else None,
    ))

    t0 = time.perf_counter()
    trace: list[dict] = []
    n_premise_ok = 0
    n_premise_rejected = 0
    n_deduce_ok = 0
    n_deduce_rejected = 0
    n_answer_attempts = 0
    answer_letter: str | None = None
    turn_end_reason: str | None = None

    for event in session.advance():
        if isinstance(event, ProgramInvoked):
            rejected = bool(event.result and event.result.get("rejected"))
            entry = {
                "name": event.name,
                "args": event.args
                if isinstance(event.args, dict)
                else list(event.args)
                if isinstance(event.args, tuple)
                else event.args,
                "rejected": rejected,
                "error": event.result.get("error") if rejected else None,
            }
            trace.append(entry)
            if event.name == "premise":
                if rejected:
                    n_premise_rejected += 1
                else:
                    n_premise_ok += 1
                    record_invocation(event.name, event.args)
            elif event.name == "deduce":
                if rejected:
                    n_deduce_rejected += 1
                else:
                    n_deduce_ok += 1
                    record_invocation(event.name, event.args)
            elif event.name == "answer":
                n_answer_attempts += 1
                if not rejected:
                    args = event.args
                    if isinstance(args, dict):
                        answer_letter = args.get("letter")
                    elif isinstance(args, tuple) and args:
                        answer_letter = args[0]
        elif isinstance(event, TurnEnded):
            turn_end_reason = event.reason
            break
        elif isinstance(event, NewProgramRegistered | FreeText):
            trace.append({"event": type(event).__name__})

    elapsed = time.perf_counter() - t0
    set_active(None)

    extracted = f"({answer_letter})" if answer_letter else None
    correct = extracted is not None and extracted == problem.target.upper()
    return OrateRun(
        subtask=problem.subtask,
        index=problem.index,
        target=problem.target,
        extracted=extracted,
        correct=correct,
        wall_time_s=elapsed,
        n_premise_ok=n_premise_ok,
        n_premise_rejected=n_premise_rejected,
        n_deduce_ok=n_deduce_ok,
        n_deduce_rejected=n_deduce_rejected,
        n_answer_attempts=n_answer_attempts,
        turn_end_reason=turn_end_reason,
        items=list(ex.items),
        options=dict(ex.options),
        trace=trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subtask",
        choices=list(SUBTASKS) + ["all"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turn-tokens", type=int, default=1024)
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--out-dir", type=str, default=str(_REPO / "bench" / "results"))
    parser.add_argument("--stamp", type=str, default=None)
    args = parser.parse_args()

    subtasks = list(SUBTASKS) if args.subtask == "all" else [args.subtask]

    model = _pick_model()
    print(f"=== Loading {Path(model).name} ===", flush=True)
    engine = XGrammarEngine(
        model_path=model,
        max_tokens_per_sample=args.max_turn_tokens,
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
        runs: list[OrateRun] = []
        n_correct = 0
        n_unanswered = 0
        t0 = time.perf_counter()
        for i, p in enumerate(problems):
            r = run_problem(
                engine, p,
                max_turn_tokens=args.max_turn_tokens,
                max_calls=args.max_calls,
            )
            runs.append(r)
            if r.correct:
                n_correct += 1
            if r.extracted is None:
                n_unanswered += 1
            mark = "✓" if r.correct else "✗"
            print(
                f"  [{i + 1:>3}/{len(problems)}] {mark} target={p.target} "
                f"got={r.extracted} ({r.wall_time_s:.1f}s, "
                f"prem={r.n_premise_ok}/{r.n_premise_ok + r.n_premise_rejected}, "
                f"ded={r.n_deduce_ok}/{r.n_deduce_ok + r.n_deduce_rejected}, "
                f"end={r.turn_end_reason})",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        accuracy = n_correct / max(1, len(problems))
        print(
            f"  -> {n_correct}/{len(problems)} = {100 * accuracy:.1f}%  "
            f"(unanswered: {n_unanswered}; {elapsed:.0f}s total)",
            flush=True,
        )

        out_path = out_dir / f"bbh_orate_{subtask}_{stamp}.json"
        out_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "subtask": subtask,
                    "stamp": stamp,
                    "n_problems": len(problems),
                    "n_correct": n_correct,
                    "n_unanswered": n_unanswered,
                    "accuracy": accuracy,
                    "wall_time_s": elapsed,
                    "max_turn_tokens": args.max_turn_tokens,
                    "max_calls": args.max_calls,
                    "runs": [asdict(r) for r in runs],
                },
                indent=2,
            )
        )
        print(f"  -> wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
