"""Single-problem orate debug harness — verbose tracing.

Runs ONE BBH problem under the orate session and prints every
sample_under output, every parsed @-call, every predicate result.
Used to diagnose why deductions are getting rejected en masse.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bench"))

from bbh.constraint_parser import extract_premises  # noqa: E402
from bbh.extractor import extract_problem  # noqa: E402
from bbh.loader import SUBTASK_OBJECT_COUNT, load_subtask  # noqa: E402
from bbh.programs import (  # noqa: E402
    Knowledge, answer, deduce, premise, record_invocation, set_active,
)
from bbh.run_orate import SYSTEM, _user_prompt  # noqa: E402

from orate import (  # noqa: E402
    FreeText,
    NewProgramRegistered,
    ProgramInvoked,
    Session,
    TurnEnded,
)
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402


def _pick_model() -> str:
    return "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"


def main() -> None:
    import os
    idx = int(os.environ.get("DEBUG_IDX", "0"))
    rows = load_subtask("logical_deduction_three_objects", limit=idx + 1)
    p = rows[idx]
    ex = extract_problem(p.question)
    n_items = SUBTASK_OBJECT_COUNT[p.subtask]
    pre_premises = extract_premises(p.question, n_items)
    knowledge = Knowledge(items=list(ex.items), options=dict(ex.options))
    knowledge.premises = list(pre_premises)
    set_active(knowledge)

    print("=== Problem ===")
    print(p.question)
    print()
    print("Items:", ex.items)
    print("Options:", ex.options)
    print("Target:", p.target)
    print(f"Pre-extracted premises: {[f.render() for f in pre_premises]}")
    print()

    print("=== Loading engine ===")
    engine = XGrammarEngine(
        model_path=_pick_model(),
        max_tokens_per_sample=1024,
        n_ctx=8192,
    )
    engine.load()
    engine.warm()

    session = Session(
        engine=engine,
        programs={"deduce": deduce, "answer": answer},
        system=SYSTEM,
        max_turn_tokens=1024,
        max_calls_per_turn=20,
        allow_free_text=False,
    )
    session.user(_user_prompt(p, ex.items, ex.options, known_facts=pre_premises))

    print("=== Session run ===")
    for ev in session.advance():
        if isinstance(ev, ProgramInvoked):
            rejected = bool(ev.result and ev.result.get("rejected"))
            mark = "REJ" if rejected else "OK "
            args = ev.args if isinstance(ev.args, dict | tuple | list) else ev.args
            print(f"[{mark}] {ev.name}({args})")
            if rejected:
                print(f"      error: {ev.result.get('error')}")
            if not rejected:
                record_invocation(ev.name, ev.args)
            # Also dump current Knowledge state
            print(f"      premises now: {[f.render() for f in knowledge.premises]}")
            print(f"      deductions now: {[f.render() for f in knowledge.deductions]}")
        elif isinstance(ev, TurnEnded):
            print(f"[end] {ev.reason}")
            break
        elif isinstance(ev, FreeText):
            print(f"[txt] {ev.text!r}")
        elif isinstance(ev, NewProgramRegistered):
            print(f"[new] {ev.name}")


if __name__ == "__main__":
    main()
