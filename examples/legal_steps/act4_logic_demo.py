"""Act-4 logic demo: Qwen-7B derives a conclusion under predicate-bound
@inference_step calls.

Same shape as :mod:`act4_algebra_demo` but using the propositional
inference rules (modus ponens, hypothetical syllogism, etc.).

Run:
    .venv/bin/python examples/legal_steps/act4_logic_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from legal_steps.logic import inference_step, qed  # noqa: E402
from orate import (  # noqa: E402
    FreeText,
    NewProgramRegistered,
    ProgramInvoked,
    Session,
    TurnEnded,
)
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402


def _pick_model() -> str:
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


SYSTEM = """\
You output ONLY @inference_step calls and one final @qed. No
markdown, no prose. Exact syntax:

@inference_step("premises", rule, "conclusion")

`premises` is a JSON-quoted string of semicolon-separated
propositions (e.g. "P -> Q; P"). `rule` is a bare identifier.
`conclusion` is a JSON-quoted proposition. The runtime verifies
derivability; if invalid, the call is rejected.

Rules:
  modus_ponens             — from "P -> Q; P" derive "Q"
  modus_tollens            — from "P -> Q; ~Q" derive "~P"
  hypothetical_syllogism   — from "P -> Q; Q -> R" derive "P -> R"
  conjunction              — from "P; Q" derive "P & Q"
  simplification           — from "P & Q" derive "P" (or "Q")

Use '->' for implies, '&' for and, '|' for or, '~' for not.

When you've derived the goal, emit @qed("R") to end the turn.

Worked example — given P -> Q, Q -> R, P; prove R:

@inference_step("P -> Q; P", modus_ponens, "Q")
@inference_step("Q -> R; Q", modus_ponens, "R")
@qed("R")
"""


PROBLEM = """\
Use @inference_step calls in the exact syntax from the system prompt.
End with @qed when done.

Given:
  A -> B
  B -> C
  A

Prove C.
"""


def _render(event) -> None:
    if isinstance(event, FreeText):
        text = event.text.strip()
        if text:
            print(f"[text]   {text!r}")
    elif isinstance(event, NewProgramRegistered):
        print(f"[+tool]  {event.name}")
    elif isinstance(event, ProgramInvoked):
        if event.result.get("rejected"):
            print(f"[REJ]    @{event.name}{event.args}")
            print(f"         → {event.result['error']}")
        else:
            print(f"[ok]     @{event.name}{event.args}")
    elif isinstance(event, TurnEnded):
        print(f"[turn end: {event.reason}]")


def main() -> None:
    model = _pick_model()
    print(f"=== Loading {Path(model).name} ===")
    engine = XGrammarEngine(
        model_path=model,
        max_tokens_per_sample=512,
        n_ctx=16384,
    )

    session = Session(
        engine=engine,
        programs={"inference_step": inference_step, "qed": qed},
        system=SYSTEM,
        max_turn_tokens=4096,
        max_calls_per_turn=12,
        allow_free_text=False,  # tool-only — every sample is an @-call
    )

    print()
    print("=" * 72)
    print("ACT 4 — logic. Model authors a chain of legal-deduction calls.")
    print("=" * 72)
    print(PROBLEM)
    print("-" * 72)

    session.user(PROBLEM)
    for event in session.advance():
        _render(event)

    print()
    print("-" * 72)
    print(f"Registry at session end: {sorted(session.registry.keys())}")


if __name__ == "__main__":
    main()
