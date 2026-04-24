"""Act-4 algebra demo: Qwen-7B solves a constraint problem under
predicate-bound @algebra_step calls.

The session is seeded with @algebra_step pre-registered. Every call
the model emits gets predicate-verified at decode time: if the
sampled (rule, after) pair isn't algebraically equivalent to before
under rule, the call is rejected with a session note and the model
re-decodes.

Run:
    .venv/bin/python examples/legal_steps/act4_algebra_demo.py

The script picks the largest local Qwen2.5 GGUF it can find. Bring
your own GGUFs (or symlink them into ``~/models``) and the path
resolution in :func:`_pick_model` will find them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from legal_steps.algebra import algebra_step, done  # noqa: E402
from orate import (  # noqa: E402
    FreeText,
    NewProgramRegistered,
    ProgramInvoked,
    Session,
    TurnEnded,
)
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402


def _pick_model() -> str:
    """Prefer the biggest Qwen we have locally."""
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


SYSTEM = """\
You output ONLY @algebra_step calls. No markdown, no prose, no
commentary. Exact syntax:

@algebra_step("before", rule, "after")

`before` and `after` are JSON-quoted equation strings. `rule` is a
bare identifier. The runtime verifies algebraic equivalence —
including scalar multiplication, so 2x = 8 ↔ x = 4 is legal.

Rules:
  simplify     — distribute, combine numeric terms
  combine_like — collect terms with the same variable
  isolate_var  — produce an equation whose LHS is a single variable
  evaluate     — produce an equation with a pure number on one side

CRITICAL — applying a known fact:
  When you reuse a fact from a previous step (like "x = 5 - y"),
  bake the substitution INTO `before` directly. The runtime is
  stateless — it does not remember prior facts.

When the unknown is solved, emit @done("x = N") to end the turn.

Worked example — solve 2x + 1 = 7:

@algebra_step("2x + 1 = 7", simplify, "2x = 6")
@algebra_step("2x = 6", isolate_var, "x = 3")
@done("x = 3")
"""


PROBLEM = """\
Solve for x. Use @algebra_step calls in the exact syntax from the
system prompt. Output nothing else.

Equation:
  3x + 5 = 14
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
        programs={"algebra_step": algebra_step, "done": done},
        system=SYSTEM,
        max_turn_tokens=4096,
        max_calls_per_turn=12,
        allow_free_text=False,  # tool-only — every sample is an @-call
    )

    print()
    print("=" * 72)
    print("ACT 4 — algebra. Model authors a chain of legal-step calls.")
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
