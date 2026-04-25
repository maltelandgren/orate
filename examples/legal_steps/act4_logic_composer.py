"""Act-4 logic via a composer @program — no Session class.

Same shape as ``act4_algebra_composer.py`` but for propositional logic.
The agent loop is *inside* a ``@program(invocable=False)``. The leaves
(``@inference_step``, ``@qed``) are dispatched via ``gen.alternative([...])``
at every yield. There's no Session, no registry — just a composer
running on a persistent-KV engine.

Each iteration:
  1. The model picks a leaf via ``gen.alternative``.
  2. The leaf's body grammar binds the args.
  3. ``@inference_step``'s third yield's ``where=`` runs SymPy /
     ``derivable_under`` on the model's emission. If invalid, the
     call is rejected and the model re-decodes.
  4. The composer's Python loop reads the result and decides whether
     to terminate.

Run:
    .venv/bin/python examples/legal_steps/act4_logic_composer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from legal_steps.logic import inference_step, qed  # noqa: E402
from orate import gen, program  # noqa: E402
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402


def _pick_model() -> str:
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


SYSTEM = """\
You output ONLY @-calls. No markdown, no prose, no commentary.

Available calls:
  @inference_step("premises", rule, "conclusion")
  @qed("final_conclusion")

`premises` is JSON-quoted, semicolon-separated. `conclusion` is
JSON-quoted. `rule` is a bare identifier.

The runtime verifies that `conclusion` is derivable from `premises`
under `rule`. If invalid, the call is rejected.

Rules:
  modus_ponens             — from "P -> Q; P" derive "Q"
  modus_tollens            — from "P -> Q; ~Q" derive "~P"
  hypothetical_syllogism   — from "P -> Q; Q -> R" derive "P -> R"
  conjunction              — from "P; Q" derive "P & Q"
  simplification           — from "P & Q" derive "P" (or "Q")

Use '->' for implies, '&' for and, '|' for or, '~' for not.

Worked example — given P -> Q, Q -> R, P; prove R:

@inference_step("P -> Q; P", modus_ponens, "Q")
@inference_step("Q -> R; Q", modus_ponens, "R")
@qed("R")
"""


PROBLEM = """\
Use @inference_step then @qed. End with @qed.

Given:
  A -> B
  B -> C
  A
Prove: C
"""


@program(invocable=False)
def derive():
    """Composer: yield gen.alternative until the model emits @qed.

    Each iteration the model picks one of the two leaves. The
    runtime predicate-checks (via ``derivable_under``) before the
    yield resolves; the composer's Python loop just reads the
    result and decides whether to terminate.
    """
    trace: list[dict] = []
    step = 0
    while True:
        action = yield gen.alternative([inference_step, qed])
        step += 1
        trace.append({"step": step, "name": action.name, "args": action.args})
        print(f"[step {step}]  @{action.name}{action.args}")
        if action.name == "qed":
            return {"final": action.value, "trace": trace}


def main() -> None:
    model = _pick_model()
    print(f"=== Loading {Path(model).name} ===")
    engine = XGrammarEngine(
        model_path=model,
        max_tokens_per_sample=512,
        n_ctx=16384,
    )
    engine.begin_session(SYSTEM)
    engine.append(f"\n<|user|>\n{PROBLEM}\n<|assistant|>\n")

    print()
    print("=" * 72)
    print("ACT 4 — logic via COMPOSER @program (no Session class).")
    print("=" * 72)
    print(PROBLEM)
    print("-" * 72)

    # The agent run is one line.
    result = derive().run(engine=engine)
    print()
    print("-" * 72)
    print(f"=== Final: {result['final']}")
    print(f"=== Steps taken: {len(result['trace'])}")


if __name__ == "__main__":
    main()
