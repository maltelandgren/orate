"""Act-4 algebra via a composer @program — no Session class.

Demonstrates the Flavor-B-full unification: the agent loop is *inside*
a ``@program(invocable=False)``. The leaves (``@algebra_step``,
``@done``) are dispatched via ``gen.alternative([...])`` at every
yield. There's no Session, no registry, no _build_outer_grammar —
just a composer running on a persistent-KV engine.

Compare with ``act4_algebra_demo.py`` (Session-based). Same model,
same problem, same predicates — different wiring.

Run:
    .venv/bin/python examples/legal_steps/act4_algebra_composer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from legal_steps.algebra import algebra_step, done  # noqa: E402
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
  @algebra_step("before", rule, "after")
  @done("answer")

The runtime mathematically verifies that `after` equals `before`
under `rule`. If not, the call is rejected.

Rules: simplify, combine_like, isolate_var, evaluate.

Worked example — solve 2x + 1 = 7:

@algebra_step("2x + 1 = 7", simplify, "2x = 6")
@algebra_step("2x = 6", isolate_var, "x = 3")
@done("x = 3")
"""


PROBLEM = """\
Solve for x. End with @done.

  3x + 5 = 14
"""


@program(invocable=False)
def solve():
    """Composer: yield gen.alternative until the model emits @done.

    Each iteration the model picks one of the two leaves. The
    composer reads the result and decides whether to terminate. No
    explicit grammar, no session — the loop is the agent.

    The trace list captures every action; the host inspects it after
    the run for display. This is plain Python state inside the
    composer's scope.
    """
    trace: list[dict] = []
    step = 0
    while True:
        action = yield gen.alternative([algebra_step, done])
        step += 1
        trace.append({"step": step, "name": action.name, "args": action.args})
        print(f"[step {step}]  @{action.name}{action.args}")
        if action.name == "done":
            return {"answer": action.value, "trace": trace}


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
    print("ACT 4 — algebra via COMPOSER @program (no Session class).")
    print("=" * 72)
    print(PROBLEM)
    print("-" * 72)

    # The "agent run" is one line.
    result = solve().run(engine=engine)
    print()
    print("-" * 72)
    print(f"=== Final answer: {result['answer']}")
    print(f"=== Steps taken: {len(result['trace'])}")


if __name__ == "__main__":
    main()
