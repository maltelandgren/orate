"""Act-4 finisher: the model authors its own primitive mid-session, then uses it.

This is the climactic beat. The session starts with @algebra_step + @done
pre-registered (single-equation rule-based rewriting). We pose a problem
that doesn't fit the existing primitives well — a quadratic. The model
recognises the gap and emits @make_new_program, authoring a structured
emission that captures the new shape (e.g. factor pair of a quadratic).
The runtime grammar-switches mid-decode, samples the source under
PROGRAM_SOURCE_GRAMMAR, validates, sandbox-execs, registers, rebuilds
the outer grammar — all on the same KV. Then the model uses the
primitive it just defined to finish the solve.

PROGRAM_SOURCE_GRAMMAR now admits ``where=<lib_predicate>(<bound_args>)``
clauses, so the model-authored body can carry a real logical
constraint (e.g. ``where=factors_to(equation)``) and the runtime
verifies it against the host's predicate library. Allowed predicate
names live in ``src/orate/meta_predicates.py``.

What the visual reveals: source materialising on screen with a
``where=`` clause, runtime compile log, then the new primitive being
invoked with grammar-bound args that ALSO satisfy the predicate. The
library grew during the inference. One KV.

Run:
    .venv/bin/python examples/legal_steps/act4_meta_finisher.py
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
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


SYSTEM = """\
You output ONLY @-calls. No prose, no markdown.

Pre-registered tools:
  @algebra_step("before", rule, "after") — one linear-equation
    transformation, where rule ∈ {simplify, combine_like,
    isolate_var, evaluate}. NOTE: these rules cover linear
    equations; they don't have a clean factoring move.
  @done("answer") — terminate the chain.
  @make_new_program("name", "description") — author a NEW @program
    on the fly. The runtime then prompts you for the source body.

Authoring a body — strict rules:
  * Each line is exactly: `<var> = yield gen.<method>(<literal_args>)`.
  * The return MUST be `return {"key1": var1, ...}` where every var
    on the right side is one of the LHS names from the yields above.
    Never invent a new name in the return.
  * Optionally add `where=<predicate>(<args>)` to a yield. Allowed
    predicates: is_prime, digit_sum_eq, lt, gt, equivalent_under,
    factors_to. Args must be names BOUND BY EARLIER yields.

Worked example. User asks for "find a prime less than 100 whose digits
sum to 10". ``@algebra_step`` doesn't fit. You author:

@make_new_program("prime_with_sum", "a prime under 100 whose digits sum to 10")

@program
def prime_with_sum():
    target = yield gen.integer(0, 99)
    n = yield gen.integer(0, 99, where=is_prime())
    return {"target": target, "n": n}

(The ``where=is_prime()`` clause makes the runtime reject any
non-prime emission token-by-token. The grammar is the gate.)

Then invoke:

@prime_with_sum(10, 19)
@done("19")
"""


PROBLEM = """\
Solve x^2 - 5x + 6 = 0 by factoring.

@algebra_step's rules don't include factoring — they're linear-
equation moves. Use @make_new_program to author a primitive that
captures a factor pair, with a where=factors_to(equation) clause
so the runtime verifies the factors actually expand to the target
polynomial. Then use it. End with @done.
"""


def _render(event) -> None:
    if isinstance(event, FreeText):
        text = event.text.strip()
        if text:
            print(f"[text]   {text!r}")
    elif isinstance(event, NewProgramRegistered):
        print(f"[+tool]  {event.name}")
        print(f"--- source ---")
        print(event.source)
        print(f"--------------")
    elif isinstance(event, ProgramInvoked):
        if event.result.get("rejected"):
            print(f"[REJ]    @{event.name}{event.args}")
            print(f"         → {event.result['error']}")
        else:
            print(f"[ok]     @{event.name}{event.args}")
            print(f"         → {event.result}")
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
        max_calls_per_turn=8,
        allow_free_text=False,
    )

    print()
    print("=" * 72)
    print("ACT 4 finisher — model authors its own primitive, then uses it.")
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
