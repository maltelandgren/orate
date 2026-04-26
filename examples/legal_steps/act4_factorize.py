"""Act-4 factorize: model authors a program that verifies its own answer.

The pitch: hand the model a problem it routinely fumbles in free text
(non-trivial integer factorization), tell it about a predicate library
(``is_prime``, ``divides``, ``multiplies_to``, ...) and the
``@make_new_program`` mechanism, and watch it author a tiny ``@program``
whose ``where=`` clauses *guarantee* the answer is correct before the
runtime returns it.

The pre-registered surface is intentionally minimal — only ``@done`` and
``@make_new_program`` — so there's no off-the-shelf factoring tool for
the model to fall back on. To answer the question it has to author the
verifier itself, then invoke it.

Run:
    .venv/bin/python examples/legal_steps/act4_factorize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from legal_steps.algebra import done  # noqa: E402  — only @done is pre-registered
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


# The system prompt is the contract: tells the model exactly what's
# pre-registered, what the meta-authoring loop looks like, and what
# predicates it can reference inside an authored body.
SYSTEM = """\
You output ONLY @-calls. No prose, no markdown.

Pre-registered tools:
  @done("answer")               — end the chain.
  @make_new_program("name", "short description")
      Author a NEW @program at runtime. KEEP the description SHORT
      (one phrase, under 50 chars). After you emit this, you write
      the @program body directly. Once it validates, the new program
      is registered and you can invoke it as @<name>(...).

Authoring rules (grammar-enforced; bad tokens are masked out):

  * Header is exactly:
        @program
        def <name>():
  * Body lines are: `<var> = yield gen.<method>(<literal_args>)`.
  * Allowed methods:
        gen.integer(lo, hi)         — two int literals
        gen.choice(["a", "b", ...]) — string-literal options
        gen.string(max_len=N)
        gen.boolean()
  * Each yield can OPTIONALLY end with `, where=<predicate>(<bound_args>)`.
  * IMPORTANT: <bound_args> are the LHS variable names from yields
    ABOVE. They are NEVER the function's own name. Reference only
    names you have written on the left side of an earlier `=`.
  * Keep bodies SHORT — one yield per piece of data. Do not pad
    with extra `gen.boolean()` lines.
  * Return is `return {"key": var, ...}` where every value is a
    bound name from the LHS of a yield above.

Predicate library you may use in `where=` clauses (all curried — call
with the relevant bound names; the runtime supplies the candidate):

  is_prime()                  — candidate is prime
  digit_sum_eq(target)        — digits of candidate sum to target
  lt(bound)                   — candidate < bound
  gt(bound)                   — candidate > bound
  multiplies_to(target, other)— candidate * other == target
  sums_to(target, other)      — candidate + other == target
  divides(target)             — candidate is a non-zero divisor of target
  divisible_by(divisor)       — divisor evenly divides candidate
  is_square()                 — candidate is a perfect square
  is_palindrome()             — str(candidate) reads the same backwards
  coprime_with(other)         — gcd(candidate, other) == 1
  length_eq(target)           — len(str(candidate)) == target

Each yield supports ONE where= clause. Chain yields to express
multiple constraints — earlier values become bound names for the
predicates on later yields.

================================================================
Worked example. The user asks:
    "Factorize 35 into two factors a and b both > 1."

You recognise there's no factoring tool, so you author one. Note
how `n` is bound on line 1 and then referenced BY NAME inside the
predicates on lines 2 and 3 — and how `a` (bound on line 2) is
referenced inside the predicate on line 3. Use SHORT, SINGLE-LETTER
variable names — they survive tokenization cleanly.

@make_new_program("factor_35", "two factors of 35")

@program
def factor_35():
    n = yield gen.integer(35, 35)
    a = yield gen.integer(2, 34, where=divides(n))
    b = yield gen.integer(2, 34, where=multiplies_to(n, a))
    return {"a": a, "b": b}

Now invoke it. The args MUST satisfy every where= clause; if not,
the runtime rejects and you re-sample.

@factor_35(35, 5, 7)
@done("5 and 7")

Why this works:
  - n=35 is a fixed literal yield so later predicates can name it.
  - a must divide 35: legal values in [2, 34] are {5, 7}.
  - b must satisfy b * a == 35: given a=5, b is forced to 7.
  - The grammar guarantees BOTH yields satisfy their predicates
    before any value reaches you.

CRITICAL: in the return dict, the VALUES on the right side of `:`
must be the variable names you bound on each yield's LHS — `a` and
`b` in this example, NOT `n` (which is just the target literal).
================================================================
"""


PROBLEM = """\
Factorize 1147 into two factors p and q, both greater than 1.

You don't have a factoring tool. Use @make_new_program to author a tiny
@program whose `where=` predicates guarantee that:
  - p is a divisor of 1147 (use `divides`)
  - q satisfies p * q == 1147 (use `multiplies_to`)
Then invoke it. End with @done("p and q").
"""


def _render(event) -> None:
    if isinstance(event, FreeText):
        text = event.text.strip()
        if text:
            print(f"[text]   {text!r}")
    elif isinstance(event, NewProgramRegistered):
        print(f"[+tool]  {event.name}")
        print("--- source ---")
        print(event.source)
        print("--------------")
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
        programs={"done": done},
        system=SYSTEM,
        max_turn_tokens=4096,
        max_calls_per_turn=8,
        allow_free_text=False,
    )

    print()
    print("=" * 72)
    print("ACT 4 factorize — model authors a verifier, then uses it.")
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
