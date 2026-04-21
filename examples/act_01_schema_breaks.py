"""Act 1 — Schemas are a ceiling.

A JSON schema can declare types. It cannot declare "this integer must
be a prime whose digits sum to 10." That is a *value-level* property,
and types are the wrong vocabulary for it.

This script demonstrates the problem. It asks a model (via MockEngine
for the offline runnable version) to produce an integer that's both
prime and has digit-sum 10. With a plain schema, the model is free to
emit any integer — most violate the constraint, and nothing in the
schema catches it. Post-hoc validation is a losing game.

Act 2 fixes it. Act 3 subsumes tool-calling into the same primitive.
Act 4 has the model author its own constraint program at runtime.
"""

from __future__ import annotations

from orate import gen, program
from orate.engine.mock import MockEngine


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


@program
def schema_only_attempt():
    """Act 1: the integer comes out of the model, no predicate. Type: int."""
    n = yield gen.integer(10, 99)
    return n


def main() -> None:
    engine = MockEngine(seed=3)

    # Run 20 samples. Count how many satisfy the post-hoc predicate.
    attempts = [schema_only_attempt().run(engine=engine) for _ in range(20)]
    satisfying = [n for n in attempts if is_prime(n) and digit_sum(n) == 10]

    print("Act 1 — schemas can't express 'prime with digit-sum 10'.")
    print()
    print("20 samples from gen.integer(10, 99) — no predicate attached:")
    print(f"  {attempts}")
    print()
    print(f"  satisfying the constraint: {len(satisfying)}/20")
    if satisfying:
        print(f"  those values: {satisfying}")
    else:
        print("  (none — and nothing in the schema would have told you that.)")
    print()
    print("The schema typed the output; the *value* broke the implicit contract.")
    print("Continue: act_02_predicate_fixes.py")


if __name__ == "__main__":
    main()
