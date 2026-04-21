"""Act 2 — Predicates move constraints from types to properties.

Same constraint, same model, different primitive. `where=` turns a
value-level property into a first-class part of the gen spec. On
predicate rejection, the accept set tightens (the offending value is
excluded) and the engine re-samples. ADR-0014: deterministic by
default, no dice rolling to reach correctness.

With `reject_message`, the model *also* gets a natural-language note
about why the last sample failed (Phase B context injection). The
grammar still tightens; the injection adds a steering signal so the
model's argmax moves to a different region of the accept set rather
than the next-lexically-adjacent one.
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
def prime_digitsum10():
    """Act 2: the constraint is a first-class citizen of the gen spec."""
    n = yield gen.integer(
        10,
        99,
        where=lambda v: is_prime(v) and digit_sum(v) == 10,
        reject_message=lambda v: f"{v} failed: prime={is_prime(v)}, digit_sum={digit_sum(v)}",
        max_retries=100,
    )
    return n


def main() -> None:
    print("Act 2 — `where=` moves the constraint from type to property.")
    print()

    # Enumerate the satisfying values in [10, 99] for reference.
    all_ok = [n for n in range(10, 100) if is_prime(n) and digit_sum(n) == 10]
    print(f"Values in [10, 99] satisfying prime ∩ digit-sum-10: {all_ok}")
    print()

    # Solve it 5 times with different seeds; each result is guaranteed valid.
    for seed in range(5):
        engine = MockEngine(seed=seed)
        result = prime_digitsum10().run(engine=engine)
        rejections = len(engine._context)
        print(f"  seed={seed}: result={result}  (Phase-B notes injected: {rejections})")
        assert result in all_ok

    print()
    print("Every output is correct by construction. The predicate is load-bearing.")
    print("Continue: act_03_unified_yield.py")


if __name__ == "__main__":
    main()
