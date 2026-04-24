"""Smoke the whole new stack against the real local model.

Exercises:
  * Layer 1 — witness enumeration (prime-digit-sum: zero rejects)
  * Layer 3 — forward-checking on a struct (x+y==10)
  * Layer 4 — source-in-prompt + Gen descriptions
  * Verifiers — first-class @verifier decorator

Run: .venv/bin/python examples/smoke_layer_1234.py
"""

from __future__ import annotations

from pathlib import Path

from orate import (
    Accept,
    Reject,
    build_prompt,
    gen,
    program,
    verifier,
)
from orate.engine.xgrammar import XGrammarEngine


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))


def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


@verifier
def is_under_100(n):
    if n < 100:
        return Accept()
    return Reject(f"value {n} should be below 100")


@program
def prime_with_digit_sum_10():
    n = yield gen.integer(
        10,
        99,
        where=lambda v: is_prime(v) and digit_sum(v) == 10,
        description="a two-digit prime whose digits sum to 10",
        reject_message=lambda v: f"{v} failed the constraint",
    )
    yield is_under_100(n)
    return n


@program
def sum_to_ten():
    pair = yield gen.struct(
        x=gen.integer(0, 10, description="first addend"),
        y=gen.integer(0, 10, description="second addend"),
        where=lambda d: d["x"] + d["y"] == 10,
        description="two non-negative integers that sum to 10",
    )
    return pair


def _pick_model() -> str:
    for c in [
        "/Users/maltelandgren/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(c).exists():
            return c
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


def main() -> None:
    model_path = _pick_model()
    print(f"Loading {Path(model_path).name}...")
    engine = XGrammarEngine(model_path=model_path)

    # --- Layer 1 demo -----------------------------------------------------
    print()
    print("─" * 72)
    print("LAYER 1 — witness enumeration on prime-with-digit-sum-10")
    print("─" * 72)

    prompt = build_prompt(
        prime_with_digit_sum_10,
        user_prompt="Respond with the number only.",
        show_source=True,
    )
    engine.prime(prompt)
    print("First 10 lines of the prompt the model actually sees:")
    for line in prompt.split("\n")[:10]:
        print(f"  | {line}")
    print("  | ...")

    result = prime_with_digit_sum_10().run(engine=engine)
    print()
    print(f"Model returned: {result}")
    print(f"Phase-B notes injected: {len(engine._context)}  (zero = Layer 1 worked)")
    assert result in {19, 37, 73}

    # --- Layer 3 demo -----------------------------------------------------
    print()
    print("─" * 72)
    print("LAYER 3 — forward-checking on struct(x+y==10)")
    print("─" * 72)

    engine2 = XGrammarEngine(model_path=model_path)
    prompt2 = build_prompt(
        sum_to_ten,
        user_prompt="Return two integers that sum to 10.",
        show_source=True,
    )
    engine2.prime(prompt2)

    pair = sum_to_ten().run(engine=engine2)
    print()
    print(f"Model returned: x={pair['x']}, y={pair['y']}  (sum = {pair['x'] + pair['y']})")
    assert pair["x"] + pair["y"] == 10
    print(f"Phase-B notes injected: {len(engine2._context)}  (zero = Layer 3 worked)")

    # --- source-in-prompt preview -----------------------------------------
    print()
    print("─" * 72)
    print("Layer 4 — source-in-prompt preview (what the model sees)")
    print("─" * 72)
    print()
    print(prompt2)


if __name__ == "__main__":
    main()
