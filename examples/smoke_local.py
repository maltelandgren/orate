"""Smoke demo: run Act 2 and Act 4 against the real local engine.

Uses Qwen2.5-0.5B-Instruct via llama-cpp-python + XGrammar. The
forced-token optimization is on by default; sampling is deterministic
argmax. This is the truest version of the library — local inference,
controlled stack, grammar-mask on every token.

Prereq: a GGUF file at /Users/maltelandgren/models/qwen2.5-*.gguf.
The engine auto-detects the tokenizer from the filename.

Run from repo root:

    .venv/bin/python examples/smoke_local.py
"""

from __future__ import annotations

import time
from pathlib import Path

from orate import gen, program
from orate.engine.xgrammar import XGrammarEngine


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
def prime_digitsum10_hard():
    """Act 2 against a real model: narrow constraint, model-guided proposer."""
    n = yield gen.integer(
        10,
        99,
        where=lambda v: is_prime(v) and digit_sum(v) == 10,
        reject_message=lambda v: f"{v} is not a prime whose digits sum to 10",
        max_retries=40,
    )
    return n


def _pick_model() -> str:
    candidates = [
        "/Users/maltelandgren/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        "No local Qwen2.5 GGUF found. Expected one under /Users/maltelandgren/models/."
    )


def main() -> None:
    model_path = _pick_model()
    print(f"Loading {Path(model_path).name} (first call takes a few seconds)...")
    t0 = time.time()
    engine = XGrammarEngine(model_path=model_path)
    engine.prime(
        "You will answer each question with a single value that satisfies the stated constraint. "
        "Be concise. Use only the format requested.\n\n"
        "Q: Give me a two-digit prime number whose digits sum to 10. "
        "Return only the number.\nA: "
    )
    print(f"  loaded in {time.time() - t0:.1f}s")
    print()

    print("Act 2 (real local model): prime_digitsum10")
    print("-" * 60)
    t0 = time.time()
    result = prime_digitsum10_hard().run(engine=engine)
    dt = time.time() - t0
    rejects = len(engine._context)
    print(f"  answer: {result}")
    print(f"  rejects before accept: {rejects}")
    print(f"  wall time: {dt:.2f}s")
    print()
    # Correctness is guaranteed by the predicate — any non-satisfying
    # value would have raised GrammarExhausted.
    assert is_prime(result) and digit_sum(result) == 10

    print("Phase-B notes injected (first 3):")
    for note in engine._context[:3]:
        print(f"  {note}")


if __name__ == "__main__":
    main()
