"""Real-model smoke: the Act-4 self-referential loop end-to-end.

Phase 1: Qwen2.5 writes its own ``@program`` source (grammar-constrained).
Phase 2: The same engine runs the compiled @program. Its argmax is now
         shaped by the grammar it itself authored one second earlier.

If you only read one example in this repo, read this one. The loop is
cheap to describe and weird to watch work.

    .venv/bin/python examples/smoke_meta.py
"""

from __future__ import annotations

import time
from pathlib import Path

from orate import meta_solve, synthesize_program
from orate.engine.xgrammar import XGrammarEngine


def _pick_model() -> str:
    # Prefer the 7B for synthesis — small models under argmax tend to get
    # stuck in recursive-rule loops (the infinite-assign-stmt failure mode).
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


TASK = (
    "Pick a favorite color (from red / blue / green / yellow), "
    "pick a lucky number between 1 and 20, and answer whether "
    "you prefer day over night."
)


def _rule(char: str = "─", width: int = 72) -> str:
    return char * width


def main() -> None:
    model_path = _pick_model()
    print(f"Loading {Path(model_path).name}...")
    engine = XGrammarEngine(model_path=model_path, max_tokens_per_sample=1024)

    # --- Phase 1: synthesize the program ----------------------------------
    print()
    print(_rule("═"))
    print("PHASE 1 — the model writes its own @program")
    print(_rule("═"))
    print()
    print(f"Task: {TASK}")

    t0 = time.time()
    try:
        compiled_fn, source, trace = synthesize_program(
            engine,
            task=TASK,
            max_retries=3,
            max_tokens=2048,
        )
    except Exception as e:
        print()
        print(f"Synthesis failed: {e}")
        trace = getattr(e, "trace", []) or []
        for entry in trace:
            print(f"  [attempt {entry['attempt']}] status={entry['status']}")
            src = entry.get("source", "")
            if src:
                print("    Source (first 500 chars):")
                for line in src[:500].splitlines():
                    print(f"      | {line}")
                if len(src) > 500:
                    print(f"      | ... ({len(src) - 500} more bytes)")
            errs = entry.get("errors") or []
            for err in errs[:2]:
                print(f"    Error: {err[:200]}")
        raise
    dt1 = time.time() - t0

    print()
    print(f"Synthesis attempts: {len(trace)}  (wall: {dt1:.1f}s)")
    print()
    print("AUTHORED @program source:")
    print(_rule())
    print(source, end="")
    print(_rule())

    for entry in trace:
        status = entry["status"]
        errors = entry.get("errors") or []
        if status == "accepted":
            print(f"  [attempt {entry['attempt']}] accepted")
        else:
            err = errors[0] if errors else "unknown"
            print(f"  [attempt {entry['attempt']}] rejected — {err[:80]}")

    # --- Phase 2: the same engine runs the compiled program ---------------
    print()
    print(_rule("═"))
    print("PHASE 2 — the same engine runs the compiled program")
    print(_rule("═"))

    t0 = time.time()
    invocation = compiled_fn()
    result = invocation.run(engine=engine)
    dt2 = time.time() - t0

    print()
    print(f"Phase-2 wall: {dt2:.2f}s")
    print(f"Result: {result}")

    # --- meta_solve does both in one call ---------------------------------
    print()
    print(_rule("═"))
    print("SHORTHAND — meta_solve() does phases 1+2 in one call")
    print(_rule("═"))
    engine2 = XGrammarEngine(model_path=model_path, max_tokens_per_sample=1024)
    t0 = time.time()
    outcome = meta_solve(
        engine2,
        task="Draft a short hero name and choose a class from knight, mage, or ranger.",
        max_retries=3,
    )
    dt3 = time.time() - t0
    print()
    print(f"Wall: {dt3:.1f}s ({outcome.synthesis_attempts} synthesis attempt(s))")
    print()
    print("Authored source:")
    print(_rule())
    print(outcome.source, end="")
    print(_rule())
    print(f"Result: {outcome.value}")


if __name__ == "__main__":
    main()
