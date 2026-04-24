"""Trace exactly what the model sees and how the grammar tightens across retries.

Scenario: `gen.integer(1, 20, where=lambda x: x == 17)` against
Qwen2.5-0.5B-Instruct. A single-value predicate over 20 candidates
guarantees at least one rejection, and because the grammar tightens
deterministically after every reject, we get a clean, reproducible
trace.

This script wraps XGrammarEngine so we can observe:

  - the exact prompt bytes fed into the model on each sample
  - the GBNF grammar compiled for each sample
  - which token is masked in / masked out
  - which token argmax picks
  - when the predicate rejects and what message gets injected
  - how the grammar and prompt change on the next pass

Run: .venv/bin/python examples/trace_retry.py
"""

from __future__ import annotations

from pathlib import Path

from orate import gen, program
from orate.engine.xgrammar import XGrammarEngine


class TracingXGrammarEngine(XGrammarEngine):
    """Same engine, but prints every constraint/action at the boundary."""

    _sample_number: int = 0

    def __post_init__(self) -> None:
        # dataclass has no __post_init__ in parent; just reset counter.
        self._sample_number = 0

    def sample_int(self, min_val, max_val, *, excluded=None):
        excluded = excluded or set()
        self._sample_number += 1
        n = self._sample_number

        print()
        print("=" * 72)
        print(
            f"SAMPLE #{n}  — engine.sample_int({min_val}, {max_val}, excluded={sorted(excluded)})"
        )
        print("=" * 72)

        # Show the exact prompt bytes fed into the model.
        full_prompt = self._prime_text or ""
        if self._context:
            full_prompt += "\n" + "\n".join(f"[note: {t}]" for t in self._context) + "\n"
        print("PROMPT (bytes fed through llama.cpp on this sample):")
        for line in full_prompt.rstrip().split("\n"):
            print(f"  | {line}")
        if self._context:
            print(f"  (^ {len(self._context)} context note(s) injected via inject_context())")

        # Build the grammar the same way the parent does so we can show it.
        from orate.engine.xgrammar import _int_grammar

        grammar = _int_grammar(min_val, max_val, excluded)
        print()
        print("GRAMMAR compiled for this sample (GBNF):")
        print(f"  {grammar}")
        allowed = [i for i in range(min_val, max_val + 1) if i not in excluded]
        print(f"  (accept set: {allowed}  | {len(allowed)} candidates)")
        if excluded:
            print(f"  (tightened: these values are masked out: {sorted(excluded)})")

        value = super().sample_int(min_val, max_val, excluded=excluded)

        print()
        print(f"MODEL ARGMAX (under grammar mask) → {value}")
        return value

    def inject_context(self, text: str) -> None:
        print()
        print(f"  >> inject_context({text!r})")
        super().inject_context(text)


def _pick_model() -> str:
    for c in [
        "/Users/maltelandgren/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(c).exists():
            return c
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


@program
def pick_17():
    n = yield gen.integer(
        1,
        20,
        where=lambda x: x == 17,
        reject_message=lambda v: f"{v} is not the number 17",
        max_retries=30,
    )
    return n


def main() -> None:
    print("TRACE — gen.integer(1, 20, where=lambda x: x == 17) on Qwen2.5-0.5B")
    print()
    print("What to watch for:")
    print("  * grammar tightens on every reject (the excluded value literally")
    print("    cannot appear in the allowed set — the mask zeros its logit)")
    print("  * reject_message is appended to the prompt via inject_context()")
    print("  * the next sample re-tokenizes prompt + notes — the model sees")
    print("    the rejection as natural-language context before its next argmax")

    engine = TracingXGrammarEngine(model_path=_pick_model())
    # Deliberately neutral prompt — we DON'T tell the model the target
    # is 17. The predicate does the filtering; the model's argmax will
    # propose its favorite numbers first, each gets rejected, the
    # grammar tightens, and we watch convergence.
    engine.prime(
        "Pick any two-digit-or-less whole number from 1 to 20. Return only the number.\n\nA: "
    )

    result = pick_17().run(engine=engine)

    print()
    print("=" * 72)
    print(f"FINAL RESULT: {result}")
    print("=" * 72)
    print()
    print(f"Total samples: {engine._sample_number}")
    print(f"Total context notes injected: {len(engine._context)}")
    print()
    print("Notice the grammar narrowed on each reject. Argmax is deterministic,")
    print("so the ONLY way the model moves to a different value is if the accept")
    print("set changes (grammar tightens) or the prompt changes (context injected).")
    print("Both happen on every reject. That's the mechanism.")


if __name__ == "__main__":
    main()
