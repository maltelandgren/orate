"""Act-4 finisher: the model authors its own primitive mid-session, then uses it.

This is the climactic beat. The session starts with @algebra_step + @done
pre-registered (single-equation rule-based rewriting). We pose a problem
that doesn't fit the existing primitives well — a quadratic. The model
recognises the gap and emits @make_new_program, authoring a structured
emission that captures the new shape (e.g. roots of a quadratic). The
runtime grammar-switches mid-decode, samples the source under
PROGRAM_SOURCE_GRAMMAR, validates, sandbox-execs, registers, rebuilds
the outer grammar — all on the same KV. Then the model uses the
primitive it just defined to finish the solve.

The honest scope: model-authored programs admit *typed schemas*, not
predicate-bound bodies. The structure of the output is enforced (the
model can't emit ill-typed args) but the mathematical correctness of
the answer rides on the model's reasoning. That's the current shape;
predicate-bound meta-authorship is on the JIT segmentation roadmap.

What the visual reveals: source materialising on screen, runtime
compile log, then the new primitive being invoked with grammar-bound
args. The library grew during the inference. One KV.

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
You output ONLY @-calls. No prose. No markdown.

Pre-registered tools:
  @algebra_step("before", rule, "after") — one legal algebraic
    transformation, where rule ∈ {simplify, combine_like,
    isolate_var, evaluate}. The runtime verifies algebraic
    equivalence — applying a rule that doesn't preserve equivalence
    is rejected.
  @done("answer") — terminate the chain with the final answer.

You may also emit:
  @make_new_program("name", "description") — author a NEW @program
    on the fly. The runtime will prompt you for the source body
    (yields + return), validate, compile, and register it. From then
    on, you can invoke `@name(args)` and the runtime will decode the
    args under that program's grammar.

When the existing primitives don't fit the problem cleanly (e.g. a
quadratic where step-by-step linear algebra is awkward), prefer
@make_new_program to design a primitive that captures the move you
actually want to make. Then use it.
"""


PROBLEM = """\
Solve the quadratic: x^2 - 5x + 6 = 0. Find both roots.

Hint: this isn't a linear equation. Consider whether @algebra_step's
rules fit cleanly, and whether you'd rather author a primitive that
captures the structure of a quadratic root pair.
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
