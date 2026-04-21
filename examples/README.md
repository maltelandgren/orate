# Examples — the four acts

The library's pitch is a four-act progression. Each act is a runnable
script that stands on its own and reads in under a minute.

Run from the repo root:

```bash
.venv/bin/python examples/act_01_schema_breaks.py
.venv/bin/python examples/act_02_predicate_fixes.py
.venv/bin/python examples/act_03_unified_yield.py
.venv/bin/python examples/act_04_arc_sketch.py
```

All four run against `MockEngine` by default — a random sampler with
no real model. Acts 1-3 show the library behavior deterministically
(the mechanism isn't about the model's output quality). Act 4's
success-rate depends entirely on the proposer quality; with MockEngine
you see the retry loop and Phase-C context injection, not convergence.

## The arc

- **Act 1 — schemas are a ceiling.** Types can't express "prime with
  digit-sum 10." Post-hoc validation throws work away. Sketched in
  `act_01_schema_breaks.py`.
- **Act 2 — `where=` makes the constraint first-class.** Grammar
  tightens on reject; `reject_message` injects a steering hint (Phase
  B). Deterministic correctness, no dice. `act_02_predicate_fixes.py`.
- **Act 3 — tool calls are just yields.** One `@program` mixes
  `gen.choice`, `gen.struct`, `gen.tool`, and `gen.integer` — no
  separate tool-use API. `act_03_unified_yield.py`.
- **Act 4 — the LLM writes its own `@program`.** Applied to
  ARC-AGI-2: the model emits a `Program` AST; the verifier checks it
  against the training demonstrations; `describe_mismatch` gets
  injected as Phase-C retry context. `act_04_arc_sketch.py`.

## For real runs

Swap the engine. The library was designed engine-agnostic on day one:

```python
# Local + grammar-constrained:
from orate.engine.xgrammar import XGrammarEngine
engine = XGrammarEngine(model_path="/path/to/qwen2.5-7b.gguf")

# OpenRouter fallback (no forced-token, uses JSON mode + retry):
from orate.engine.openrouter import OpenRouterEngine
engine = OpenRouterEngine(model="anthropic/claude-opus-4-7")
```

Then pass `engine=...` instead of `MockEngine(...)` in any of the
scripts.
