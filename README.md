# orate

**Programmatic decoding over local LLM inference.** The developer writes a generator; the model is an oracle consulted at marked yield points. Structured output, tool calls, and agent control flow collapse into one primitive: `yield`.

> Pre-alpha. Built for *Built with Opus 4.7: a Claude Code hackathon* (Apr 21–26, 2026). Problem statement **#2 — Build For What's Next**.

## The argument, in four acts

**Act 1 — Schemas are a ceiling.** A JSON schema can declare types. It cannot declare "a prime whose digits sum to 10" or "a word that is both a fruit and a color." These are *value-level* properties; types are the wrong vocabulary for them. See [`examples/act_01_schema_breaks.py`](examples/act_01_schema_breaks.py).

**Act 2 — Predicates move the bar.** A `where=` predicate on a gen spec turns a value-level constraint into a first-class citizen. On reject the accept set tightens and the engine re-samples; `reject_message` injects a steering hint. Deterministic correctness, no dice. See [`examples/act_02_predicate_fixes.py`](examples/act_02_predicate_fixes.py).

**Act 3 — Programs subsume tool-calling.** A coroutine's `yield` is a decision point. So is a tool call. So is a sub-agent handoff. Why are these three different APIs today? They need not be:

```python
@program
def turn():
    action = yield gen.choice(["attack", "speak"])
    if action == "attack":
        target = yield gen.choice(["dragon", "goblin", "ghost"])
        weakness = yield gen.tool(lookup_enemy_weakness, enemy=target)  # tool-as-yield
        attack = yield gen.struct(
            weapon=gen.choice(["sword", "bow", "staff"]),
            stance=gen.choice(["aggressive", "defensive"]),
        )
        damage = yield gen.integer(1, 10 + (5 if weakness != "none" else 0))
        return {"target": target, "weakness": weakness, "attack": attack, "damage": damage}
    line = yield gen.string(max_len=140)
    return {"line": line}
```

One `@program`. One KV cache. One engine. No separate tool-use API. See [`examples/act_03_unified_yield.py`](examples/act_03_unified_yield.py).

**Act 4 — The model writes the program.** For a new task, the model authors its own `@program` at runtime — a typed AST over a small DSL, grammar-constrained by the same machinery that filtered scalar values in Act 2. The program is verified against the task's demonstrations; mismatches tighten the program-level grammar and inject the diff as natural-language context. The same mechanism, one level up.

Applied to **ARC-AGI-2**: the model proposes a transformation rule (a `Program` over the grid DSL), orate verifies it on the training demonstrations, and Phase-C retry surfaces `describe_mismatch(...)` as a steering note. See [`examples/act_04_arc_sketch.py`](examples/act_04_arc_sketch.py) and [`src/orate/arc/solve.py`](src/orate/arc/solve.py).

## Status snapshot

- **Kernel:** `@program` decorator + generator runner; `gen.choice / integer / string / boolean / struct / tool` primitives with deterministic grammar tightening on `where=` reject; Phase-B context injection via `reject_message`; Phase-C whole-program retry via `@program(whole_program_retries=N)` + `reject_program(msg)`.
- **Engine:** `XGrammarEngine` running llama-cpp-python + XGrammar locally, with forced-token (jump-forward-decode) optimization and grammar-mask on every sample. Tested against `/Users/maltelandgren/models/qwen2.5-*.gguf`. `MockEngine` for offline tests.
- **ARC:** data loader (task JSON + grid helpers), ASCII + PNG rendering (canonical 10-color palette), 14-primitive DSL executor, verifier with `describe_mismatch` (for Phase-C retry context), `solve_task` proposer that runs the full meta-programming loop.
- **Tests:** 80 unit tests + 12 model-gated (`tests/test_xgrammar_engine.py`, `tests/test_gen_against_local.py`) passing.
- **Lines of code:** ~2000 (src) + ~1200 (tests) + ~400 (examples).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"           # kernel + tests
.venv/bin/pip install -e ".[local,dev]"     # + llama-cpp-python + xgrammar (local engine)
.venv/bin/pip install -e ".[arc,dev]"       # + matplotlib (PNG rendering for ARC)
```

## Quickstart

```python
from orate import gen, program
from orate.engine.xgrammar import XGrammarEngine

@program
def two_digit_prime_with_digit_sum_10():
    n = yield gen.integer(
        10, 99,
        where=lambda v: is_prime(v) and digit_sum(v) == 10,
        reject_message=lambda v: f"{v} is not a prime whose digits sum to 10",
    )
    return n

engine = XGrammarEngine(model_path="/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf")
engine.prime("You answer with a single value matching the stated constraint.\n\nQ: a two-digit prime whose digits sum to 10\nA: ")

n = two_digit_prime_with_digit_sum_10().run(engine=engine)
# => 19 (or 37 or 73 — the three satisfying values in range)
```

For the ARC demo:

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 arc-data/ARC-AGI-2
.venv/bin/python examples/act_04_arc_sketch.py
```

## Layout

```
src/orate/
  program.py          # @program decorator + runner (Phase-C retry)
  gen.py              # gen.choice/integer/string/boolean/struct/tool + tightening
  engine/
    protocol.py       # Engine Protocol + optional capabilities
    mock.py           # MockEngine — random sampler, seeded
    xgrammar.py       # XGrammarEngine — local grammar-constrained decoding
  arc/
    data.py           # ArcTask + task JSON loader
    render.py         # grid_to_ascii / save_grid_png (matplotlib, 10-color palette)
    dsl.py            # 14 transformation primitives + Program AST + execute
    verify.py         # verify_on_train + describe_mismatch (Phase-B retry context)
    solve.py          # solve_task — the meta-programming proposer (Act 4)

examples/
  act_01_schema_breaks.py
  act_02_predicate_fixes.py
  act_03_unified_yield.py
  act_04_arc_sketch.py
  smoke_local.py      # runs against real Qwen2.5 locally

tests/                  # 80 unit + 12 model-gated
```

## Design stance

- **Determinism by default.** The engine uses argmax over grammar-masked logits. Stochastic sampling is a future explicit opt-in, not a hidden correctness mechanism.
- **Grammar is the guarantee; the model is the proposer.** `where=` predicates never silently drop constraints — on exhaustion we raise `GrammarExhausted`, we do not return a wrong value.
- **Engine-agnostic authoring layer.** Every example ran against `MockEngine` before a real model touched it. Swapping in `XGrammarEngine` or (future) an API-backed engine changes the proposer's quality, not the program's correctness.
- **Local first.** The library's truest form constrains inference at the logit level — something only a controlled inference stack exposes. API fallbacks are structural (JSON mode + retry), not fundamental.

## License

MIT. See [LICENSE](LICENSE).
