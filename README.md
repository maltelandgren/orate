# orate

**Programmatic grammar decoding for local LLM inference.**

> 🎬 **Watch the 3-min submission video:** [`video/orate_submission.mp4`](video/orate_submission.mp4)
>
> 📖 **Read the in-depth writeup:** [`docs/project_overview.md`](docs/project_overview.md)

> Structured output constrained the shape.
> Tool calling constrained the side effect.
> **orate lets the model enforce the legality of its own thought.**

You write a Python generator. At every `yield`, the model proposes; orate's
grammar + your `where=` predicate decide what's allowed. Types, tool calls,
and control flow are all the same yield stream. No JSON-mode fallback, no
post-parse, no XML scaffolding — the runtime constrains the next token at
the logit level.

> Pre-alpha. Built for *Built with Opus 4.7: a Claude Code hackathon*
> (Apr 21–26, 2026). Problem statement **#2 — Build For What's Next**.

---

## The one primitive

```python
from orate import program, gen

@program
def two_digit_prime_with_digit_sum_10():
    n = yield gen.integer(
        10, 99,
        where=lambda v: is_prime(v) and digit_sum(v) == 10,
    )
    return n
```

Every `yield gen.X(...)` is a decision point. The grammar argument
constrains *shape*; the `where=` predicate constrains *value*. On
predicate reject, the engine tightens its accept set and re-samples from
the same KV — never returns a value the predicate didn't approve.

---

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"           # kernel + tests + MockEngine
.venv/bin/pip install -e ".[local,dev]"     # + llama-cpp-python + xgrammar
```

Local engine needs a GGUF model. Demos auto-discover Qwen2.5-7B / 3B at
`~/models/`. Bring your own:

```bash
mkdir -p ~/models && cd ~/models
# (download a Qwen2.5 instruct GGUF from Hugging Face)
```

---

## 30-second tour (no model required)

`MockEngine` is a deterministic random sampler — perfect for shaking
out a `@program` before paying for a real run.

```python
from orate import program, gen
from orate.engine.mock import MockEngine

@program
def square_pair():
    a = yield gen.integer(1, 9)
    b = yield gen.integer(1, 9, where=lambda v: v == a)  # closure on `a`
    return {"a": a, "b": b}

print(square_pair().run(engine=MockEngine(seed=7)))
# => {'a': 6, 'b': 6}
```

Predicates close over earlier yields. They run in pure Python — anything
you can `import`, you can constrain on.

---

## The four moments (each maps to a runnable demo)

The submission video tells the story in four beats. Each is a real
program in `examples/`.

### 1 · Predicates beat schemas

A JSON schema can declare types. It cannot declare *"end ≥ start +
duration"* or *"start hour ∈ business_hours"*. orate makes that an
expression, not a workaround:

```python
business_hours = range(9, 18)

@program
def book_meeting(duration_h: int):
    start = yield gen.datetime(where=lambda d: d.hour in business_hours)
    end   = yield gen.datetime(
        where=lambda e: e - start == timedelta(hours=duration_h),
    )
    return {"start": start, "end": end}
```

Cross-field equation closing over a parameter. Single yield stream.

> *Demo:* the `@program` shape above runs in any of the bench / d20
> examples; the [Quickstart](#-30-second-tour-no-model-required) covers
> the predicate-on-`gen.integer` case.

---

### 2 · Legal-step enforcement on hard problems

> Free-text Qwen2.5-7B got **5 of 10** algebra problems.
> The same model, same weights, under one `@algebra_step` program,
> got **9 of 10** — with **16 illegal-step rejections** caught en route.

The whole composer is six lines:

```python
from orate import program, gen
from examples.legal_steps.checkers import equivalent_under

@program
def algebra_step():
    before = yield gen.string(max_len=30, pattern="[0-9a-z +\\-*/=()]")
    rule   = yield gen.choice(["simplify", "combine_like", "isolate_var", "evaluate"])
    after  = yield gen.string(
        max_len=30, pattern="[0-9a-z +\\-*/=()]",
        where=lambda s: equivalent_under(rule, before, s),  # SymPy on every emit
    )
    return {"before": before, "rule": rule, "after": after}
```

A trace of Qwen-7B solving `3x + 5 = 14`, every line predicate-verified
before the next yield is allowed:

```
@algebra_step("3x + 5 = 14", simplify, "3x = 9")    ✓
@algebra_step("3x = 9", isolate_var, "x = 3")       ✓
@done(3)
```

```bash
# Run the algebra composer (needs local Qwen2.5):
.venv/bin/python examples/legal_steps/act4_algebra_demo.py

# Reproduce the 5/10 vs 9/10 benchmark:
.venv/bin/python bench/measure_legal_steps.py
# Outputs bench/results/legal_steps_<date>.{json,md}
```

Most recent run: [`bench/results/legal_steps_2026-04-26_1759.md`](bench/results/legal_steps_2026-04-26_1759.md).
Same pattern works for propositional logic — see
[`examples/legal_steps/logic.py`](examples/legal_steps/logic.py)
(`@inference_step` with a `derivable_under` predicate covering modus
ponens, modus tollens, hypothetical syllogism, conjunction,
simplification) and the
[`act4_logic_demo.py`](examples/legal_steps/act4_logic_demo.py) runner.

---

### 3 · One KV cache, many grammars

`Session` is the persistent-KV driver. The model emits `@`-calls; the
runtime decodes each one under its body grammar, runs the body (verifying
predicates), feeds the result back into the same KV, and continues.

The D&D session demo runs a tavern scene through narrative tools, a
client-resolved skill check, an out-of-character meta-comment, then
**reshapes the grammar atomically** to per-character combat programs
when the model emits `@enter_combat`:

```python
session = Session(
    engine=XGrammarEngine(model_path="..."),
    programs={"narrate": narrate, "roll": roll, "meta": meta,
              "enter_combat": enter_combat},
    system=SYSTEM,
    allow_free_text=False,    # tool-only — every sample is an @-call
)
# Combat-mode programs (visible only after @enter_combat):
session.register("aria_attack", aria_attack, mode="combat")
session.register("borin_attack", borin_attack, mode="combat")
session.register("hooded_figure_attack", hooded_figure_attack, mode="combat")
session.register("exit_combat", exit_combat, mode="combat")

session.user("Run this tavern-then-combat scene as @-calls...")
for event in session.advance():
    ...   # FreeText / ProgramInvoked / TurnEnded
```

A clean trace from the demo (one inference, two grammar swaps):

```
@narrate("You try to convince the hooded figure this is all a misunderstanding…")
@roll(persuasion, 14)
        → {d20: 1, success: false}
@meta("Haha — a 1. Sorry, won't cut it.")
@narrate("'My fist is going to make you miss understanding, punk.'")
@enter_combat(hooded_figure)
        ↑  grammar swap on the same KV
@hooded_figure_attack(dagger, aria, 3)
@aria_attack(longsword, hooded_figure, 5)
@borin_attack(warhammer, hooded_figure, 6)
@exit_combat(victory)
```

Aria's `@aria_turn` enforces the D&D action-economy rule across two
fields in plain Python — *"you can't both cantrip and cast a spell on
the same turn"* — which JSON Schema cannot express:

```python
@program
def aria_turn():
    move = yield gen.struct(
        action       = gen.choice(["longsword", "fireball", "vicious_mockery", "hold"]),
        bonus_action = gen.choice(["dagger", "healing_word", "thorn_whip", "hold"]),
        where        = lambda d: not (
            d['action'] in NON_CANTRIPS
            and d['bonus_action'] in SPELLS
        ),
    )
    return move
```

```bash
.venv/bin/python examples/d20/act3_full_demo.py
```

---

### 4 · The model authors its own primitive

The finisher. Hand the model a problem with no built-in tool — *factorize
1147 into p × q with p, q > 1* — and pre-register only `@done` and
`@make_new_program`. The model writes its own `@program` whose `where=`
clauses **guarantee** the answer is correct before the runtime returns
it:

```
@make_new_program("factor_1147", "two factors of 1147 greater than 1")

[session: synthesizing program…  grammar switch → PROGRAM_SOURCE_GRAMMAR]

@program
def factor_1147():
    n = yield gen.integer(1147, 1147)
    p = yield gen.integer(2, 1146, where=divides(n))
    q = yield gen.integer(2, 1146, where=multiplies_to(n, p))
    return {"p": p, "q": q}

[session: validated · compiled · registered  →  @factor_1147 callable]

@factor_1147(1147, 31, 37)
        → {'p': 31, 'q': 37}
        divides(1147)(31)             → 1147 % 31 == 0    ✓
        multiplies_to(1147, 31)(37)   → 31 × 37 == 1147   ✓

@done("31 and 37")
```

The body's six lines are sampled under `PROGRAM_SOURCE_GRAMMAR` (the meta
grammar). The invocation `(1147, 31, 37)` is sampled under the body
grammar derived from those six lines, with `divides(1147)` and
`multiplies_to(1147, 31)` re-run on every candidate token. **31 × 37 =
1147 isn't a guess — the grammar + predicate gate forced it.**

```bash
.venv/bin/python examples/legal_steps/act4_factorize.py
```

The available `where=` predicates the model can compose are in
[`src/orate/meta_predicates.py`](src/orate/meta_predicates.py): `is_prime`,
`divides`, `multiplies_to`, `sums_to`, `divisible_by`, `is_square`,
`is_palindrome`, `coprime_with`, `length_eq`, `digit_sum_eq`, `lt`, `gt`.

---

## What works today

| Capability | State |
|---|---|
| `@program` decorator + generator runner with predicate tightening | shipped |
| `gen.{integer, choice, string, boolean, struct, datetime, tool}` | shipped |
| `where=` predicate verification on every emission | shipped |
| `XGrammarEngine` (llama-cpp-python + XGrammar, local GGUF) | shipped |
| `MockEngine` (deterministic random sampler, model-free) | shipped |
| Compiled-grammar cache + warmup (kills cold-start latency) | shipped |
| `Session` driver: persistent KV, @-call emission, mode switching | shipped |
| Tool calls = `gen.tool` yield with `ends_turn=True` (no separate API) | shipped |
| Per-leaf body grammar + transition-based composition | shipped |
| `gen.alternative([leaves])` composer primitive | shipped |
| `@make_new_program` — model authors a `@program` mid-session | shipped |
| 14-entry predicate library for model-authored `where=` clauses | shipped |
| Argmax decoding by default; per-call temperature override | shipped |
| Session-level T-escalation on consecutive predicate rejections | shipped |
| 10-problem algebra benchmark (free-text vs constrained) | shipped (5/10 vs 9/10) |

---

## Layout

```
src/orate/
  program.py            # @program decorator + ProgramInvocation.run()
  gen.py                # gen.{choice,integer,string,boolean,struct,tool,datetime,alternative}
  session.py            # Session — persistent-KV @-call driver
  meta.py               # PROGRAM_SOURCE_GRAMMAR + @make_new_program
  meta_predicates.py    # 14 predicates the model can compose in where=
  body_grammar.py       # AST → GBNF derivation for yield bodies
  engine/
    protocol.py         # Engine Protocol + optional capabilities
    mock.py             # MockEngine — deterministic random sampler
    xgrammar.py         # XGrammarEngine — llama-cpp + XGrammar local decode

examples/
  legal_steps/
    algebra.py            # @algebra_step + @done + equivalent_under
    logic.py              # @inference_step + @qed + derivable_under
    checkers.py           # SymPy-backed equivalent_under, derivable_under
    act4_algebra_demo.py  # Qwen-7B solves an algebra problem under @algebra_step
    act4_logic_demo.py    # Qwen-7B proves a propositional theorem under @inference_step
    act4_factorize.py     # Qwen-7B authors its own factorizer (Page 5)
  d20/
    dice.py               # @narrate + @roll + @meta tools
    characters.py         # aria/borin/hooded_figure programs + enter/exit_combat
    act3_full_demo.py     # one KV: narrative → combat → narrative

bench/
  measure_legal_steps.py  # 10-problem free-text vs constrained suite
  results/                # markdown + JSON per run

video/
  scenes/full_video_v2.py # the 1080p60 submission scene
  renders/                # rendered mp4 (1080p60 submission committed)
```

---

## Design stance

- **The grammar is the guarantee.** `where=` predicates never silently
  drop a constraint — on accept-set exhaustion the runtime raises
  rather than returning a value the predicate didn't approve.
- **Argmax by default.** Stochastic sampling is an opt-in
  (`temperature=...`); Session-level escalation kicks in only when
  consecutive predicate rejections suggest the model has locked into a
  wrong attractor.
- **Engine-agnostic authoring layer.** Every example runs against
  `MockEngine` before a real model touches it. Swapping in
  `XGrammarEngine` changes the proposer's quality, not the program's
  correctness.
- **Local first.** The truest form of this library constrains inference
  at the logit level — something only a controlled inference stack
  exposes. API fallbacks would be structural (JSON mode + retry), not
  fundamental.
- **One primitive.** Types, tool calls, control flow, and even
  *meta-authoring* of new programs are all `yield gen.X` against an
  engine — same KV, same grammar machinery, no separate APIs.

---

## More

For the architectural deep-dive — three layers of correctness, the
two-tier `@program` split, transition-based composition, the JIT
grammar segmentation roadmap, and the long-form benchmark commentary
— see [`docs/project_overview.md`](docs/project_overview.md).

## License

MIT. See [LICENSE](LICENSE).

## Credits

Submission video music: **"On The Eve"** by *The Grey Room / Density & Time*, from the [YouTube Audio Library](https://www.youtube.com/audiolibrary).
