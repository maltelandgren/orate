# orate — project overview

## What this is

orate is a Python library for **programmatic decoding** over a local LLM. The developer writes an ordinary `def`, marks decision points with `yield`, and decorates the function with `@program`. Each yield hands a typed spec to a runtime that narrows the model's next-token logits to a grammar derived from the spec, samples one value, and feeds it back into the generator. Structured output, tool calls, and agent control flow stop being three different APIs and become one primitive: a generator that yields. The interesting bit is that yields can carry arbitrary Python predicates (`where=lambda v: ...`), so the constraint vocabulary is *logic*, not just types — the kind of thing JSON Schema can't say.

## The gap it fills

Today's stack pretends three problems are different:

1. **Structured output** — Pydantic / JSON Schema / `response_format`. Type-shaped. Can't express "a prime whose digits sum to 10."
2. **Tool calling** — a separate API surface (`tools=[...]`), a separate response shape, a separate dispatch path.
3. **Agent control flow** — you write a `while True: result = call_model(); if ...: break` loop in Python yourself.

None of them admit *value-level* constraints. JSON Schema can say "int between 0 and 99"; it cannot say "prime whose digits sum to 10." None of them compose: you can't have a model emit a structured object that *contains* a tool call mid-stream, on the same KV. And the agent loop runs in user code, blind to the model's logits — so even a structurally illegal step can only be caught after the fact, by a validator, with a retry that pays the prefix cost again.

orate folds the three problems into one mechanism: a generator whose yields are decision points, evaluated against grammar-constrained decoding at the logit-mask level on a single persistent KV. The same `yield` is structured output, tool call, sub-agent handoff. The same `where=` predicate gates all of them.

## The primitive

A `@program`-decorated Python generator. At each `yield gen.X(...)` the runner inspects the spec, builds a GBNF fragment, masks the logits, samples, sends the value back into the generator. The spec vocabulary:

- `gen.choice([...])` — one of a fixed set
- `gen.integer(lo, hi, where=...)` — bounded int with optional predicate
- `gen.string(max_len=, pattern=, where=...)` — unbounded string, optional regex
- `gen.boolean()`
- `gen.struct(x=..., y=..., where=lambda d: ...)` — a dict with a cross-field predicate
- `gen.tool(fn, **args)` — call a Python function, the result is the yielded value

The load-bearing piece is `where=`. It's a Python predicate. Not a post-hoc validator that runs after the model emitted; not a docstring hint. The runtime treats it as **part of the grammar**: where it can, the predicate becomes the constraint. Where it can't (because the domain is unbounded or the predicate closes over a yielded string), the runtime falls back to running the body and rejecting at the closure boundary. Either way the contract is: a value reaches the next yield only if it satisfies the predicate. On exhaustion we raise `GrammarExhausted` — we do not silently drop the constraint and return a wrong value.

Code lives in [`src/orate/gen.py`](../src/orate/gen.py) and [`src/orate/program.py`](../src/orate/program.py).

## How the constraint actually fires

Three layers of correctness, cheapest first:

**Layer 1 — Witness enumeration.** For finite domains (`Choice`, `Integer` in practical ranges, `Boolean`) the runtime evaluates the predicate over every candidate at compile time and narrows the grammar to the accept set. The model literally cannot emit a rejected value. So `gen.integer(0, 99, where=lambda n: is_prime(n) and digit_sum(n) == 10)` resolves to a 3-element grammar `19 | 37 | 73`. No retry loop. No dice. The grammar makes the wrong answer unrepresentable.

**Layer 3 — Forward-checking on `gen.struct`.** With a cross-field `where=`, each field binds, then the remaining fields' domains are recompiled with the predicate closed over the bound values. So `gen.struct(x=gen.integer(0, 10), y=gen.integer(0, 10), where=lambda d: d["x"] + d["y"] == 10)` works without either coordinate being hardcoded — `x` is sampled freely, then `y`'s domain narrows to `{10 - x}`.

**Predicate verification on yields with closures.** When a `where=` references a previously-yielded value on an unbounded domain — the canonical case is the `algebra_step` body where the third yield's predicate is `equivalent_under(rule, before, after)` — neither witness enumeration nor forward checking can fire. The grammar can only enforce *syntactic* shape. So the engine samples under the syntactic grammar, then re-drives the program body to verify the closure with the parsed values; on failure the call is rejected, a session note is appended to the KV, and the model re-decodes. This is how `@algebra_step("2x + 3y = 12", "isolate_var", "x = (12 - 3y) / 2")` works: the third yield's `where=` runs SymPy at decode time, and a model emission like `"x = 12 - 3y"` (which forgets the divide-by-2) gets rejected, with the rejection note in the KV when the model tries again.

The "two-tier" doc covers the architectural choice that makes this clean: see [`docs/design/two-tier-and-transitions.md`](design/two-tier-and-transitions.md).

## Where this fits relative to existing tools

XGrammar, Outlines, Guidance, JSONFormer all do grammar-constrained decoding at the logit-mask level. orate uses XGrammar as its mask-applier — that's plumbing. The contribution is what you build on top:

- **Predicates over Python.** The constraint vocabulary is full Python, not a schema language. `is_prime`, `equivalent_under`, `derivable_under`, anything you can write a function for. Witness enumeration makes this practical for finite domains; the rejection-with-closure path handles the rest.
- **One primitive across structured output / tool calling / agent control.** A `@program` body can interleave a `gen.string`, a `gen.tool`, a `gen.choice`, and another `gen.string`. One inference. One KV. No tool-use API.
- **Source-in-prompt.** The model sees the same `@program` source the developer wrote, with `description=` annotations baked in as comments. The model's mental model of "what is this thing asking for" is the source code, not a schema's JSON serialization.
- **Persistent-KV session mode.** A `Session` is a long-lived KV tape with an outer grammar that admits free text *or* one `@-call`. The model can author and call its own tools mid-inference (`@make_new_program`) without ever leaving the same KV — the bootstrap is grammar-switched, not session-switched.

## Session mode and the model authoring its own tools

Set up a `Session(engine=..., programs={...})` ([`src/orate/session.py`](../src/orate/session.py)). The session has a registry of `@program`s and an outer grammar of the form `text_chunk | "@a(" a_body ")" | "@b(" b_body ")" | …`. Each registered program contributes its own body grammar as a named sub-rule. The model samples one continuous token stream; when the matcher accepts on a call-prefix, the runtime knows which program is being invoked and decodes the args under that program's body grammar. There's no separate "tool dispatch" turn — the call is just a region of the grammar.

Now the trick. `make_new_program` is registered by default. When the model emits `@make_new_program("algebra_step", "one legal algebraic transformation")`, the runtime grammar-switches mid-decode to `PROGRAM_SOURCE_GRAMMAR` (a small Python subset — see [`src/orate/meta.py`](../src/orate/meta.py)) and resumes sampling on the same KV. The model emits a valid `@program` source — the grammar guarantees well-formedness, the validator AST-checks it, the compiler sandbox-execs it under a locked-down builtins dict, and the runtime registers the new tool. The outer grammar is rebuilt with a new sub-rule for the new program. From this point on, the same model in the same KV can invoke `@algebra_step(...)` and the args are grammar-bound to whatever schema the model just wrote for itself.

This is demoed end-to-end on Qwen-7B local in [`examples/smoke_session.py`](../examples/smoke_session.py): the model emits `@make_new_program("theme_ideas", ...)`, the runtime compiles + registers, and the next turn's `@theme_ideas(medieval feast, chainmail banners, roast beef, flute)` decodes each arg under a `gen.choice` populated with options the model itself just authored.

## Per-leaf grammars + transition-based composition

A recent architectural cleanup. Each leaf `@program` carries its own self-contained GBNF (call-site grammar derived once, at registration). The "outer grammar" the engine samples under is just a prefix-alternation `text_chunk | "@a(" | "@b(" | ...` — no leaf bodies inlined. When the matcher accepts on a call-prefix `"@a("`, the runtime calls `engine.sample_under(leaf_a_body_grammar)` — a *fresh sample call* against `a`'s body grammar — then appends the literal `")"` to the KV and returns to outer sampling.

Same shape `make_new_program` already used; we generalised. The implications matter: adding a leaf only recompiles its own body grammar; mode switches and `make_new_program` mutations are cheap; the matcher state can never leak across leaves; per-leaf rule conflicts (helper rules with the same name) become impossible because helpers live in their own namespace.

ADR: [`docs/design/two-tier-and-transitions.md`](design/two-tier-and-transitions.md).

## Two tiers of @program

The split that drops out of the cleanup:

- **Leaves** (`invocable=True`, default) — straight-line yields, derivable call-site grammar. Embeddable in a parent's grammar as a named sub-rule. Run via `.run(engine=...)` or, in session mode, invoked by the model emitting `@name(args)`.
- **Composers** (`invocable=False`) — arbitrary Python control flow (loops, branches, state), no derivable call-site grammar. Run directly via `.run(engine=...)`. Orchestrate leaves.

Both are `@program`. The split exists because some programs are *fully grammar-describable* — the matcher can run the whole emission in one shot — and some need Python in the loop between yields. The `gen.alternative([leaves])` primitive (on `feat/flavor-b-full`) lets composers expose a runtime alternation over leaves to the model: the model picks which leaf to run next, the matcher dispatches, the composer keeps looping. This is how you write an agent loop without a `Session` class — see [`examples/legal_steps/act4_algebra_composer.py`](../examples/legal_steps/act4_algebra_composer.py) on `feat/flavor-b-full`. Same algebra-solve trace as the Session version. No driver framework. Just a `@program(invocable=False)` with `while True: yield gen.alternative([algebra_step, done])`.

## Where this is going

JIT grammar segmentation. The architectural endpoint that collapses the leaf/composer distinction. Walk a program body, identify segments of yields whose grammars are independent of prior yielded values, fuse each segment into one grammar, recompile only at boundaries where Python state actually matters. The compiler decides how many grammar segments to emit; the runtime executes them. Some programs end up as one segment (today's "leaf"), some as N segments interleaved with Python (today's "composer"), and most are somewhere in between. The user writes the same `@program` either way.

Vision doc: [`docs/design/jit-grammar-segmentation.md`](design/jit-grammar-segmentation.md).

## The benchmark

A free-text vs constrained Qwen2.5-7B-Instruct-Q4_K_M head-to-head on 7 algebra problems, deterministic argmax, same model, same prompt material.

- **Free text**: 4/7 correct in 258s. The slips: `x=4` on `3x + 5 = 14`, `x=12` on `2x + 3 = x + 9`, `x=39` on `5 - 2x = 1`. Plausible-looking arithmetic that's just wrong.
- **Constrained `@algebra_step`**: 6/7 correct in 75s (4× faster than free text wall-clock, on a *constrained* path), with **11 predicate-rejected steps caught en route**. The single failure is `eq_distribute`, which hit `max_calls` — a budget issue, not a correctness one.
- **Throughput**: free-text 10–17 tok/s; constrained 5–8 tok/s steady-state. (An earlier run had a 232s cold-start outlier — XGrammar's first-compile JIT cost — which we killed by caching compiled grammars by GBNF source on the engine and warming the cache during `Session.__init__`.)

Source: [`bench/results/legal_steps_2026-04-25_1200.md`](../bench/results/legal_steps_2026-04-25_1200.md).

The take: the predicate isn't "trying harder" or "thinking step-by-step." It's a gate the model can't pass without producing a step that is, in fact, mathematically valid under SymPy. The 11 rejections are the gate firing. The model proposes; the predicate disposes; the predicate is the law.

## Three end-to-end demos on Qwen-7B local

- **[`examples/legal_steps/act4_algebra_demo.py`](../examples/legal_steps/act4_algebra_demo.py)** — the model solves a linear equation by emitting a sequence of `@algebra_step("before", rule, "after")` calls. Each call's `after` is verified equivalent to `before` under `rule` via SymPy. Terminates with `@done("x = 3")`.
- **[`examples/legal_steps/act4_logic_demo.py`](../examples/legal_steps/act4_logic_demo.py)** — the model proves `A → B; B → C; A ⊢ C` by emitting `@inference_step(premises, rule, conclusion)` calls, each verified `derivable_under` the named rule (modus ponens, hypothetical syllogism, …). Terminates with `@qed`.
- **[`examples/d20/act3_combat_demo.py`](../examples/d20/act3_combat_demo.py)** — narrative session mode-switches into combat (`@enter_combat`), three composed NPC `@program`s (Aria, Borin, the hooded figure — each with its own action set and damage cap) take turns, mode-switches back (`@exit_combat`). Same KV, same model, the grammar reshapes mid-conversation.

## What this lets you do

Concrete things that were hard before, with where they live:

**Predicate constraints on output.**
```python
gen.integer(0, 99, where=lambda n: is_prime(n) and digit_sum(n) == 10)
```
JSON Schema can't say this. Witness enumeration runs the predicate over the 100 candidates, the grammar narrows to `{19, 37, 73}`, the model picks one. The grammar makes everything else unrepresentable.

**Cross-field constraints, no retry.**
```python
gen.struct(x=gen.integer(0, 10), y=gen.integer(0, 10),
           where=lambda d: d["x"] + d["y"] == 10)
```
Forward-checking. As `x` binds, `y`'s domain narrows. No validator-after-the-fact. No retry loop.

**Tool calls and structured output in the same yield stream.**
```python
@program
def turn():
    target = yield gen.choice(["dragon", "goblin", "ghost"])
    weakness = yield gen.tool(lookup_enemy_weakness, enemy=target)
    attack = yield gen.struct(weapon=..., stance=...)
    damage = yield gen.integer(1, 10 + (5 if weakness != "none" else 0))
```
One inference. One KV. One program. The `weakness` tool call's return value participates in the *next* yield's bound — the integer's `max_val` is a Python expression closing over the tool result.

**Models authoring their own tools mid-inference.** [`examples/smoke_session.py`](../examples/smoke_session.py). Qwen-7B emits `@make_new_program("theme_ideas", ...)`, the runtime compiles + registers + rebuilds the grammar, the same model invokes `@theme_ideas(...)` with each arg constrained to options it just wrote. Same KV from start to finish.

**Provably-legal multi-step reasoning.** The algebra benchmark above. [`examples/legal_steps/checkers.py`](../examples/legal_steps/checkers.py) is SymPy doing equivalence under named rules. The grammar makes "produce a step" the only available action; the predicate makes "produce a step that's actually a legal transformation" the only acceptable action. Eleven model proposals failed the predicate in the 7-problem benchmark; none of them reached the user.

## Where you'd push back

A few honest places:

1. **Predicate verification on closures isn't free.** When the third yield's `where=` closes over the first two, witness enumeration can't fire and the runtime falls back to syntactic-grammar sampling + post-hoc closure check. The 11 rejections in the benchmark are real tokens spent. We have a story for this — JIT grammar segmentation should compile cross-yield predicates into a fused grammar where possible — but today the rejection loop is the honest fallback.
2. **`PROGRAM_SOURCE_GRAMMAR` is a small Python subset.** Straight-line `var = yield gen.method(...)` then `return`. No branches. No loops. No `yield from`. The model can author tools but only of the leaf shape. Composers it doesn't write yet. This is fine for the four-act story; it'll need to grow for richer self-authoring.
3. **Local-first by design.** The truest form of orate constrains inference at the logit level, which means a controlled inference stack (llama-cpp + XGrammar). API-backed engines can do something structural (JSON mode + retry) but not fundamental. If you want this against Claude or GPT, you'd be writing a different library.

## Repo layout, in one breath

[`src/orate/gen.py`](../src/orate/gen.py) holds the Gen primitives. [`src/orate/program.py`](../src/orate/program.py) is the decorator. [`src/orate/session.py`](../src/orate/session.py) is the persistent-KV runner. [`src/orate/meta.py`](../src/orate/meta.py) is the program-source grammar + sandbox compiler. [`src/orate/body_grammar.py`](../src/orate/body_grammar.py) is the AST → GBNF derivation. [`src/orate/engine/xgrammar.py`](../src/orate/engine/xgrammar.py) is the local engine. Demos in [`examples/legal_steps/`](../examples/legal_steps/) and [`examples/d20/`](../examples/d20/). Benchmarks in [`bench/results/`](../bench/results/). Design docs in [`docs/design/`](design/).

Status: pre-alpha, hackathon birth (Built with Opus 4.7, Apr 2026), in active development. The primitives work end-to-end on Qwen 0.5B through 7B. The interesting questions are upstream: how much of a `@program`'s body can the compiler statically grammar-fuse, and where does the rejection-with-closure path stop being the honest fallback and start being the bottleneck.
