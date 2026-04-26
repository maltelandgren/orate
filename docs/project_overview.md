# orate — project overview

> 3-min submission video: [`video/orate_submission.mp4`](../video/orate_submission.mp4).

## What this is

orate is a Python library for **programmatic decoding** over a local LLM. The developer writes an ordinary `def`, marks decision points with `yield`, and decorates the function with `@program`. Each yield hands a typed spec to a runtime that narrows the model's next-token logits to a grammar derived from the spec, samples one value, and feeds it back into the generator. Structured output, tool calls, and agent control flow stop being three different APIs and become one primitive: a generator that yields. The interesting bit is that yields can carry arbitrary Python predicates (`where=lambda v: ...`), so the constraint vocabulary is *logic*, not just types — the kind of thing JSON Schema can't say.

## The gap it fills

Today's stack pretends three problems are different:

1. **Structured output** — Pydantic / JSON Schema / `response_format`. Type-shaped. Can't express "a prime whose digits sum to 10."
2. **Tool calling** — a separate API surface (`tools=[...]`), a separate response shape, a separate dispatch path.
3. **Agent control flow** — you write a `while True: result = call_model(); if ...: break` loop in Python yourself.

None of them admit *value-level* constraints. JSON Schema can say "int between 0 and 99"; it cannot say "prime whose digits sum to 10." None of them compose: you can't have a model emit a structured object that *contains* a tool call mid-stream, on the same KV. And the agent loop runs in user code, blind to the model's logits — so even a structurally illegal step can only be caught after the fact, by a validator, with a retry that pays the prefix cost again.

orate folds the three problems into one mechanism: a generator whose yields are decision points, evaluated against grammar-constrained decoding at the logit-mask level on a single persistent KV. The same `yield` is structured output, tool call, sub-agent handoff. The same `where=` predicate gates all of them.

This is why orate is local-first by design, not by accident. Operating at the logit-mask level — narrowing the model's next-token distribution *before* the sampler runs — requires controlling the inference stack (llama-cpp + XGrammar here). API-backed models can ship JSON mode and tool schemas, but their constraints fire *after* a token has been emitted; the only recovery is retry-and-validate. That's the shape of solution orate exists to replace: every retry pays the prefix cost again, every extra model call is a place the constraint silently softened, and a `where=` predicate that's a closure over earlier yields can't even be expressed as a re-prompt. Local inference is the cost of paying once and being right.

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

This is demoed end-to-end on Qwen2.5-7B-Instruct local in [`examples/legal_steps/act4_factorize.py`](../examples/legal_steps/act4_factorize.py): the model is handed *factorize 1147 into p × q with p, q > 1*, told the predicate library it can compose under `where=`, and emits `@make_new_program("factor_1147", ...)`. Under `PROGRAM_SOURCE_GRAMMAR` it then writes a six-line body whose `where=divides(n)` and `where=multiplies_to(n, p)` clauses *guarantee* the output. The runtime compiles + registers + rebuilds the outer grammar, the model invokes `@factor_1147(1147, 31, 37)`, and 31 × 37 = 1147 is the answer the predicate gate forced — not a guess.

The set of predicates the model can compose inside an authored `where=` lives in [`src/orate/meta_predicates.py`](../src/orate/meta_predicates.py): `is_prime`, `divides`, `multiplies_to`, `sums_to`, `divisible_by`, `is_square`, `is_palindrome`, `coprime_with`, `length_eq`, `digit_sum_eq`, `lt`, `gt` — all curried so the model writes `where=divides(n)` and the runtime supplies the candidate. The grammar is split by predicate arity (`pred-zero | pred-one | pred-two`) so the model physically can't emit `divides()` empty.

## Per-leaf grammars + transition-based composition

The architectural cleanup that landed during the hackathon and now lives on `main`. Each leaf `@program` carries its own self-contained GBNF (call-site grammar derived once, at registration). The "outer grammar" the engine samples under is just a prefix-alternation `text_chunk | "@a(" | "@b(" | ...` — no leaf bodies inlined. When the matcher accepts on a call-prefix `"@a("`, the runtime calls `engine.sample_under(leaf_a_body_grammar)` — a *fresh sample call* against `a`'s body grammar — then appends the literal `")"` to the KV and returns to outer sampling.

Same shape `make_new_program` already used; we generalised. The implications matter: adding a leaf only recompiles its own body grammar; mode switches and `make_new_program` mutations are cheap; the matcher state can never leak across leaves; per-leaf rule conflicts (helper rules with the same name) become impossible because helpers live in their own namespace. A compiled-grammar cache keyed by GBNF source plus per-leaf body-grammar pre-warming during `Session.__init__` mean the first sample no longer pays a cold-compile cost — was ~200s on the algebra bench, now <5s.

ADR: [`docs/design/two-tier-and-transitions.md`](design/two-tier-and-transitions.md).

## Two tiers of @program

The split that drops out of the cleanup:

- **Leaves** (`invocable=True`, default) — straight-line yields, derivable call-site grammar. Embeddable in a parent's grammar as a named sub-rule. Run via `.run(engine=...)` or, in session mode, invoked by the model emitting `@name(args)`.
- **Composers** (`invocable=False`) — arbitrary Python control flow (loops, branches, state), no derivable call-site grammar. Run directly via `.run(engine=...)`. Orchestrate leaves.

Both are `@program`. The split exists because some programs are *fully grammar-describable* — the matcher can run the whole emission in one shot — and some need Python in the loop between yields. The `gen.alternative([leaves])` primitive (now on `main`) lets composers expose a runtime alternation over leaves to the model: the model picks which leaf to run next, the matcher dispatches, the composer keeps looping. This is how you write an agent loop without a `Session` class — see [`examples/legal_steps/act4_algebra_composer.py`](../examples/legal_steps/act4_algebra_composer.py). Same algebra-solve trace as the Session version. No driver framework. Just a `@program(invocable=False)` with `while True: yield gen.alternative([algebra_step, done])`.

## Where this is going

JIT grammar segmentation. The architectural endpoint that collapses the leaf/composer distinction. Walk a program body, identify segments of yields whose grammars are independent of prior yielded values, fuse each segment into one grammar, recompile only at boundaries where Python state actually matters. The compiler decides how many grammar segments to emit; the runtime executes them. Some programs end up as one segment (today's "leaf"), some as N segments interleaved with Python (today's "composer"), and most are somewhere in between. The user writes the same `@program` either way.

Vision doc: [`docs/design/jit-grammar-segmentation.md`](design/jit-grammar-segmentation.md).

## The benchmark

A free-text vs constrained Qwen2.5-7B-Instruct-Q4_K_M head-to-head on **10 algebra problems** (7 originals + 3 added to round to a clean ten), deterministic argmax decoding, same model, same prompt material.

- **Free text**: 5/10 correct in 309s. Plausible-looking arithmetic that's just wrong: `x=4` on `3x + 5 = 14`, `x=12` on `2x + 3 = x + 9`, `x=39` on `5 - 2x = 1`, `x=5` on `7x - 5 = 16`, `x=4` on `4x + 7 = 2x + 13`.
- **Constrained `@algebra_step`**: **9/10 correct in 127s** (~2.4× faster wall-clock than free text, on a *constrained* path), with **16 predicate-rejected steps caught en route**. The single miss is `eq_negative` (`5 - 2x = 1`): on T=0 Qwen2.5-7B locks into `x = -2`; the Session-level temperature escalation breaks the lock and the model emits valid steps under T>0, but it meanders through 15 of them without committing to `@done` before hitting `max_calls`.
- **Throughput**: free-text 12–20 tok/s; constrained 7–10 tok/s steady-state. The first sample no longer pays a cold-compile cost — was ~200s on an early run, now <5s thanks to the compiled-grammar cache + per-leaf body-grammar pre-warming.

Source: [`bench/results/legal_steps_2026-04-26_1759.md`](../bench/results/legal_steps_2026-04-26_1759.md).

The take: the predicate isn't "trying harder" or "thinking step-by-step." It's a gate the model can't pass without producing a step that is, in fact, mathematically valid under SymPy. The 16 rejections are the gate firing. The model proposes; the predicate disposes; the predicate is the law. The honest framing on the single miss: **step correctness doesn't guarantee a solution** — orate's `where=` predicates verify each step, not the global trajectory.

Decoding default is argmax (T=0). The Session escalates body-sample temperature on consecutive predicate rejections (schedule `0.0 → 0.5 → 1.0 → 1.5 → 2.0`, reset on any successful dispatch); the eq_negative case is precisely why — at T=0 Qwen2.5-7B cannot break out of the wrong attractor without help, and this gives it room to try a different sample without abandoning the run.

## Four end-to-end demos on Qwen2.5-7B-Instruct local

- **[`examples/legal_steps/act4_algebra_demo.py`](../examples/legal_steps/act4_algebra_demo.py)** — the model solves a linear equation by emitting a sequence of `@algebra_step("before", rule, "after")` calls. Each call's `after` is verified equivalent to `before` under `rule` via SymPy. Terminates with `@done(3)` (typed integer, no regex parse).
- **[`examples/legal_steps/act4_logic_demo.py`](../examples/legal_steps/act4_logic_demo.py)** — the model proves `A → B; B → C; A ⊢ C` by emitting `@inference_step(premises, rule, conclusion)` calls, each verified `derivable_under` the named rule (modus ponens, hypothetical syllogism, …). Terminates with `@qed`.
- **[`examples/d20/act3_full_demo.py`](../examples/d20/act3_full_demo.py)** — narrative tools (`@narrate` / `@roll` / `@meta`) under one outer grammar, then `@enter_combat` reshapes the grammar atomically on the same KV; three composed NPC `@program`s (Aria, Borin, the hooded figure — each with its own action set, target list, and damage cap) take turns; `@exit_combat` flips back. One inference, two grammar swaps, no API hops. The video's Page 4.
- **[`examples/legal_steps/act4_factorize.py`](../examples/legal_steps/act4_factorize.py)** — the meta-authorship finisher. The session is seeded with only `@done` and `@make_new_program`; the model authors a `factor_1147` program with `where=divides(n)` + `where=multiplies_to(n, p)` clauses, the runtime registers it, the model invokes it, and the predicate gate forces 31 × 37 = 1147. The video's Page 5.

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

**Models authoring their own tools mid-inference.** [`examples/legal_steps/act4_factorize.py`](../examples/legal_steps/act4_factorize.py). Qwen2.5-7B is given factorize-1147 with no factoring tool; it emits `@make_new_program("factor_1147", ...)`, the runtime grammar-switches to `PROGRAM_SOURCE_GRAMMAR`, the model writes a six-line body with `where=divides(n)` and `where=multiplies_to(n, p)` clauses composed from the meta-predicate library, the runtime compiles + registers + rebuilds the outer grammar, the same model invokes `@factor_1147(1147, 31, 37)` and the predicate gate forces correctness. Same KV from start to finish.

**Provably-legal multi-step reasoning.** The algebra benchmark above. [`examples/legal_steps/checkers.py`](../examples/legal_steps/checkers.py) is SymPy doing equivalence under named rules. The grammar makes "produce a step" the only available action; the predicate makes "produce a step that's actually a legal transformation" the only acceptable action. Sixteen model proposals failed the predicate across the 10-problem benchmark; none of them reached the user.

## Where you'd push back

A few honest places:

1. **Predicate verification on closures isn't free.** When the third yield's `where=` closes over the first two, witness enumeration can't fire and the runtime falls back to syntactic-grammar sampling + post-hoc closure check. The 16 rejections in the 10-problem benchmark are real tokens spent. We have a story for this — JIT grammar segmentation should compile cross-yield predicates into a fused grammar where possible — but today the rejection loop is the honest fallback.
2. **`PROGRAM_SOURCE_GRAMMAR` is a small Python subset.** Straight-line `var = yield gen.method(...)` then `return`. No branches. No loops. No `yield from`. The model can author tools but only of the leaf shape. Composers it doesn't write yet. This is fine for the four-act story; it'll need to grow for richer self-authoring.
3. **Local-only scope.** The logit-level contract requires a controlled inference stack (llama-cpp + XGrammar here) — see the intent section above for *why*. If you want grammar-bound decoding against Claude or GPT, you'd be writing a different library; the structural shape is not the same problem.

## Repo layout, in one breath

[`src/orate/gen.py`](../src/orate/gen.py) holds the Gen primitives. [`src/orate/program.py`](../src/orate/program.py) is the decorator. [`src/orate/session.py`](../src/orate/session.py) is the persistent-KV runner. [`src/orate/meta.py`](../src/orate/meta.py) is the program-source grammar + sandbox compiler. [`src/orate/body_grammar.py`](../src/orate/body_grammar.py) is the AST → GBNF derivation. [`src/orate/engine/xgrammar.py`](../src/orate/engine/xgrammar.py) is the local engine. Demos in [`examples/legal_steps/`](../examples/legal_steps/) and [`examples/d20/`](../examples/d20/). Benchmarks in [`bench/results/`](../bench/results/). Design docs in [`docs/design/`](design/).

Status: pre-alpha, hackathon birth (Built with Opus 4.7, Apr 2026), in active development. The primitives work end-to-end on Qwen2.5 instruct GGUFs from 0.5B through 7B; all benchmark + demo numbers in this doc come from **Qwen2.5-7B-Instruct-Q4_K_M** specifically. The interesting questions are upstream: how much of a `@program`'s body can the compiler statically grammar-fuse, and where does the rejection-with-closure path stop being the honest fallback and start being the bottleneck.
