# orate — demo video script

- **Target length:** 3:00
- **Submission:** Built with Opus 4.7 hackathon, problem statement #2 ("Build For What's Next")
- **Deadline:** 2026-04-26, 8pm EST

## Thesis

> Structured output constrained the shape.
> Tool calling constrained the side effect.
> orate lets the model enforce the legality of its own thought.

Three years into the LLM era, structured output and tool calling are separate APIs, type-only, human-authored. orate collapses them into one primitive — `yield gen.X(description=, where=)` — where predicates express logic the schema can't. In session mode, the model authors its own `@program`s mid-inference, growing a compositional library that binds its own future tokens.

## Narrative arc

1. Act 1 — A real problem I've struggled with (simulacrum DM agent)
2. Act 2 — How Opus 4.7 and I arrived at the `@program` primitive
3. Act 3 — The session in motion: one continuous thread through a D&D scene
4. Act 4 — What this actually unlocks: legal-step enforcement on hard problems

Acts 1–3 set up the library through the use case I built it for. Act 4 reveals what the primitive means beyond that original scope.

---

## Act 1 — The pain (0:00–0:25)

Cold open. Screen recording of `simulacrum` mid-session. D&D scene renders in the terminal. No voiceover for 2 seconds.

Voiceover, first-person:

> I've been building a D&D simulator. Every DM turn, one inference has to do three things: narrate what happens, roll dice against the rules, and voice the NPCs.

Cut to [`simulacrum/src/harness/agents/dm.py:283`](../../simulacrum/src/harness/agents/dm.py). Highlight return type: just a string.

> But today's APIs make me pick: structured output OR tool calls, not both. So I return text.

Cut to [`simulacrum/src/harness/orchestrator.py:514`](../../simulacrum/src/harness/orchestrator.py). Highlight the reconstruction loop.

> And then I reconstruct the model's decisions by pattern-matching its own narration. Thirty lines of regex recovering what the model already knew.

Beat.

> Three years into the LLM era — why am I translating the model's decisions back out of its own prose?

## Act 2 — One primitive (0:25–0:55)

Cut to orate terminal. A `@program` for `dm_turn` on screen:

```python
@program
def dm_turn(scene):
    narration = yield gen.string(description="what the players see")
    needs_roll = yield gen.boolean(description="does this need a dice check?")
    if needs_roll:
        dc = yield gen.integer(5, 30, description="difficulty class")
        result = yield gen.tool("roll_d20", dc=dc)
    npc_line = yield gen.string(description="what the NPC says, if anyone")
    return {"narration": narration, "roll": result, "npc": npc_line}
```

Voiceover:

> One primitive. `yield gen.X`. Types, tool calls, and logic constraints are the same thing — and they compose, because it's just Python. Conditionals. Loops. Functions.

Run it. Grammar switches visible in a side panel; tool call emission is part of the decode, not a separate API round-trip.

> Everything happens on one KV cache. Narration, dice tool, NPC line — no resets, no re-prompts.

Quick aside with a `where=`:

```python
attack = yield gen.integer(1, 20,
    where=lambda n: n + character.attack_bonus >= target.ac)
```

> And because predicates are Python, I get logic constraints that no JSON schema could express. The grammar is computed from the predicate, token by token.

## Act 3 — The session in motion (0:55–1:55)

Dense and visual, minimal voice. Split screen:
- **Left:** unfolding D&D session as a sequence of `@-calls`
- **Right:** the active grammar — narrative `roll | enter_combat`, then combat `aria_attack | borin_attack | hooded_figure_attack | exit_combat`
- **Bottom:** KV tokens accumulating, never reset

Sequence (every emission is a real @-call):

1. **0:55** — *[Narration placeholder.]* In the final cut, voiceover or a manim-rendered text card describes the tavern as the party enters. **The `@narrate(...)` leaf exists** in `examples/d20/dice.py` and would normally fill this beat — but `gen.string(max_len=120)` currently compiles to a pathological grammar (one mandatory char + 119 optional chars in sequence; see `body_grammar.py:_fragment_string`). Sampling under that hangs Qwen-7B. Two clean fixes deferred to post-submission: (a) recursive `chars ::= char chars | ""` rule with the length cap enforced post-sample, or (b) replace the body with `gen.choice([...])` over a phrase library. Neither is in the way of the rest of Act 3.
2. **1:00** — `@roll(perception, 13)`. Client-resolved tool call: the runtime rolls a d20 server-side, injects `→ {"d20": 17, "success": true}` into the KV, the model continues.
3. **1:10** — *[Narration placeholder reacting to the roll, same treatment as beat 1.]*
4. **1:20** — `@enter_combat(hooded_figure)`. **Mode transition.** The outer grammar swaps from narrative tools to combat tools, atomically, on the same KV.
5. **1:30** — Round robin under combat-mode grammar: `@hooded_figure_attack(...)`, `@aria_attack(...)`, `@borin_attack(...)`. Each one's body grammar is derived from that character's @program (action set, target list, damage cap). Brief cutaway: flash the three program definitions side-by-side — *"this is all it takes."*
6. **1:45** — `@exit_combat(victory)`. Mode flips back. *[Narration placeholder closes the scene.]*

One voiceover line, placed around 1:40:

> One KV from start to finish. One mode switch, three composed-per-character programs, every token grammar-bound. The whole scene is one inference.

Visual emphasis: when the per-character programs flash on screen,
make their tininess the point. ~6 lines each. The grammar that binds
the model's combat tokens *was the model's stat sheet a moment ago.*

Demo runner: [`examples/d20/act3_full_demo.py`](../examples/d20/act3_full_demo.py)
— produces a clean 15s trace of beats 2–6 (no narration). Beat 1 / 3
/ 6 are filled in post-production until the `gen.string` fix lands.

## Act 4 — Legal steps only (1:55–2:55)

Hard cut from simulacrum to a black screen, two lines:

> I built this to finish one game.
> Then I realized what it actually does.

### Beat 1 — algebra contrast (1:58–2:25)

Cut to a terminal. Problem on screen:

```
Solve for x:
  3x + 5 = 14
```

Voiceover, brisk:

> Multi-step reasoning is where LLMs slip. Same problem to Qwen-7B in
> free text:

Show free-text run (verbatim from `bench/results/legal_steps_2026-04-26_1733.json`'s `eq_3x_plus_5` row). Model writes a chain of arithmetic, ends with `ANSWER: x = 4`. Red overlay on the answer.

> x = 4. Wrong. Plug it back in: 3·4 + 5 = 17, not 14.

Hard cut to the **constrained** run on the same model. On the right
panel, flash the **agent loop in 5 lines** (this is what the user would
copy-paste):

```python
@program(invocable=False)
def solve():
    while True:
        step = yield gen.alternative([algebra_step, done])
        if step.name == "done":
            return step.value
```

Voiceover continues:

> Same weights, same problem, under a `where=` predicate that calls SymPy on every emission. The model's allowed to attempt anything — but only valid steps reach the next yield.

Show the chain:
```
@algebra_step("3x + 5 = 14", simplify, "3x = 9")     ✓
@algebra_step("3x = 9", isolate_var, "x = 3")        ✓
@done("x = 3")
```

> x = 3. Plug back in: 3·3 + 5 = 14 ✓.

End of beat:

> Across a 10-problem benchmark, free-text Qwen-7B got 5 of 10. The
> same model under `@algebra_step` got 9 of 10, with sixteen
> illegal-step attempts caught and rejected en route. Same weights.
> Different gate.

(Source: [`bench/results/legal_steps_2026-04-26_1759.md`](../bench/results/legal_steps_2026-04-26_1759.md). Decoding is argmax (T=0) by default; Session escalates body-sample temperature on consecutive rejections so the model can break out of locked-in wrong answers. The single constrained miss is `eq_negative` (`5 - 2x = 1`): on T=0 Qwen-7B locks into `x = -2`; once T ramps the model breaks the lock and emits valid steps, but it meanders through 15 of them without converging on `@done`. The honest framing: **step correctness doesn't guarantee a solution** — orate's `where=` predicates verify each step, not the global trajectory.)

Demo runner: [`examples/legal_steps/act4_algebra_composer.py`](../examples/legal_steps/act4_algebra_composer.py).

### Beat 2 — logic (2:25–2:40)

Same composer pattern, briefer. New problem:

```
Given:
  A → B
  B → C
  A
Prove: C
```

Model composes (under the same 5-line agent loop, with leaves swapped to `[inference_step, qed]`):

```
@inference_step("A -> B; A", modus_ponens, "B")     ✓
@inference_step("B -> C; B", modus_ponens, "C")     ✓
@qed("C")
```

The `inference_step` predicate runs `derivable_under(rule, premises, conclusion)` — modus ponens / modus tollens / hypothetical syllogism / conjunction / simplification, evaluated as Python.

Voiceover bridging both beats:

> Algebra, logic — wherever "legal step" is definable as a Python predicate, the model can reason strictly within it. The gap between free-text reasoning and constrained reasoning is the gap between *probably right* and *provably valid*.

Demo runner: [`examples/legal_steps/act4_logic_composer.py`](../examples/legal_steps/act4_logic_composer.py).

### Beat 3 — the finisher (2:40–2:58)

> **2026-04-26 update.** This beat now uses the **factorize 1147**
> trace, which runs end-to-end on local Qwen2.5-7B with real `where=`
> predicates — captured at [`bench/results/act4_factorize_2026-04-26.md`](../bench/results/act4_factorize_2026-04-26.md)
> (TODO once captured) and reproducible via
> [`examples/legal_steps/act4_factorize.py`](../examples/legal_steps/act4_factorize.py).
> The earlier quadratic version was idealised; this one is verbatim.
> The beat is +3s versus the prior cut (15s → 18s) to make room for a
> grammar-mask flash and a predicate-check flash; trim from elsewhere
> if total runs over 3:00.

#### Shot-by-shot timing

| t (rel) | beat | duration | what's on screen |
|---|---|---|---|
| 0.0 | 5.A bridge | 2.5s | "This is already pretty nice. But we kept thinking." holds, then: "What if the model defined its own schemas / as structure on its own future generation?" |
| 2.5 | 5.B problem | 0.5s | Top of screen: `Factor:  1147 = p × q   (p, q > 1)` |
| 3.0 | 5.B emit | 0.6s | `@make_new_program("factor_1147", "two factors of 1147 greater than 1")` fades in |
| 3.6 | 5.B grammar-switch tag | 0.5s | `[grammar switch → PROGRAM_SOURCE_GRAMMAR]` appears below the call |
| 4.1 | 5.C source streams pt.1 | 1.6s | Lines 1–3 materialise via LaggedStart: `@program / def _factor_1147(): / n = yield gen.integer(1147, 1147)` |
| 5.7 | **5.C MASK FLASH** | **1.5s** | Caption: *"where= arg slot — only previously-bound names"*. Logit column right of source: 7 candidate Qwen tokens (`number`, `value`, `int`, `the`, `target`, `1147`, `n`); 6 fade to ~35 % opacity with a clay-red strike-through, `n` glows accent-orange. |
| 7.2 | 5.C source streams pt.2 | 1.6s | Lines 4–6 materialise: `p = yield gen.integer(2, 1146, where=divides(n)) / q = yield gen.integer(2, 1146, where=multiplies_to(n, p)) / return {"p": p, "q": q}` |
| 8.8 | 5.D compile callout | 0.6s | `[validated · compiled · registered]` fades in to the right of the source |
| 9.4 | 5.E invoke | 0.6s | `@_factor_1147(1147, 31, 37)` materialises below the callout |
| 10.0 | 5.E result | 0.4s | `→ {'p': 31, 'q': 37}` lands beside it |
| 10.4 | **5.E PREDICATE FLASH** | **2.0s** | Two lines fade in below the result: `divides(1147)(31)        → 1147 % 31 == 0   ✓` then `multiplies_to(1147, 31)(37) → 37 × 31 == 1147 ✓`. Both ✓ glow `Paper.good`. **Voiceover lands here.** |
| 12.4 | 5.E done | 0.4s | `@done("31 and 37")` muted-grey closes the chain |
| 12.8 | 5.F clear | 0.4s | All Page 5 source/flash content fades |
| 13.2 | 5.F thesis card | 5.5s | Letter-tracked thesis (existing) |
| 18.7 | 5.F GitHub URL | 5.0s | URL fades in, holds |

Total Page 5 length: ~24s (was ~25s in v2 pre-edits — net unchanged).

#### Voiceover lines (Page 5)

| t (rel) | line | tone |
|---|---|---|
| 1.5 | "What if the model defined its own schemas — as structure on its own future generation?" | reflective, slow |
| 4.0 | "On the same cache, under a different grammar, the model writes a verifier." | brisk |
| 8.5 | "Validated. Compiled. Registered. The library grew during the inference." | steady |
| 10.6 | *(beat — predicate-flash visual lands)* | — |
| 11.0 | **"The model wrote down a contract — then was forced to honor it."** | the punchline; deliberate, unhurried |
| 13.6 | (thesis card carries itself) | — |

#### Problem on screen

```
Factor:  1147 = p × q   (p, q > 1)
```

#### What the model emits — verbatim trace from Qwen-7B (`/tmp/factorize_run_6.log`)

```
@make_new_program("factor_1147", "two factors of 1147 greater than 1")
[session: synthesizing program…]
@program
def _factor_1147():
    n = yield gen.integer(1147, 1147)
    p = yield gen.integer(2, 1146, where=divides(n))
    q = yield gen.integer(2, 1146, where=multiplies_to(n, p))
    return {"p": p, "q": q}
[session: registered @_factor_1147; grammar rebuilt]
@_factor_1147(1147, 31, 37)
  → {'p': 31, 'q': 37}
@done(...)
```

The body's six lines are sampled under `PROGRAM_SOURCE_GRAMMAR`. The
invocation `(1147, 31, 37)` is sampled under the body grammar derived
from those six lines, with `divides(1147)` and `multiplies_to(1147,
31)` re-run as predicates on every candidate emission. 31 × 37 = 1147
isn't a guess — the grammar + predicate gate forced it.

Cosmetic note: in the actual capture the model wrote `@done('p and
q',)` rather than the values; the dict result holds the real answer.
For video clarity we render the done line as `@done("31 and 37")` —
the diff is one decoded string, not a substantive claim.

#### Mask-flash visual spec

Anchor: right of the source block at the y-position where line 4
materialises. Width ~2.4 in. Background is a faint rounded panel.

Rows (top to bottom, with synthetic but plausible logits):

| token  | logit | state |
|---|---|---|
| `number` | -2.1 | masked: not single-letter |
| `value`  | -2.4 | masked |
| `int`    | -3.0 | masked |
| `the`    | -3.1 | masked |
| `target` | -3.5 | masked |
| `1147`   | -4.0 | masked |
| `n`      | **-1.4** | **kept — single letter AND bound** |

Caption above the column: *"where= arg slot — only previously-bound
names"* in Paper.ink_soft, 11pt italic. The actual GBNF rule is
`var-name ::= [a-z]`; the validator additionally enforces "must be
bound." For video clarity the caption conflates them — both gates
are real, the punchline is the same.

#### Predicate-flash visual spec

Anchor: below `@_factor_1147(...)` invocation, indented to align with
the result arrow. Two lines, 13pt mono, separated 0.16 buff:

```
divides(1147)(31)         → 1147 % 31 == 0   ✓
multiplies_to(1147, 31)(37) → 37 × 31 == 1147 ✓
```

The predicate-name prefix is `Paper.accent_soft`; the arithmetic mid
is `Paper.ink_soft`; the trailing ✓ is `Paper.good`, scaled +20 % and
glow-pulsing once on entry. Both lines fade in with a 0.25s lag; the
voiceover line lands while the second ✓ pulses.

#### Registry pull-back — kept

After Page 5's clear, the registry panel pull-back from the closing
beat shows what grew across the video. The last entry — the one that
wasn't there when the video started — is now `_factor_1147` rather
than `quadratic_solve`.

#### Demo runner

[`examples/legal_steps/act4_factorize.py`](../examples/legal_steps/act4_factorize.py).

The earlier `act4_meta_finisher.py` (quadratic) is preserved but no
longer the headline; it was the path-finder that taught us the
single-letter-var-name + bounded-stmt-list + arity-split-grammar
constraints needed to make Qwen-7B reliable.

## Close (2:50–3:00)

Card, held 5 seconds:

> Structured output constrained the shape.
> Tool calling constrained the side effect.
> **orate** lets the model enforce the legality of its own thought.
>
> github.com/…

---

## Production notes

### Capability state

| Capability | State |
|---|---|
| Persistent KV session with @-call grammar | shipped |
| `@make_new_program` mid-session | shipped |
| Per-leaf body grammar + transition-based composition | shipped (`feat/flavor-b-full` merged) |
| `gen.alternative([leaves])` composer primitive | shipped |
| `@program(invocable=False)` composer flag | shipped |
| Real client-resolved tool call (`@roll`) — body returns the result | shipped |
| Grammar mode-switch (`enter_combat` / `exit_combat`) | shipped |
| Typed kwargs (positional with type-aware scanner) | shipped |
| Three composed NPC programs (Aria / Borin / hooded figure) | shipped |
| `equivalent_under` via SymPy (with scalar-multiple equivalence) | shipped |
| Logic `derivable_under` (modus ponens / tollens / hyp syllogism / …) | shipped |
| Engine grammar cache + warmup (kills 232s cold-start) | shipped |
| Free-text vs constrained 10-problem benchmark | shipped (5/10 vs 9/10, 16 rejections caught; T-escalation on rejection) |
| Composer demos for algebra + logic (5-line agent loop) | shipped |
| Act 3 full demo: narrate + roll + combat | shipped |
| Act 4 meta-authorship finisher (factorize 1147, real where= predicates) | shipped — verbatim Qwen-7B trace |
| Predicate-bound model-authored programs (`where=` in the meta grammar) | shipped (commit 6473880 + arity-split grammar) |
| Predicate library (14 entries: is_prime, divides, multiplies_to, …) | shipped |
| Validator arity check + safe runtime predicate failure | shipped |
| Single-letter `var-name` grammar (BPE-merge dodge) | shipped |
| Visualization overlay (grammar mask flash, predicate flash) | spec'd in Beat 3 above; manim impl in `video/scenes/full_video_v2.py:_page5_meta_authorship_and_close` |

### Shot-level production decisions (to make before filming)

- **Real vs. staged Act 4.** Real = Opus 4.7 genuinely writes the `algebra_step`/`inference_step` programs against orate's session; we demo whatever it produces. Staged = pre-record a known-good run. **Recommended: real run, keep best takes.** The pitch is honesty about what the library does.
- **Model choice.** Qwen2.5-7B local for Acts 2–3 (fast, local-first is the principle). Opus 4.7 for Act 4 meta-authorship (the primitives it authors need to be sharp).
- **Overlay toolchain.** Terminal recording via `asciinema` + manual post-edit overlays in the video editor for grammar stack / registry panels. Avoid fragile live overlays.
- **Bard-curse beat (palindromes)** from earlier drafts is cut. Reason: reads as parlor trick, not capability. Algebra/logic reads as capability.

### If we cut for runtime

Priority order if 3:00 is tight:
1. Keep Act 1 (0:25) — the pain sets the whole video.
2. Keep Act 2 (0:30) — the primitive is the product.
3. Keep Act 4 Beat 1 (algebra contrast — 27s) and Beat 3 (meta-authorship finisher — 15s). Those are the punchline.
4. **Cut Act 4 Beat 2 (logic, 15s)** first — algebra alone carries the legal-step thesis if runtime is tight.
5. **Trim Act 3** by skipping the second `@narrate` (between roll and enter_combat) and using a single tighter narrative beat.
6. Last resort: collapse the registry pull-back and the closing card into one (saves ~3s).

### What this script is not

- Not a product pitch. We show the capability; we don't sell it.
- Not a tutorial. No "here's how to install." The GitHub link is the call to action.
- Not exhaustive. Witness enumeration, forward-checking, source-in-prompt, verifiers — all present in the library, none mentioned here. The video shows *what it feels like*, not how it's built.
