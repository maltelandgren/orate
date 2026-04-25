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
- **Right:** the active grammar — narrative `narrate | roll | enter_combat`, then combat `aria_attack | borin_attack | hooded_figure_attack | exit_combat`
- **Bottom:** KV tokens accumulating, never reset

Sequence (every emission is a real @-call — no free text, no prose
prediction):

1. **0:55** — `@narrate("…")` × 1. The model writes one short sentence under a string grammar.
2. **1:00** — `@roll(perception, 13)`. Client-resolved tool call: the runtime rolls a d20 server-side, injects `→ {"d20": 17, "success": true}` into the KV, the model continues.
3. **1:10** — `@narrate("…")` reacting to the roll result.
4. **1:20** — `@enter_combat(hooded_figure)`. **Mode transition.** The outer grammar swaps from narrative tools to combat tools, atomically, on the same KV.
5. **1:30** — Round robin under combat-mode grammar: `@hooded_figure_attack(...)`, `@aria_attack(...)`, `@borin_attack(...)`. Each one's body grammar is derived from that character's @program (action set, target list, damage cap). Brief cutaway: flash the three program definitions side-by-side — *"this is all it takes."*
6. **1:45** — `@exit_combat(victory)`. Mode flips back. `@narrate("…")` closes the scene.

One voiceover line, placed around 1:40:

> One KV from start to finish. One mode switch, three composed-per-character programs, every token grammar-bound. The whole scene is one inference.

Visual emphasis: when the per-character programs flash on screen,
make their tininess the point. ~6 lines each. The grammar that binds
the model's combat tokens *was the model's stat sheet a moment ago.*

Demo runner: [`examples/d20/act3_full_demo.py`](../examples/d20/act3_full_demo.py).

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

Show free-text run (verbatim from `bench/results/legal_steps_2026-04-25_1200.json`'s `eq_3x_plus_5` row). Model writes a chain of arithmetic, ends with `ANSWER: x = 4`. Red overlay on the answer.

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

> Across a 7-problem benchmark, free-text Qwen-7B got 4 of 7. The same
> model under `@algebra_step` got 6 of 7, with eleven illegal-step
> attempts caught and rejected en route. Same weights. Different gate.

(Source: [`bench/results/legal_steps_2026-04-25_1200.md`](../bench/results/legal_steps_2026-04-25_1200.md). Decoding is deterministic argmax — these aren't sampling-variance numbers.)

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

### Beat 3 — the finisher (2:40–2:55)

The climax. Hard cut. New problem on screen:

```
Solve: x² - 5x + 6 = 0
```

Voiceover:

> The pre-registered `@algebra_step` doesn't fit a quadratic — its
> rules are linear-equation moves. Watch the model decide.

Model emits — *on the same KV*:

```
@make_new_program("quadratic_solve",
  "find both roots of a quadratic in standard form")
```

**Grammar switches.** Source materialises on screen, sampled token by
token under `PROGRAM_SOURCE_GRAMMAR`:

```python
@program
def quadratic_solve():
    equation = yield gen.string(max_len=40)
    a = yield gen.integer(-20, 20)
    b = yield gen.integer(-20, 20)
    c = yield gen.integer(-20, 20)
    root1 = yield gen.integer(-20, 20)
    root2 = yield gen.integer(-20, 20)
    return {"equation": equation, "a": a, "b": b, "c": c,
            "roots": [root1, root2]}
```

Validates. AST-checks. Sandbox-execs. Registers. Outer grammar rebuilds
to include `@quadratic_solve(`.

> The model just designed its own data type. The schema's structure is
> now grammar-bound — every future emission of `@quadratic_solve(...)`
> must fit it.

Model uses what it just authored:

```
@quadratic_solve("x^2 - 5x + 6 = 0", 1, -5, 6, 2, 3)
@done("x = 2 or x = 3")
```

Verify: 2² - 5·2 + 6 = 0 ✓, 3² - 5·3 + 6 = 0 ✓.

Pull back to a registry panel showing what grew across the video:
`narrate`, `roll`, `enter_combat`, `aria_attack`, `borin_attack`,
`hooded_figure_attack`, `exit_combat`, `algebra_step`, `done`,
`inference_step`, `qed`, `quadratic_solve`. The last one wasn't there
when the video started.

> Instruction-following is soft pressure on a probability distribution.
> This is hard constraint — schema with logic — authored mid-inference,
> binding forward, composable. A new primitive for reasoning under
> guarantee.

Demo runner: [`examples/legal_steps/act4_meta_finisher.py`](../examples/legal_steps/act4_meta_finisher.py).

> ⚠️ Honest scope. Today the model-authored body is a *typed schema* —
> `gen.choice / integer / string / boolean` with no `where=` clause.
> The grammar enforces type structure but not the math. The
> hand-authored leaves (`algebra_step`, `inference_step`) carry the
> SymPy predicates; meta-authored leaves don't yet. Extending
> `PROGRAM_SOURCE_GRAMMAR` with a `where=<lib_predicate>` form is on
> the JIT segmentation roadmap. For the video this distinction stays
> off-mic — but it's documented here so we don't oversell.

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
| Free-text vs constrained 7-problem benchmark | shipped (4/7 vs 6/7, 11 rejections caught) |
| Composer demos for algebra + logic (5-line agent loop) | shipped |
| Act 3 full demo: narrate + roll + combat | shipped |
| Act 4 meta-authorship finisher (typed schema only) | shipped |
| Visualization overlay (grammar stack, registry panel, KV bar) | not shipped — manim work |
| Predicate-bound model-authored programs (`where=` in the meta grammar) | **NOT shipped** — see Beat 3 honest-scope footnote |

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
