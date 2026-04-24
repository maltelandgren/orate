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
- **Left:** unfolding D&D session transcript
- **Right:** grammar stack indicator showing the active grammar
- **Bottom:** KV cache tokens accumulating, never reset

Sequence:

1. **0:55** — Narration flows. Grammar: `narrative_outer`.
2. **1:05** — Model emits `@roll(perception, DC=15)`. Stack pushes `roll_args`. `ends_turn=True`, client resolves, `19` injects back into the same KV. Stack pops. Narration continues.
3. **1:20** — Model emits `@remember({kind: "npc_introduced", who: "hooded_figure", traits: ["armed", "wary"]})`. Structured block. Client buffers it for next turn's context.
4. **1:30** — `@enter_combat(participants=["Aria", "hooded_figure", "Borin"])`. Grammar stack **replaces** `narrative_outer` with `combat_outer` composed from the three participants' programs.
5. **1:40** — Round robin: `hooded_figure_turn` → `aria_turn` → `borin_turn`. Each under its own grammar, derived from its stat sheet. Cutaway diagram: stat sheet → action program → combat loop.
6. **1:50** — `@exit_combat`. `narrative_outer` restored. The earlier `@remember` block surfaces as context.

One voiceover line, placed around 1:45:

> One KV from start to finish. Six grammar switches, two mode transitions, three composed-per-character programs. One thread.

## Act 4 — Legal steps only (1:55–2:50)

Hard cut from simulacrum to a black screen, two lines:

> I built this to finish one game.
> Then I realized what it actually does.

### Beat 1 — algebra (1:58–2:25)

Cut to a terminal. Problem on screen:

```
Find integers x, y such that:
  2x + 3y = 12
  x + y   = 5
  x > y
```

Voiceover, brisk:

> Multi-step reasoning is where LLMs slip. Free-text model, two runs, same prompt:

Show two parallel runs. Left run correct; right run has a subtle arithmetic error (e.g., `10 - 2y + 3y → 10 + y = 12 → y = 4`). Red overlay on the wrong line.

> One correct, one wrong. Same model, same prompt. The math doesn't constrain the tokens.

Cut to session mode. Model emits:

```
@make_new_program(
  "algebra_step",
  "one legal algebraic transformation: (rule, before, after)
   where after is provably equivalent to before under rule"
)
```

Grammar switches. Source synthesized on screen:

```python
@program
def algebra_step(before):
    rule = yield gen.choice(
        ["substitute", "simplify", "combine_like",
         "isolate_var", "evaluate"],
        description="the algebraic move")
    after = yield gen.string(
        where=lambda s: equivalent_under(rule, before, s),
        description="the result, verified equivalent")
    return {"rule": rule, "before": before, "after": after}
```

Validates, compiles.

> The `where=` predicate is the gate. If the sampled `after` isn't equivalent to `before` under `rule`, it's rejected — token by token. The model cannot emit an illegal step.

Model composes the solve:

```
@algebra_step("x + y = 5")          → isolate_var → "x = 5 - y"
@algebra_step("2(5-y) + 3y = 12")   → simplify   → "10 + y = 12"
@algebra_step("10 + y = 12")        → isolate_var → "y = 2"
@algebra_step("x = 5 - 2")          → evaluate   → "x = 3"
```

Each line appears under its own grammar switch. Green check on each. Final verify: `2(3) + 3(2) = 12 ✓`, `3 + 2 = 5 ✓`, `3 > 2 ✓`.

### Beat 2 — logic (2:25–2:45)

Optional; cut if runtime tight. New problem:

```
Given:
  P → Q
  Q → R
  P
Prove: R
```

Model emits:

```
@make_new_program(
  "inference_step",
  "one legal deduction: (rule, premises, conclusion)
   where conclusion follows from premises under rule"
)
```

Predicate checks that `rule ∈ {modus_ponens, modus_tollens, hypothetical_syllogism, ...}` and that `conclusion` is derivable from `premises` under that rule.

Model composes:

```
@inference_step(["P → Q", "P"])              → modus_ponens → "Q"
@inference_step(["Q → R", "Q"])              → modus_ponens → "R"
```

Green check. QED.

Voiceover bridging both beats:

> Algebra, logic — anywhere "legal step" is definable, the model can author a schema that enforces legality and then reason strictly within it. The gap between free-text reasoning and constrained reasoning is the gap between *probably right* and *provably valid*.

### Beat 3 — the thesis (2:45–2:50)

Pull back. Registry panel shows programs accumulated across the video: `dm_turn`, `hooded_figure_turn`, `palindromic_line` (cut this one if cut earlier), `algebra_step`, `inference_step`. The library grew during the demo.

> Instruction-following is soft pressure on a probability distribution. This is hard constraint — schema with logic — authored mid-inference, binding forward, composable. A new primitive for reasoning under guarantee.

## Close (2:50–3:00)

Card, held 5 seconds:

> Structured output constrained the shape.
> Tool calling constrained the side effect.
> **orate** lets the model enforce the legality of its own thought.
>
> github.com/…

---

## Production notes

### What needs to ship for this demo to be real

| Capability | State | Effort |
|---|---|---|
| Persistent KV session with `@call` grammar | shipped | — |
| `@make_new_program` mid-session | shipped | — |
| `@program` composition (program invokes program) | shipped (Flavor B minimal) | — |
| `@remember({...})` — just `gen.struct`, client-side buffering | shipped | 0 |
| `ends_turn=True` loop (client resolves → inject) | field exists, loop unwired | ~3 hr |
| Grammar mode-switch (replace outer grammar mid-session) | not shipped | ~1–2 hr |
| Typed kwargs (`@algebra_step("x + y = 5")`) | positional-only | ~2 hr |
| 3 composed NPC programs for combat | Python | ~1 hr |
| `equivalent_under(rule, before, after)` via SymPy | not shipped | ~1 hr |
| Logic `derivable_under(rule, premises, conclusion)` | not shipped | ~1 hr |
| Demo `@program`s + problems (algebra + logic) | not shipped | ~1 hr |
| Visualization overlay (grammar stack, registry panel, KV bar) | not shipped | ~3 hr |

**Total:** ~12 hr. Two working days.

### Shot-level production decisions (to make before filming)

- **Real vs. staged Act 4.** Real = Opus 4.7 genuinely writes the `algebra_step`/`inference_step` programs against orate's session; we demo whatever it produces. Staged = pre-record a known-good run. **Recommended: real run, keep best takes.** The pitch is honesty about what the library does.
- **Model choice.** Qwen2.5-7B local for Acts 2–3 (fast, local-first is the principle). Opus 4.7 for Act 4 meta-authorship (the primitives it authors need to be sharp).
- **Overlay toolchain.** Terminal recording via `asciinema` + manual post-edit overlays in the video editor for grammar stack / registry panels. Avoid fragile live overlays.
- **Bard-curse beat (palindromes)** from earlier drafts is cut. Reason: reads as parlor trick, not capability. Algebra/logic reads as capability.

### If we cut for runtime

Priority order if 3:00 is tight:
1. Keep Act 1 (0:25) — the pain sets the whole video.
2. Keep Act 2 (0:30) — the primitive is the product.
3. Trim Act 3 from 60s → 45s by dropping the `@remember` beat (step 3 in the sequence).
4. In Act 4, cut the logic beat (2:25–2:45), keep algebra. The thesis still lands.

### What this script is not

- Not a product pitch. We show the capability; we don't sell it.
- Not a tutorial. No "here's how to install." The GitHub link is the call to action.
- Not exhaustive. Witness enumeration, forward-checking, source-in-prompt, verifiers — all present in the library, none mentioned here. The video shows *what it feels like*, not how it's built.
