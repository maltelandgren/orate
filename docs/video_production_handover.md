# Video production handover

**Read this once, top to bottom. Then read [`video_script.md`](video_script.md). Then start.**

You're producing a 3-minute demo video for the **Built with Opus 4.7** hackathon (problem statement #2, "Build For What's Next"). Submission is due **Sunday 2026-04-26, 8pm EST**. The library being demoed is **orate** at `/Users/maltelandgren/code/Private/sandbox/orate`.

This document tells you everything you need: the thesis, what's built vs. what needs building, the exact visualizations to produce (manim), the terminal recordings to capture (asciinema), and the open decisions to bring back to the user.

---

## 1. The product (read this even if you skim everything else)

orate is a Python library for grammar-constrained LLM decoding. Its core primitive is one yield surface that unifies what today's LLM APIs split:

```python
@program
def plan(topic):
    theme = yield gen.choice(["medieval", "space"], description="the theme")
    guests = yield gen.integer(1, 50, where=lambda g: g > 0, description="guest count")
    return {"theme": theme, "guests": guests}
```

- `description=` — additional instruction space the model sees in-prompt
- `where=` — Python predicate; the grammar is *computed* from the predicate token-by-token via witness enumeration
- `@program` decorators compose like Python functions — conditionals, loops, recursion all work

**Session mode** is the killer feature. A persistent KV cache holds the entire conversation. The outer grammar is `(text_chunk | @call)*`. The model can emit `@make_new_program("name", "description")` mid-decode; the runtime grammar-switches to a Python-subset DSL, the model authors a new `@program`, the runtime validates, compiles, registers it, rebuilds the outer grammar to include the new tool, and decoding continues — same KV, no resets.

Net: **the model authors schemas with logic that bind its own future tokens, mid-inference, on a persistent KV.**

Thesis line we're shipping with:
> Structured output constrained the shape. Tool calling constrained the side effect. orate lets the model enforce the legality of its own thought.

---

## 2. Repo layout

```
/Users/maltelandgren/code/Private/sandbox/orate/
├── docs/
│   ├── video_script.md              ← THE SHOT LIST. Read after this doc.
│   ├── video_production_handover.md ← this file
│   └── design/                       ← Flavor-B and other ADRs
├── examples/
│   ├── smoke_session.py              ← the existing Act-4 demo (Qwen-7B)
│   ├── smoke_local.py                ← Qwen-0.5B prime+digitsum10
│   ├── smoke_meta.py                 ← model writes one @program one-shot
│   ├── trace_retry.py                ← annotated retry trace
│   ├── act_01_schema_breaks.py       ← Act 1 supporting demo
│   ├── act_02_predicate_fixes.py     ← Act 2 supporting demo
│   ├── act_03_unified_yield.py       ← Act 3 supporting demo
│   └── act_04_arc_sketch.py          ← (older ARC sketch, not used in this video)
├── src/orate/
│   ├── __init__.py
│   ├── gen.py             ← Choice, Integer, String, Boolean, Struct, ToolCall
│   ├── program.py         ← @program decorator, ProgramInvocation, reject_program
│   ├── compile.py         ← witness enumeration, forward checking
│   ├── verify.py          ← @verifier, Accept/Reject
│   ├── meta.py            ← PROGRAM_SOURCE_GRAMMAR, validate_program_source, compile_program_source
│   ├── body_grammar.py    ← derive_body_grammar_rules — call-site arg grammar
│   ├── prompt.py          ← source-in-prompt with description= as comments
│   ├── session.py         ← Session class, outer grammar, registry, dispatch
│   └── engine/
│       ├── xgrammar.py    ← llama-cpp-python + XGrammar; begin_session / append / sample_under
│       └── mock.py
├── tests/                  ← 232+ tests; ruff clean
└── pyproject.toml
```

**Models on disk** (`/Users/maltelandgren/models/`):
- `Qwen2.5-7B-Instruct-Q4_K_M.gguf` ← primary demo model
- `qwen2.5-0.5b-instruct-q4_k_m.gguf` ← cheap smoke tests
- `qwen2.5-1.5b-instruct-q4_k_m.gguf`, `qwen2.5-3b-instruct-q4_k_m.gguf` available

**Run the existing session demo** (verifies your environment):
```bash
cd /Users/maltelandgren/code/Private/sandbox/orate
.venv/bin/python examples/smoke_session.py
```
Expect: Qwen-7B authors a tool, then invokes it, all on one KV. Takes ~1–2 minutes.

---

## 3. Companion repo (Act 1 source material)

Act 1 grounds the pain in real code from a sister project:
`/Users/maltelandgren/code/Private/sandbox/simulacrum`

Key files referenced in the script:
- `src/harness/agents/dm.py:283` — DM agent returns plain string, no structured output
- `src/harness/orchestrator.py:514` — post-hoc reconstruction of tool calls by parsing narration
- `src/harness/models.py:35–39` — model upgrade forced because gemini-flash-lite can't handle structured output + tool calls together
- `src/types/action_semantics.py:76–231` — conditional union types schema can't express
- `src/harness/valid_actions.py:1–50` — post-hoc validation that `where=` would shift to decode-time

Open both repos in a split editor for filming Act 1.

---

## 4. The script in one paragraph

[`video_script.md`](video_script.md) is the shot list — read it next. In short: Act 1 (0:00–0:25) is the simulacrum pain. Act 2 (0:25–0:55) is the `@program` primitive that solves it. Act 3 (0:55–1:55) is one continuous D&D session showing narration, dice tool with `ends_turn`, deferred `@remember` blocks, mode-switching to combat, composed-per-stat-sheet NPC programs. Act 4 (1:55–2:50) is the uppercut: the model authors `@algebra_step` and `@inference_step` programs that enforce legal-step semantics on hard math/logic problems — capability the same model lacks in free text. Close (2:50–3:00) is the thesis card.

**Two beats to call out:**
- Act 4 has **algebra (mandatory)** + **logic (cuttable)**. Cut logic if total runtime exceeds 3:00.
- An older draft used a "bard speaks only in palindromes" beat. **It is cut** — reads as parlor trick, not capability. Don't reintroduce it.

---

## 5. Engineering punch list (must ship before filming)

What's already shipped (don't redo):
- Persistent KV session, `@call` outer grammar
- `@make_new_program` mid-session, source synthesis under `PROGRAM_SOURCE_GRAMMAR`
- `@program` composition (Flavor B minimal — programs invoke programs)
- `gen.struct(...)` (used for `@remember`, just a schema, client-side buffering)
- 232+ tests, ruff clean

What needs to ship (~12 hr total):

| Capability | File(s) | Effort | Notes |
|---|---|---|---|
| `ends_turn=True` loop | `session.py`, `program.py` | ~3 hr | Field exists on `ProgramInvocation`. Need: when set, session yields structured args to client, client returns result, session injects result text into KV, decoding continues under outer grammar. |
| Grammar mode-switch | `session.py` | ~1–2 hr | Today the outer grammar is fixed at session construction. Needs `session.set_outer_grammar(g)` swap — the engine just uses whichever grammar is active for the next `sample_under` call. |
| Typed kwargs (`@algebra_step("x + y = 5")`) | `body_grammar.py`, `session.py:_parse_args` | ~2 hr | Today's `_parse_args` splits on `, ` and treats everything as strings. Needs to consult the body grammar's per-yield types to coerce ints, bools, strings. |
| 3 composed NPC programs for combat | new `examples/d20/` | ~1 hr | Hand-author `hooded_figure_turn`, `aria_turn`, `borin_turn` per stat-sheet. Compose into `combat_round`. |
| `equivalent_under(rule, before, after)` | new helper | ~1 hr | SymPy: `sympy.simplify(parse(before) - parse(after)) == 0`, scoped per rule. |
| `derivable_under(rule, premises, conclusion)` | new helper | ~1 hr | Modus ponens, modus tollens, hypothetical syllogism — explicit case dispatch over the small rule set. |
| Demo `@program`s + problem statements | new `examples/legal_steps/` | ~1 hr | `algebra_step.py`, `inference_step.py`, plus the two problems in the script. |
| Visualization assets | manim — see §6 | ~3 hr | The reason you have the manim skill. |

If you only have time for one cut: **drop typed kwargs**. Use string-typed args in the demo (`@algebra_step("x + y = 5")` already passes a string). Saves ~2 hr.

---

## 6. Manim visualizations needed

You have the `manimce-best-practices` skill at `.agents/skills/manimce-best-practices/`. Follow it. Targets are 1080p, transparent background where possible (we composite over terminal recordings in post), 30 fps.

### V1 — Grammar stack indicator (Act 3, persistent on right side)

A vertical stack of grammar-name pills. The active grammar at top is highlighted. When the model calls into a sub-grammar (`@roll(...)`), a new pill pushes onto the stack with a small slide animation. When the sub-grammar exits, it pops. When mode-switches to combat happen, the entire stack is *replaced* with a swap animation (fade out → fade in, not push/pop).

Sequence to animate (matches script Act 3 timestamps):
- 0:55 — single pill: `narrative_outer`
- 1:05 — push `roll_args`, hold ~0.5s, pop back to `narrative_outer`
- 1:20 — push `remember_struct`, hold, pop
- 1:30 — replace `narrative_outer` with `combat_outer` (mode-switch animation)
- 1:40 — under combat: push `hooded_figure_turn` → pop → `aria_turn` → pop → `borin_turn` → pop
- 1:50 — replace `combat_outer` with `narrative_outer` (mode-switch back)

### V2 — KV cache token bar (Act 3, persistent at bottom)

A horizontal bar that grows monotonically. Token count ticks up smoothly as the session proceeds. **It never resets.** A small label "KV: N tokens" follows the right edge. Color-code segments: narration (blue), tool args (orange), tool results injected back (green), structured blocks (purple), combat (red).

The point this visualization makes: *one continuous KV from start to finish.*

### V3 — Stat sheet → action program → combat loop diagram (Act 3, ~1:40, ~5s cutaway)

A three-column flowchart. Left: a D&D stat sheet (HP, AC, attacks list). Middle: arrows showing how each stat-sheet field becomes a `@program` element (attacks → `gen.choice(attacks)`, AC → `where=` predicate on attack rolls, HP → conditional flow control). Right: the combat-round program composing the three character programs round-robin.

Animate: stat sheet draws in → arrows trace one by one → action program assembles → three action programs slot into the combat loop.

### V4 — Algebra step transformation (Act 4, ~2:10–2:25)

Each line of the algebra solve animates: starting equation written, the rule label appears (e.g., `isolate_var`), arrows show the algebraic move (annotation: "subtract y from both sides"), new equation morphs into place. **TransformMatchingTex is the right primitive here** — match `x` to `x`, `y` to `y`, `=` to `=`, watch the structure rearrange.

Four steps per the script:
```
x + y = 5         → isolate_var → x = 5 - y
2(5-y) + 3y = 12  → simplify   → 10 + y = 12
10 + y = 12       → isolate_var → y = 2
x = 5 - 2         → evaluate   → x = 3
```

Final verification animation: `2(3) + 3(2) = 12 ✓`, `3 + 2 = 5 ✓`, `3 > 2 ✓`. Each ✓ pops in with a small bounce.

### V5 — Logic deduction tree (Act 4 Beat 2, ~2:25–2:45) — cuttable

Premises at the top: `P → Q`, `Q → R`, `P`. Animated tree expands downward as each `@inference_step` runs:
- Step 1: lines from `P → Q` and `P` converge into a node labeled `modus_ponens`, output `Q` drops below
- Step 2: lines from `Q → R` and the just-produced `Q` converge, output `R` drops below
- Final `R` lights up with QED

### V6 — Registry growing panel (Act 4 close, ~2:45–2:50)

A list of program names appearing one by one in the order the demo introduced them: `dm_turn`, `roll`, `remember`, `enter_combat`, `hooded_figure_turn`, `aria_turn`, `borin_turn`, `algebra_step`, `check_solution`, `inference_step`. Each appears with a small fade-in.

The visual point: *the library grew during the demo*.

### V7 — Closing card (3:00)

```
Structured output constrained the shape.
Tool calling constrained the side effect.
orate lets the model enforce the legality of its own thought.

github.com/...
```

Hold 5 seconds. Simple, typographic, no animation beyond a slow fade-in line by line.

---

## 7. Terminal recordings needed (asciinema, ~720p crop)

R1. **Simulacrum mid-session** (Act 1, ~5s) — any active D&D scene playing
R2. **simulacrum/src/harness/agents/dm.py at line 283** — editor view, return type highlighted (~3s)
R3. **simulacrum/src/harness/orchestrator.py at line 514** — editor view, reconstruction loop highlighted (~3s)
R4. **`dm_turn` `@program` source** — code on screen, syntax highlighted (~5s)
R5. **`@program` running** — terminal output of `dm_turn` invocation, narration → tool call → narration (~10s)
R6. **`where=` aside** — code snippet for attack roll predicate (~3s)
R7. **The full Act 3 D&D session** — one continuous run, ~50s. This is the hardest record. Use a real run from `examples/d20/full_session.py` (which you'll author per §5).
R8. **Free-text Qwen-7B solving the algebra problem, two seeds** — show one correct, one wrong (~10s, side-by-side)
R9. **Act 4 algebra session** — model authors `@algebra_step`, composes the solve, verifier passes (~25s)
R10. **Act 4 logic session** — model authors `@inference_step`, derives R (~15s, cuttable)

For R7, R9, R10: **real runs, take best of three**. The pitch is the library actually does this. Pre-record into asciinema files; you can speed up dead air in post.

---

## 8. Decisions locked in

- **Algebra + logic in Act 4.** Cut logic if runtime tight.
- **No bard/palindrome beat.** Reads as parlor trick.
- **No "model commits code to library" beat.** Different thesis (code execution exists everywhere; what's new is operations on own future generation).
- **Real runs over staged.** Rerun until the take is clean, but don't fake it.
- **Local Qwen-7B as the demo model.** Local-first is a stated principle. Opus 4.7 may be used as the meta-author if Qwen-7B's authored programs are too rough — call back to the user before swapping.
- **One KV from 0:00 to 3:00 in narrative.** Even if the recordings are stitched, the visualizations and voiceover should sell continuity.

## 9. Decisions to bring back to the user

Don't make these alone. Ping back with options:

1. **Voiceover talent.** User does it himself, AI TTS, or hired? Currently the script reads as first-person ("I've been building..."). User-voiced is on-brand.
2. **Free-text vs constrained side-by-side in Act 4.** Two parallel terminals or sequential cuts? Parallel reads stronger but is harder to film cleanly.
3. **Music.** None? Subtle bed? Pick one before you start editing.
4. **Final 10 seconds.** Just the thesis card, or also a 3-second "Built with Opus 4.7" hackathon credit?
5. **GitHub URL.** Repo is at `/Users/maltelandgren/code/Private/sandbox/orate` locally; the public URL needs to be in the closing card. Confirm it exists / is named what.

## 10. Receipts to verify the handover landed

You should be able to answer all of these without re-reading the script:

- What does `@make_new_program` do, and what grammar does the model decode under when it's invoked?
- Why is `@remember` zero engineering effort but `@enter_combat` not?
- What does the `where=` predicate constrain, and at what point in the decode loop?
- What's the difference between Act 3's grammar mode-switch and Act 3's grammar push/pop?
- What's the *one* thing Act 4 is showing that no existing API can do?

If any of those is fuzzy: re-read this doc + `video_script.md` before starting.

---

## 11. First moves (when you're ready to start)

In this order:
1. Run `examples/smoke_session.py` end-to-end. Confirm the existing Act-4 demo still works on your machine.
2. Read `video_script.md` cover to cover.
3. Open both `orate/` and `simulacrum/` in your editor; open all the script-referenced files in tabs.
4. Decide: are you starting with the engineering punch list (§5) or the manim assets (§6)? They're independent; assets can be authored against the script alone, engineering needs working code.
5. Confirm with the user before recording anything: voiceover plan, music, GitHub URL.

Ship it. The pitch is good and the library is real.
