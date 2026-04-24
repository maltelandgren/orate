# Video production exploration — findings

Work done overnight (2026-04-24 → 2026-04-25) to explore visual directions
for the 3-minute hackathon submission. This doc is the summary; the renders
are in `video/renders/` on branch `video/explore`. The source is in
`video/scenes/`.

---

## TL;DR

- **Go with D3 "Hybrid."** Warm Anthropic-style paper for voice and thesis;
  terminal-dark inset for technical moments. See `video/renders/videos/d3_hybrid/720p30/d3_hybrid_hq.mp4`.
- **The "LLM protagonist" works.** A reusable animated card that streams
  tokens, opens a logit column, and shows grammar masking. See `video/scenes/llm.py`.
- **Candidate `candidate_full.py` renders ~25s covering Acts 1→4 in
  miniature.** Good enough to judge pacing/palette end-to-end.
  `video/renders/videos/candidate_full/480p15/candidate_full.mp4`.
- **Two script framings were prototyped.** The original "pain → primitive
  → thesis" lands harder; the protagonist-first "Keep thinking" reframe is
  more poetic but buries the use case. Lean original.
- **Remaining for final render:** Act 3 D&D session, real asciinema inserts,
  higher-quality assets for algebra/logic money shot, voiceover.

## Branch and files

Everything is on a worktree at `../orate-video` on branch `video/explore`
so it doesn't mix with the main repo. Key files:

```
video/
├── scenes/
│   ├── theme.py                    — palette, typography, helpers
│   ├── llm.py                      — LLMProtagonist + LogitItem (core component)
│   ├── smoke_llm.py                — sanity test, renders LLM alone
│   ├── d1_paper.py                 — Direction 1: Anthropic warm/paper
│   ├── d2_terminal.py              — Direction 2: terminal/code-forward
│   ├── d3_hybrid.py                — Direction 3: paper + terminal inset ★
│   ├── v2_protagonist_open.py      — script variation: protagonist-first
│   └── candidate_full.py           — 25s Act 1→4 mini in D3 style ★
└── renders/videos/.../480p15/      — low-quality previews
           .../720p30/              — medium-quality (only d3_hybrid_hq)
```

Render command used (low-quality preview):
```bash
cd video/scenes
manim -ql --format mp4 --media_dir ../renders <file>.py <SceneClass>
```

## Directions explored

### D1 — Paper / Anthropic warm
`video/renders/videos/d1_paper/480p15/d1_paper.mp4` (15.5s)

- **What it is:** Closest to the reference Cowork-style announcement video.
  `#F2EBE5` background, Georgia serif for stylized text, terracotta (`#D4704A`)
  accent, soft-shadowed white cards. Grid is visible but faint.
- **What works:** The opening "orate" italic title is elegant. The serif
  closing thesis ("Structured output constrained the shape...") lands with
  weight. The grid gives the product-y "notebook" feel the reference has.
- **What doesn't:** It's hard to show *machinery* here. The logit column
  feels out-of-place on warm paper — too "technical" for the surrounding
  aesthetic. Terminals-in-paper look noisy unless deliberately framed.
- **When to use:** The opening title, closing thesis, and any narrative
  voiceover beats. Do NOT use for the algebra/logit money shot.

### D2 — Terminal / code-forward
`video/renders/videos/d2_terminal/480p15/d2_terminal.mp4` (15.3s)

- **What it is:** Dark `#0B0D10` background, monospace everywhere, amber
  and terracotta semantic colors (amber = type signatures, green = OK,
  red = strike). Grammar-stack rail on the right.
- **What works:** Shows the technical side cleanly. Mask/strike reads
  strongly on dark. Terminal aesthetic is "honest" for a hackathon — it
  says "this is the actual library."
- **What doesn't:** Feels generic. Every ML demo looks like this. The
  thesis in mono lacks weight — the line "orate lets the model enforce
  the legality of its own thought" needs serif or it reads as a commit
  message.
- **When to use:** Never as the whole video. Works great *inside* the
  hybrid direction as the inset during technical beats.

### D3 — Hybrid (recommendation) ★
Preview: `video/renders/videos/d3_hybrid/480p15/d3_hybrid.mp4` (15.5s)
High-quality: `video/renders/videos/d3_hybrid/720p30/d3_hybrid_hq.mp4` (14.9s)

- **What it is:** Paper world for voice/narrative/thesis. When we "zoom
  into the machinery," a terminal-dark inset snaps in over the paper
  background with a soft shadow. The logit column, grammar rail, and
  code all live inside that inset. When we're done, the inset fades
  and the paper bg returns for the closing thesis.
- **Why it wins:**
  1. The transition from warm narrative → technical inset does real
     semantic work. It *is* the "opening the box" metaphor — we go from
     "here's what the library feels like" to "here's what's actually
     happening" and back.
  2. It solves the single biggest visual problem: the thesis needs serif
     weight, the mechanism needs monospace precision. Hybrid lets you
     have both without a style collision.
  3. The inset shadow + border makes the inset feel like a panel, not
     a mode switch. Cohesive.
- **What still needs work:**
  - The zoom-in is currently a FadeIn with `scale=0.9`. Should be a real
    scale-from-center animation with a spring easing (`ease_out_back` or
    a custom rate function). Bigger "pop."
  - The inset currently has no grid overlay. Could add a very faint
    matrix to suggest "logit space."
  - Code should syntax-highlight more — right now it's line-by-line
    color-picking. Could use a real pygments pass + token-to-color map.

### D4 — (not prototyped) Blueprint / diagram-heavy
A flowchart-style treatment: LLM as labeled block, inputs/outputs drawn
as formal engineering arrows, grammar as a finite-state diagram. Decided
to skip: reads as a conference paper, not a pitch. The "LLM as character"
metaphor is stronger than the "LLM as block in a diagram" metaphor.

---

## Script-framing variations

### Original (pain-first)
`docs/video_script.md`. Works as-is. Opens on simulacrum pain in a real
personal project, pivots to `@program` primitive, runs a D&D scene,
reveals the meta-authorship trick. The pain opening earns the whole arc.

### V2 — Protagonist-first / "Keep thinking" reframe
`video/renders/videos/v2_protagonist_open/480p15/v2_protagonist.mp4` (12s)

- Opens on the LLM protagonist itself — a box streaming "the cat sat
  on the..." with the logit column visible. Voiceover: "Every token is a
  sample from a distribution over its vocabulary."
- Then: "Three years in, we're good at shaping those distributions" →
  chips for prompts, fine-tuning, structured output, tool calls.
- "But shape and effect aren't the same as legality."
- Two-run divergence (same algebra problem, one right / one wrong).
- Hook: "orate closes the gap."

- **Works as:** an alternative 30s opening if we want the video to feel
  less personal-narrative and more category-defining.
- **Tradeoff:** loses the concrete "I was building X and hit Y" grounding.
  Replaces it with something more essay-like. Reads more philosophical,
  less honest-maker.
- **Recommendation:** keep the original pain-first opening. The V2 is a
  backup if the simulacrum recording in Act 1 turns out to be hard to
  shoot cleanly.

---

## Candidate (full arc, ~25s)

`video/renders/videos/candidate_full/480p15/candidate_full.mp4`

A miniature of the full 3-minute video in the D3 hybrid style. Covers:

| Beat | Timing | Content |
|---|---|---|
| A | 0:00–0:03 | Title card (serif italic "orate") |
| B | 0:03–0:10 | "Same model, same prompt, two runs" — one right, one wrong |
| C | 0:10–0:15 | Zoom into orate inset, show `@program` code w/ `where=` callout |
| D | 0:15–0:19 | Grammar bites: logit column with mask, `y = 2` chosen |
| E | 0:19–0:22 | Session KV bar growing, registry populating (6 programs) |
| F | 0:22–0:25 | Thesis card |

- **What it proves:** the style carries across all the beats of the
  actual video. The aesthetic stays coherent from pain to thesis.
- **What's rough:**
  - The inset fade-in could be a proper scale-up.
  - Beat E is too fast — in the real 3-min video this should be 30s
    with the grammar stack indicator visible on the right rail and
    multiple push/pop animations.
  - Beat D's logit column position collides with the inset's left edge
    at some sizes — needs a tighter layout pass.
- **Use this to decide pacing for the real cut.** If the warm↔technical
  oscillation works for you in this 25s preview, commit to D3 for the
  full video.

---

## The LLM protagonist component

`video/scenes/llm.py` — `LLMProtagonist(VGroup)`.

Designed to appear in every scene where the model itself is the subject.
Reusable across palettes. Methods:

- `stream_tokens(scene, tokens, speed=, color=)` — typewriter effect
- `newline()` — drop cursor to a new line at the left margin
- `open_logits(scene, logits, column_width=, gap=)` — slide a logit column in
- `apply_grammar_mask(scene, mask_indices=)` — grey out + strike rows
- `choose_logit(scene, idx)` — highlight a row and pull its token into output
- `close_logits(scene)` — dismiss the column
- `clear_output(scene)` / `fade_everything(scene)` — housekeeping
- `pulse_thinking(scene, duration=)` — rotating terracotta starburst

Used in D1, D2, D3, V2, and candidate_full. Two palettes: `"paper"` and
`"terminal"`.

### Known bugs / fragility

- `stream_tokens` wrap logic: tokens past the right edge wrap to a new
  line, but if *that* line also fills, the second wrap can land back on
  the same line it just wrapped from. Workaround: call `.newline()`
  explicitly at natural breaks. Don't rely on automatic wrap beyond one
  line.
- `choose_logit` previously left a stray row copy on stage — now fixed
  (direct `.animate.set_stroke(...)` on the row itself).
- Masked-row strikes were not children of the logit group — they persist
  after `close_logits`. Fixed: strikes are now added to `self.logit_group`.

These fixes are committed; no action needed.

---

## Open decisions (for the user)

These are the same "bring back to user" items as the handover doc, plus
ones I ran into while prototyping.

1. **Commit to D3 hybrid for the final cut?** If yes, I keep building on
   `candidate_full.py` — next stops: proper scale-in inset animation,
   Act 3 D&D montage, Act 4 long-form algebra+logic money shot.
2. **Voiceover.** Malte-voiced is on-brand — do you want to record
   scratch VO against the 25s candidate so we can time the full cut?
3. **Script opening.** Original (pain-first) or V2 (protagonist-first)?
   My vote: original — stronger grounding.
4. **Real asciinema inserts.** The handover spec calls for ~10 terminal
   recordings (R1–R10). Am I doing those, or are you? I can write the
   demo `@program`s for R9/R10 (algebra_step / inference_step) tonight
   if you want.
5. **Music.** Not touched — suggest a soft ambient bed under voice with
   a hard-silence drop at "orate lets the model enforce the legality of
   its own thought." Final decision yours.
6. **GitHub URL.** Closing card currently has no URL baked in. What do
   you want on line 5 of the thesis?

---

## What I did NOT do (intentionally, for time)

- **No Act 3 D&D animation.** The grammar-stack push/pop and KV token
  bar visualizations from the handover (V1/V2) need their own scene
  file and I didn't get there. The candidate_full.py Beat E is a
  compressed proxy.
- **No TransformMatchingTex animation for algebra steps.** Per the
  handover, the four-step algebra solve should use the MatchingTex
  primitive so `x`, `y`, `=` morph in place. Currently the candidate
  doesn't render any of the steps explicitly — only the single chosen
  `y = 2` token.
- **No registry growing panel as a standalone.** Rolled into Beat E of
  candidate_full.py, but could be its own thing.
- **No asciinema inserts wired up.** The spec calls for these as overlay
  + underlay with manim on top. Haven't touched asciinema.
- **No voiceover timing pass.** Without a VO track, pacing is guesswork.

---

## Next steps if I keep going

In priority order:

1. Polish D3 candidate: proper spring scale-in on the inset, cleaner
   Beat E timing, one extra beat for Act 3 D&D session
2. Render `candidate_full.py` at `-qh` (1080p 60fps)
3. Author `examples/legal_steps/algebra_step.py` etc. so R9 (real asciinema
   of the model authoring `@algebra_step`) is actually possible
4. Build the Act 3 montage scene separately (D&D session, 50s)
5. Composite pass in an NLE (Final Cut / DaVinci / Premiere) — manim
   scenes + asciinema + voice, colored per the chosen palette
