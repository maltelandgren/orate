# video/ — manim scenes + renders

The hackathon submission video is rendered from `scenes/full_video_v2.py`.
Other files in `scenes/` are exploratory; keep them around as
references but only `full_video_v2.py` is canonical.

## The video

**`scenes/full_video_v2.py` → `FullVideoV2`** — a single Scene that plays the
complete ~2:30 pitch in five "pages":

| Page | Beat | Approx. time |
|---|---|---|
| 1 | distribution shaping → developer accessible → bridge | 0:00–0:25 |
| 2 | structured output (type) ↔ the logic (program) | 0:25–0:55 |
| 3 | `@algebra_step` + free-text vs constrained contrast | 0:55–1:25 |
| 4 | D&D session: narrate / roll / meta, regrammar to combat, Aria's `where=` | 1:25–2:15 |
| 5 | model authors its own primitive → thesis | 2:15–2:30 |

Aesthetic: D1 paper throughout. The two load-bearing visuals on Page 4
are (a) the live grammar-indicator pill row that physically reshapes
when `@enter_combat` fires, and (b) the highlight on Aria's
`where=lambda d: ...` cross-field predicate — captioned as *"logic
constraint, in Python — across fields"* to underline that JSON Schema
literally cannot express this.

The script targets **iterative reveals** rather than dump-and-hold —
text appears, holds for the read, then transitions. The "Hard to write
programs around" → "Lets us write programs around" flip is one such
transition; the grammar-pill reshape on `@enter_combat` is another.

## Render

`renders/` is gitignored — regen locally:

```bash
cd video/scenes

# Low-quality preview (480p15, ~30s render time)
manim -ql --format mp4 --media_dir ../renders full_video_v2.py FullVideoV2

# Medium quality (720p30, ~2 min render time, ~4 MB)
manim -qm --format mp4 --media_dir ../renders full_video_v2.py FullVideoV2

# High quality (1080p60, ~10 min render time) — submission
manim -qh --format mp4 --media_dir ../renders full_video_v2.py FullVideoV2
```

Output paths after a clean render:

```
video/renders/videos/full_video_v2/1080p60/FullVideoV2.mp4   ← submission
video/renders/videos/full_video_v2/720p30/FullVideoV2.mp4
video/renders/videos/full_video_v2/480p15/FullVideoV2.mp4
```

Open with `open <path>` (QuickTime).

## Other scene files (reference / exploration)

```
scenes/
├── theme.py                   — palette, typography, paper grid, card helpers
├── llm.py                     — LLMProtagonist + LogitItem (the animated LLM)
├── smoke_llm.py               — sanity test for the LLM component
├── d1_paper.py                — D1 direction (warm paper, ~12s)
├── d2_terminal.py             — D2 direction (dark code, ~12s)
├── d3_hybrid.py               — D3 direction (paper + terminal inset, ~15s)
├── v2_protagonist_open.py     — protagonist-first Act 1 prototype
├── candidate_full.py          — 25s Acts 1→4 mini in D3
├── full_video.py              — V1 cut (3:00, four-act structure)
└── full_video_v2.py           — ★ the ~2:30 submission cut, five-page pacing
```

Both `full_video.py` and `full_video_v2.py` are runnable; v2 is the
canonical submission. v1 is kept for diff/comparison purposes (it
follows the four-act script; v2 follows the five-page revision with
tighter reveal/hide pacing).

## Known issues / caveats

- **`stream_tokens` wrap is brittle past one line.** Call `.newline()`
  at natural breaks. Used by the LLMProtagonist component, not the v2
  scene proper (v2 uses static code blocks for clarity).
- **Page 4 narration as text rather than streamed tokens.** The
  `@narrate` leaf exists in `examples/d20/dice.py` but its body
  grammar (`gen.string(max_len=120)`) compiles to a pathological
  matcher that hangs Qwen-7B. Until that's fixed (recursive `chars*`
  with post-sample length cap), the video shows narration as text.
- **Page 5 meta-authorship body** in the video shows an idealised
  `quadratic_solver` body using `gen.integer`. The real Qwen-7B trace
  authored a body using `gen.boolean()` (smallest grammar — model
  picks the cheapest path). See
  `bench/results/act4_meta_finisher_2026-04-25_1515.md` for the
  honest trace; the video idealises it for visual clarity. A small
  on-screen footnote *"(today: typed schema. predicate-bound bodies
  on the roadmap.)"* keeps the scope honest.
- **Aria's `aria_turn` source on Page 4** is rendered directly in the
  scene file as a code-text mobject — it does NOT match
  `examples/d20/characters.py` literally (the on-disk character uses a
  3-yield body without `gen.struct`+`where=`). Per direction in the
  brief: "invent the on-screen `aria_turn` program directly in the
  scene file as a code-text mobject. The point is the visual on
  screen, not necessarily a runnable demo."
- **Shadows** stack rounded-rects to simulate blur. At 1080p+ may
  show banding; add another layer or gradient if it bothers.

## Production notes

The video has no voiceover yet. Captions in the manim scene act as
placeholders for what the voice would say. Add VO over the medium-
or high-quality render in any NLE (Final Cut, DaVinci, Premiere).

GitHub URL on the closing card: `github.com/maltelandgren/orate` —
update if needed.
