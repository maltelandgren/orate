# video/ — manim scenes + renders

The hackathon submission video is rendered from `scenes/full_video.py`.
Other files in `scenes/` are exploratory; keep them around as
references but only `full_video.py` is canonical.

## The video

**`scenes/full_video.py` → `FullVideo`** — a single Scene that plays the
complete 3:00 pitch:

| Beat | Time |
|---|---|
| Act 1 — protagonist-first cold open | 0:00–0:25 |
| Act 2 — the @program primitive | 0:25–0:55 |
| Act 3 — D&D session: roll, mode-switch, composed NPCs | 0:55–1:55 |
| Act 4 Beat 1 — algebra contrast (4/7 vs 6/7) | 1:55–2:25 |
| Act 4 Beat 2 — logic via composer pattern | 2:25–2:40 |
| Act 4 Beat 3 — meta-authorship finisher | 2:40–2:55 |
| Close — thesis card | 2:55–3:00 |

Aesthetic: D1 paper throughout, terminal-dark insets only where
literal source code is the visual. Beat 2 of Act 1 carries the
"Type, not logic" framing user-flagged in the feedback notes.

## Render

`renders/` is gitignored — regen locally:

```bash
cd video/scenes

# Low-quality preview (480p15, ~30s render time)
manim -ql --format mp4 --media_dir ../renders full_video.py FullVideo

# Medium quality (720p30, ~2 min render time, ~3.8 MB)
manim -qm --format mp4 --media_dir ../renders full_video.py FullVideo

# High quality (1080p60, ~10 min render time, ~10 MB) — submission
manim -qh --format mp4 --media_dir ../renders full_video.py FullVideo
```

Output paths after a clean render:

```
video/renders/videos/full_video/1080p60/FullVideo.mp4   ← submission
video/renders/videos/full_video/720p30/FullVideo.mp4
video/renders/videos/full_video/480p15/FullVideo.mp4
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
└── full_video.py              — ★ the 3:00 submission
```

`full_video.py` borrows components (`LLMProtagonist`, `LogitItem`,
`Paper` palette, `paper_grid`, `card`) from `theme.py` and `llm.py`.

## Known issues / caveats

- **`stream_tokens` wrap is brittle past one line.** Call `.newline()`
  at natural breaks; the auto-wrap in `LLMProtagonist._append_token`
  works for single overflow but stacks badly on multiple wraps.
- **`full_video.py` Act 3 narration is a placeholder.** The `@narrate`
  leaf exists in `examples/d20/dice.py` but its body grammar
  (`gen.string(max_len=120)`) compiles to a pathological matcher that
  hangs Qwen-7B. Until that's fixed (recursive `chars*` rule with
  post-sample length cap), the video shows narration as a captioned
  text card during the roll/combat sequence.
- **Act 4 Beat 3 source** in the video shows an idealised
  `quadratic_solver` body using `gen.integer`. The real Qwen-7B trace
  authored a body using `gen.boolean()` (smallest grammar — model
  picks the cheapest path). See
  `bench/results/act4_meta_finisher_2026-04-25_1515.md` for the
  honest trace; the video idealises it for visual clarity (the
  predicate-bound version is on the roadmap).
- **Shadows** stack rounded-rects to simulate blur. At 1080p+ may
  show banding; add another layer or gradient if it bothers.

## Production notes

The video has no voiceover yet. Captions in the manim scene act as
placeholders for what the voice would say. Add VO over the medium-
or high-quality render in any NLE (Final Cut, DaVinci, Premiere).

GitHub URL on the closing card: `github.com/maltelandgren/orate` —
update if needed.
