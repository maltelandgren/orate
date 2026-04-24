# video/ — overnight exploration

Built on branch `video/explore` (worktree: `../orate-video`). Read
[`docs/video_explorations.md`](../docs/video_explorations.md) first —
it's the narrative. This file is the cheat sheet.

## Watch these (in order)

All paths under `video/renders/` (gitignored — regen with the commands below).

1. **`videos/candidate_full/720p30/candidate_full_hq.mp4`** — 25s mini of
   Acts 1→4 in D3 hybrid style. The recommendation.
2. **`videos/d3_hybrid/720p30/d3_hybrid_hq.mp4`** — direction 3 in
   isolation, ~15s.
3. **`videos/d1_paper/480p15/d1_paper.mp4`** — direction 1 (warm paper only).
4. **`videos/d2_terminal/480p15/d2_terminal.mp4`** — direction 2 (terminal only).
5. **`videos/v2_protagonist_open/480p15/v2_protagonist.mp4`** — script
   variation V2, protagonist-first opening.

Open them in QuickTime or `open <path>` from the terminal.

## Re-render anything

```bash
cd video/scenes

# Low-quality preview (fast, ~15–60s)
manim -ql --format mp4 --media_dir ../renders d3_hybrid.py D3_Hybrid

# Medium quality (sharper text, ~2–3x slower)
manim -qm --format mp4 --media_dir ../renders d3_hybrid.py D3_Hybrid

# High quality 1080p60 (slow)
manim -qh --format mp4 --media_dir ../renders d3_hybrid.py D3_Hybrid
```

Scene class names match file names (e.g., `d3_hybrid.py` → `D3_Hybrid`;
`candidate_full.py` → `CandidateFull`).

## Files

```
scenes/
├── theme.py                   — palette, typography, helpers
├── llm.py                     — LLMProtagonist + LogitItem
├── smoke_llm.py               — sanity test for the LLM component
├── d1_paper.py                — D1 direction
├── d2_terminal.py             — D2 direction
├── d3_hybrid.py               — D3 direction ★
├── v2_protagonist_open.py     — V2 script variation
└── candidate_full.py          — 25s Act 1→4 mini ★★
```

## Known issues

- `LLMProtagonist.stream_tokens` wrap is brittle past one line; call
  `.newline()` explicitly at natural breaks rather than relying on
  automatic wrap.
- `candidate_full.py` has a crossfade glitch on the KV-token label
  around the ~17s mark (old + new label briefly overlap). Only
  visible as a single-frame artifact on extract; probably imperceptible
  on playback.
- Shadows are stacked rounded-rects simulating blur. At 1080p+ they
  may show banding. If so, add another layer or use a gradient.
