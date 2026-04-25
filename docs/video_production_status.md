# Video production status — Saturday Apr 25

**Submission:** Sunday Apr 26, 8pm EST.
**Manim direction:** D1 paper aesthetic (per user direction; was D3 hybrid).

> Top-line: every Act's *content* is real and runnable. Two capability
> gaps that were caveated in earlier versions are now closed:
> ``@narrate`` runs at speed (recursive ``gen.string`` grammar — commit
> ``4da326a``); model-authored ``@program``s can carry ``where=``
> predicates from a host-provided library (commit ``6473880``). The
> video v2 (``video/scenes/full_video_v2.py``) is being authored to
> match the revised script with tighter pacing.

## Saturday-evening update

What landed since the last "production status" snapshot:

| Capability | Commit | What it unblocks |
|---|---|---|
| Per-leaf grammars + composer + ``gen.alternative`` | `c042cf7` | Cleaner Act-4 agent loop on screen (5 lines of Python) |
| Engine grammar cache + warm() | `8247deb` | Kills 232s cold-start; Session calls run at steady state from token 1 |
| Real ``ends_turn`` for client-resolved tool calls | `d70ba37` | ``@roll`` returns the resolved d20; tool calls and structured output are the same yield stream |
| Recursive ``gen.string`` grammar | `4da326a` | ``@narrate`` runs in ~2s, was hanging for minutes |
| ``@narrate`` + ``@meta`` in Act 3 demo | `325d209` | In-character + out-of-character narration as same-shape tools — multiple same-shape outputs welcome, no XML-tag wrapping |
| Predicate-bound model authoring (``where=`` in PROGRAM_SOURCE_GRAMMAR) | `6473880` | Beat 3 finisher can show the model authoring **logical** constraints, not just typed schemas |

The Act-3 full trace at [`bench/results/act3_full_2026-04-25_2330.md`](../bench/results/act3_full_2026-04-25_2330.md) shows the entire scripted Page-4 sequence end-to-end on Qwen-7B: in-character narration → `@roll(perception, 13)` → `@meta` reaction → in-character narration informed by the d20 result → mode-switch into combat → three NPC turns → mode-switch out.

---

## What's ready

### Demos that run end-to-end on Qwen2.5-7B

| beat | runner | last verified | trace |
|---|---|---|---|
| Act 3 D&D combat | `examples/d20/act3_combat_demo.py` | Apr 25 (overnight) | mode-switch + composed NPCs — 5 events, clean |
| Act 4 algebra | `examples/legal_steps/act4_algebra_demo.py` | Apr 25 (overnight) | 3 steps + @done, 0 rejects |
| Act 4 logic | `examples/legal_steps/act4_logic_demo.py` | Apr 25 (overnight) | 2 modus ponens + @qed |
| Act 2 (composer variant) | `examples/legal_steps/act4_algebra_composer.py` | Apr 25 | the `gen.alternative` path, no Session |

These are reproducible. Re-run any of them; expect the same trace.

### Benchmark numbers ([`bench/results/legal_steps_2026-04-25_1132.md`](../bench/results/legal_steps_2026-04-25_1132.md))

7 problems × 2 modes × Qwen-7B argmax:

- **Free text: 4/7 correct.** Wrong answers: x=4 on `3x+5=14`, x=12 on `2x+3=x+9`, x=39 on `5-2x=1`.
- **Constrained: 6/7 correct.** The 7th (`3(x+1)=12`) hit `max_calls` after 9 predicate rejections — predicate working, model couldn't find a valid path in budget.
- **11 predicate rejections caught en route** across the constrained suite.
- **Free text: 10–17 tok/s.** Healthy.
- **Constrained: 5–8 tok/s steady-state**, plus a one-time ~10× cold-start penalty on the first session call after a mode switch (fix candidates documented in the bench markdown).

The single best contrast shot is `eq_3x_plus_5`: same model, same problem, free-text says x=4, constrained says x=3. Both runs are deterministic and reproducible.

### Manim POCs (overnight work, on `video/explore` branch)

- `theme.py` — D1/D2/D3 palettes, typography helpers
- `llm.py` — `LLMProtagonist` reusable component (token streaming, logit column, grammar masking, choice highlight)
- `d3_hybrid.py` — direction-3 standalone (~15s)
- `candidate_full.py` — 25s mini covering Acts 1→4 in D3 style
- Various rendered MP4s under `video/renders/videos/`

**Recommendation locked in:** D3 hybrid. Paper warm voice + terminal-dark inset.

### Script ([`docs/video_script.md`](video_script.md))

- 3:00 target, four-act structure, locked decisions documented (no palindrome beat, no "model commits code" beat).
- **Just updated** with the benchmark's contrast shot numbers + the easier `3x + 5 = 14` problem (which Qwen-7B reliably slips on — better contrast than the 2-variable one).

---

## What's still needed

In priority order to get to a finished video:

### 1. Asciinema recordings (~1 hour, automatable)

For each demo, capture a clean terminal run as an asciinema cast. The composite can then play these as overlays/insets on top of manim scenes. Targets:

- **R1**: `act4_algebra_demo.py` running. Already produces a clean trace.
- **R2**: `act4_algebra_demo.py` with the *free-text* path — i.e. just dump the contents of the JSON `output` field for `eq_3x_plus_5` from the benchmark, animated as if typing.
- **R3**: `act3_combat_demo.py` running. Mode-switch event included.
- **R4**: (if keeping logic beat) `act4_logic_demo.py` running.
- **R5**: simulacrum mid-session — **needs the user**, can't be automated from this repo.

The benchmark already produced the verbatim free-text output for R2; recording is just typing-animation playback.

### 2. Manim scene authoring (~3–4 hours)

Build on `candidate_full.py`. Specific scenes still missing per the handover:

- **Algebra TransformMatchingTex.** Four-step solve where `x`, `y`, `=` morph in place. Today the candidate shows only the chosen `y = 2` token.
- **Act 3 D&D montage.** Grammar-stack push/pop animations + KV bar grow + mode-switch swap. The handover spec breaks this into 6 sub-beats; none are currently animated.
- **Free-text vs constrained side-by-side.** The contrast shot for Act 4 Beat 1. Two terminals, one shows `x = 4 ✗`, the other `x = 3 ✓`. Source numbers from the benchmark.
- **Registry growing panel.** Programs accumulating across the video.

### 3. Simulacrum Act 1 recording (~30 minutes, needs user)

Open simulacrum, play one DM turn that exercises narration + dice + NPC. Record screen. Reference files (`dm.py:283`, `orchestrator.py:514`) already named in the script.

### 4. Voiceover (~1 hour, needs user)

Script is locked. Recommend a single straight read, no music yet. Time against the candidate_full to see if pacing fits 3:00.

### 5. Composite in NLE (~3–4 hours)

Manim base + asciinema overlays + voiceover audio. Whatever NLE you prefer (Final Cut / DaVinci / Premiere).

### 6. GitHub URL on the closing card (~5 minutes, needs user)

Currently a placeholder. Decide what URL goes on line 5 of the thesis.

---

## Open decisions (still)

Same as the earlier handovers, unanswered:

1. **Voiceover.** You-recorded vs AI TTS. Recommend you-recorded — it's on-brand.
2. **Music.** None / subtle bed / hard drop at the thesis line. Recommend subtle bed with a silence drop on "lets the model enforce the legality of its own thought."
3. **Logic beat — keep or cut?** Trims ~20s. Algebra alone carries the thesis if runtime is tight.
4. **Hackathon credit card?** "Built with Opus 4.7" 3s plate after the thesis card?

---

## What I'd do next if I had another hour

(Could also be done by you at any time.)

1. **Record asciinema casts of the three demos** with `asciinema rec`. ~10 minutes per take, 2–3 takes per demo for clean ones. Total ~45 minutes.
2. **Render `candidate_full.py` at 1080p60** (`-qh` flag) so we have a sharp version for compositing. ~10 minutes.
3. **Wire the benchmark's free-text wrong-answer output into a manim scene** — pre-typed playback of the chain that ends in x = 4, with the wrong line struck red. ~30 minutes if reusing `LLMProtagonist`.

If you want me to keep going, say which of those three. Otherwise the next blocker is human-in-the-loop work (simulacrum recording, voice, composite) and you're better placed to drive it.

## Re-running the benchmark

```bash
cd /Users/maltelandgren/code/Private/sandbox/orate
.venv/bin/python bench/measure_legal_steps.py        # full suite, ~6 min
.venv/bin/python bench/measure_legal_steps.py --quick # 2 problems, ~1 min
```

Outputs land in `bench/results/legal_steps_<timestamp>.{json,md}`. The markdown is what to paste into the script or any handout.
