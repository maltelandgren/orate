# Video direction — afternoon review notes

User feedback after looking at last night's exploration. Capture +
correction for whoever picks up the manim work next.

## Direction call

- **D1 paper is the look.** The warm minimalist style reads best.
  Lean into it.
- The 25s `candidate_full.py` already uses D3 (paper + terminal-dark
  inset). User's phrasing was "I like the d1 paper style" — so the
  inset framing is fine, but the dominant aesthetic is D1. Don't
  over-darken.

## Strong segment to build on

`v2_protagonist_open.py` is the keeper. Specifically the structure:

1. The LLM box appears, emits a few tokens, opens its logits, picks one.
2. We pull back: "we've gotten good at *shaping* those distributions"
   — chips for prompts, fine-tuning, structured output, tool calls.
3. Run A vs Run B side-by-side: one ends in ✓, one in ✗, same
   model, same prompt.
4. "orate closes the gap."

Use this opening (or something very close) for the final video. The
"protagonist is the model, camera on the box from frame 1" framing
beats the script's current Act 1 (which leans on the simulacrum pain
externally).

## Technical accuracy correction (non-negotiable for Beat 2)

The current chip list in `v2_protagonist_open.py` Beat 2 reads:
> "prompts", "fine-tuning", "structured output", "tool calls"

Followed by:
> "But shape and effect aren't the same as legality."

The framing is roughly right but **soft**. Sharpen it. The actual
distinction is:

> Prompts and fine-tuning shape the *distribution*.
> Structured output and tool calling constrain the *type* (they
> control the shape of what comes out — JSON, schema-conformant —
> and the side effect of a tool call).
> But **logic** has always lived outside the decoder. Predicates
> over the value, semantic constraints, "this step must be valid
> under SymPy" — none of that is something the decoder sees. It
> happens in user code, after the fact, with retries.
> orate folds that logic *into* the decoder.

Recommended replacement for the sub3 line:

> "Type, not logic. Logic has always lived outside the decoder."

And for the post-chip beat (between Beat 2 and Beat 3), a tighter line:

> "orate puts logic inside the decoder."

The Beat 3 contrast (Run A ✓ vs Run B ✗ on the same problem) is the
visual proof — keep it. The benchmark backs this with concrete
numbers (`bench/results/legal_steps_2026-04-25_1200.md`): on
`3x + 5 = 14`, free-text Qwen-7B says x = 4 (wrong); the same model
under `@algebra_step` says x = 3 (correct).

## What to update in the manim files

1. `v2_protagonist_open.py` — replace the Beat 2 sub3 line with the
   sharpened "Type, not logic" framing. Maybe split into two beats
   so the distinction lands.
2. Consider lifting the V2 opening into `candidate_full.py` as the
   replacement for the current Act 1. The simulacrum pain is the
   author's ground truth, but the protagonist-first opening is more
   visceral and doesn't require the viewer to know what simulacrum
   is.
3. The Run A / Run B side-by-side in V2 currently uses `2(5-y) + 3y
   = 12 → 10 + y = 12 → y = 2 ✓ / y = 4 ✗`. Keep this as a stylised
   illustration *or* swap to the actual benchmark contrast
   (`3x + 5 = 14 → x = 3 ✓ / x = 4 ✗`) for grounded honesty. Author
   judgement call — the benchmark version is more defensible if
   anyone asks "is this a real run?"

## Demo runners now available for asciinema overlays

Latest as of merge to main this afternoon:

- `examples/d20/act3_full_demo.py` — narration + roll + combat one KV
- `examples/legal_steps/act4_algebra_composer.py` — Act 4 Beat 1 (algebra)
- `examples/legal_steps/act4_logic_composer.py` — Act 4 Beat 2 (logic)
- `examples/legal_steps/act4_meta_finisher.py` — Act 4 Beat 3 finisher

These are the canonical paths that produce the traces the video
overlays will reference.

## Honest scope (don't oversell on screen)

The Beat 3 finisher shows the model authoring a typed schema (no
`where=` predicate yet). The structure is grammar-bound; the math
correctness rides on the model. Predicate-bound model-authored
programs need a `where=<lib_predicate>` extension to
`PROGRAM_SOURCE_GRAMMAR` — on the JIT segmentation roadmap, not
shipped today. If a manim caption claims "the model writes
predicates," that's wrong. "The model designs its own data type" is
right.
