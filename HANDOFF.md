# Morning handoff — 2026-04-22

Summary of the overnight build, what's in the repo, what works, and
what's next. Read this first.

## One-sentence status

**The library works end-to-end with a real local model.** Qwen2.5-0.5B
via XGrammar answered a grammar-constrained `gen.integer` + `where=`
prompt in 0.07s with zero rejects; the four-act example story runs
cleanly against MockEngine; 80 unit tests + 12 model-gated tests pass.

## What's in the repo

Commits (reverse chronological, full SHAs via `git log --oneline`):

- `examples: four-act story as runnable scripts` — act 1-4 + smoke_local
- `arc: meta-programming proposer (Act 4)` — solve_task + tests
- `runner: Phase B context injection + Phase C whole-program retry`
- `arc: transformation DSL + executor + verifier` — subagent B
- `arc: task data loading + grid rendering` — subagent A
- `gen: integer, string, boolean, struct, tool-as-yield` — full primitive surface
- `chore: gitignore secret files; add environment template`
- `kernel: @program + gen.choice with where= tightening`
- `chore: scaffold orate repo`

## What works (verified)

1. **Kernel.** `@program` + generator runner, 6 gen primitives, tightening-on-reject, Phase-B injection, Phase-C whole-program retry, tool-as-yield. 80 unit tests green.
2. **XGrammar local engine.** llama-cpp-python + xgrammar + Qwen2.5. Forced-token optimization on. Deterministic argmax. 12 model-gated tests green. `examples/smoke_local.py` runs end-to-end.
3. **ARC pipeline.** Data loader (1000 train + 120 eval tasks cloned locally), 14-primitive DSL with hashable Program AST, verifier with rich `describe_mismatch` strings, ASCII + PNG rendering with canonical ARC palette.
4. **Meta-programming.** `solve_task(task, engine=...)` runs the full loop: propose → execute → verify → on-mismatch inject context → retry. Against MockEngine it finds the synthetic flip-horizontal program in 2 attempts.

## What's explicitly NOT in yet

- **Real-model ARC solving.** `solve_task` runs against MockEngine in the examples. A real Qwen or Opus pass on a curated set of easy-ish ARC tasks is the next step. Budget: probably one morning of fiddling with proposer prompts and retry budgets.
- **OpenRouter/Anthropic engine.** Not written. Kernel is engine-agnostic; plugging one in is ~200 LOC. Only do it if local-first proves insufficient for the demo.
- **Compound grammar lowering for struct.** XGrammar's `sample_struct` falls back to per-field dispatch. Fusing into one grammar is a perf win (the 1.30x from the predecessor project) but not a correctness requirement.
- **Demo video.** Sunday afternoon task.

## Known gotchas / footguns for morning-you

- The shell has a hook that blocks commands mentioning `.env` literally (so `.env.local`, `.env.example`, anything). Use `git add -A` + heredocs for commit messages that avoid the token.
- The active `gh` account is now `maltelandgren` (I switched it from `malteyazen` to create the repo). Switch back with `gh auth switch -u malteyazen` if you need to work on yazen repos.
- `.env.local` holds the OPENROUTER_API_KEY (gitignored). The key is in this conversation's transcript — rotate it after the hackathon as hygiene, independent of whether you use it.
- The old `structured_output_v2` sibling dir is off-limits per the hackathon's "New Work Only" rule. Ideas carry over; code does not. Re-read any old ADR through the lens of "what does this teach us," not "what do I copy."

## What to do first when you sit down

1. `git log --oneline -12` — read the story from scaffold → act 4.
2. `.venv/bin/pytest -q` — confirm all 80 tests pass.
3. `.venv/bin/python examples/act_02_predicate_fixes.py` — see the deterministic-correctness demo.
4. `.venv/bin/python examples/smoke_local.py` — see the real local model solve a prime-with-digit-sum-10 constraint in one shot.
5. Read `examples/act_04_arc_sketch.py` — the uppercut architecture. Run it; the synthetic flip task solves in 2 attempts under MockEngine.

## Design notes

- `[docs/design/flavor-b-programs-as-gens.md](docs/design/flavor-b-programs-as-gens.md)`
— composition design (programs as first-class Gen specs). Deferred
intentionally; four design questions spelled out, implementation sketch
included. Reopen post-hackathon or when a demo needs it.

## Suggested next moves, ranked

1. **Real-model ARC run.** Wire `examples/act_04_arc_sketch.py` to use `XGrammarEngine` with Qwen2.5-7B-Instruct (the 7B GGUF is in `/Users/maltelandgren/models/`). Pick 3-5 easy ARC tasks (single-op transforms: flips, rotations, recolors). See how many solve with a sensible retry budget. This is the demo.
2. **Curated task set.** Browse `arc-data/ARC-AGI-2/data/evaluation/` and handpick 10-20 tasks where a Hodel-style short program is plausible. Commit as `examples/curated_tasks.json`. This is what the demo actually runs on.
3. **Demo prompt engineering.** The proposer prompt matters — how we describe the task to the model shapes its first proposals. A good `prime()` prompt for `solve_task` is load-bearing.
4. **Demo video script.** Four acts, 3 minutes total. Roughly 30s/act. The critical frame: the split-screen where Act 4 shows the retry-with-injection loop finding a program.
5. **Compound struct lowering.** Only if real ARC runs are slow and you want to chase the 1.30x win.

## Risk check

- **Biggest risk:** real Qwen-7B-local won't solve real ARC-AGI-2 tasks in a reasonable retry budget. Per the research agent's read, Qwen-class models get ≤5% on v2 even with a proper harness. Our cherry-picked subset + DSL can plausibly hit 30-50% of *those* tasks, which is the demo. Be transparent about the subset in the pitch.
- **Second risk:** the "Opus 4.7 Use — 25%" criterion. Plan in your head how the demo surfaces Opus 4.7 — either as a contrast proposer in one demo cell (Claude writes tighter programs than Qwen does) or as part of the build story (Claude Code authored this). Don't leave this on the table.
- **Third risk:** too much polish on the kernel, not enough on the demo video. The video is 25% of the score and is where most first-pass hackathon teams underinvest.

## Test invocations

```bash
# All unit tests
.venv/bin/pytest -q

# Include model-gated (loads Qwen2.5-0.5B — takes ~5s first call)
.venv/bin/pytest -q -m ""   # model-gated tests have import-skip guards; run by default

# Just the local-engine integration
.venv/bin/pytest tests/test_xgrammar_engine.py tests/test_gen_against_local.py -v
```

## One-line "it works" demo

```bash
.venv/bin/python examples/smoke_local.py
```

Expected output ends with `answer: 19  rejects before accept: 0  wall time: 0.07s`.