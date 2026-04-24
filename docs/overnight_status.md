# Overnight status — Apr 24/25

**Branch:** `feat/act-4-capabilities` (worktree at `../orate-act4`, pushed to origin)
**Tests:** 219 / 219 non-model pass; ruff clean.
**Demos that ran clean on Qwen2.5-7B:** Act-4 algebra, Act-4 logic, Act-3 D&D combat.

> Top-line: every capability the script storyboard depends on is shipped and demonstrated against the local 7B model. Three end-to-end runs in `/tmp/algebra_run_9.log`, `/tmp/logic_run_1.log`, `/tmp/d20_run_1.log` (won't survive a reboot — re-run if you want fresh ones).

---

## What landed (in order of commits)

### 1. `session: mode-aware registry + typed-arg parsing` ([11bd525](https://github.com/maltelandgren/orate/commit/11bd525))

- **Mode-aware registry.** `_RegistryEntry.mode` controls visibility (`None` = visible everywhere, else only when `Session._active_mode` matches). `Session.register(name, fn, *, mode="combat")` is the new surface.
- **`mode_transition`.** `@program(mode_transition="combat")` makes a tool flip the session into the named mode after a successful invocation. `Session.set_mode("combat")` is the explicit form. `make_new_program` stays unscoped (mode=None) so it's available everywhere.
- **Typed args via `derive_call_arg_types`.** New helper in `body_grammar.py` returns `[ArgType(kind="choice"|"integer"|"string"|"boolean", ...)]` per yield. Stored on `_RegistryEntry.arg_types`. Drives the new typed scanner so `@foo("hi", 42, true)` decodes to `("hi", 42, True)`, not three strings.
- 28 unit tests in `test_session_modes_typed.py` — mode round-trip, type derivation per gen.* method, escape handling, comma-in-string parsing.

### 2. `session: predicate verification on @-call emissions` ([fe976d5](https://github.com/maltelandgren/orate/commit/fe976d5))

- **The big one.** Closed the gap between "the body grammar enforces shape" and "the `where=` predicate enforces meaning." `Session._verify_program_emission` re-runs the program body against the parsed args, dispatching each yield's `Gen` against its corresponding value. Cross-yield closures (`where=lambda s: f(prev_yield, s)`) work because the verifier `.send()`s each parsed value back into the body iterator before the next yield is constructed.
- **Predicate semantics:** Choice (membership + where), Int (range + where), String (where), Bool (where). On rejection: `[session: rejected — yield #N: ... failed where= predicate. Retry the call.]` is appended to the KV; the next sample sees the failure as context.
- **Body grammar relaxations.** `where=` / `description=` / `max_retries=` / `reject_message=` are now accepted (and ignored) on every gen.* method — they're runtime-only metadata. Lambdas inside kwarg expressions and a leading docstring are also tolerated.
- **`equivalent_under` (SymPy).** Algebra checker with five rules. Accepts both same-form equivalence AND scalar multiples (so `2x = 8 ↔ x = 4` is legal). Per-rule sanity: `evaluate` requires a numeric side, `isolate_var` requires a Symbol-only LHS.
- **`derivable_under`.** Propositional checker covering modus ponens, modus tollens, hypothetical syllogism, conjunction, simplification.
- 25 checker tests + 3 verification integration tests.

### 3. `demos: legal-step Act-4 programs (algebra + logic)` ([26182cc](https://github.com/maltelandgren/orate/commit/26182cc))

- `examples/legal_steps/algebra.py` — `@algebra_step(before, rule, after)` with cross-yield `equivalent_under` predicate.
- `examples/legal_steps/logic.py` — `@inference_step(premises, rule, conclusion)` with `derivable_under` predicate (premises are `;`-separated).
- 11 integration tests in `test_legal_step_programs.py` — including a full chain-of-four solve through `_handle_call`'s predicate-verification path.

### 4. `session: tool-only mode + bare-choice scanner` ([7f447c4](https://github.com/maltelandgren/orate/commit/7f447c4))

Two fixes uncovered while running against Qwen-7B:
- **Bare-choice scanner.** Body grammar emits `gen.choice` options bare (the GBNF terminal `"substitute"` matches the literal word). The scanner was treating choice the same as gen.string (JSON-quoted). Now choice scans bare-word against the longest matching option.
- **`allow_free_text=False`.** New Session kwarg sets the outer grammar to just `at_call` (no `text_chunk` alternative). When the model sees a session note in context (e.g. on a parse error), it tends to learn the pattern and fill turns with hallucinated session-shaped text. Tool-only sessions reject text outright. Default stays True.

### 5. `algebra demo: clean end-to-end on Qwen-7B` ([489c98c](https://github.com/maltelandgren/orate/commit/489c98c))

`@done(answer)` ends_turn=True tool + tightened SYSTEM prompt with verbatim worked example. `substitute` rule dropped from the model's choice (substitution is now expressed by baking it into `before`, since the predicate is stateless). Demo run:
```
[ok]  @algebra_step('3x + 5 = 14', 'combine_like', '3x = 9')
[ok]  @algebra_step('3x = 9', 'isolate_var', 'x = 3')
[ok]  @done('x = 3',)
[turn end: ends_turn]
```

### 6. `logic demo: clean end-to-end on Qwen-7B` ([57565cd](https://github.com/maltelandgren/orate/commit/57565cd))

Same pattern: `@qed("conclusion")` ends_turn sibling. Demo run:
```
[ok]  @inference_step('A -> B; A', 'modus_ponens', 'B')
[ok]  @inference_step('B -> C; B', 'modus_ponens', 'C')
[ok]  @qed('C',)
[turn end: ends_turn]
```

### 7. `demos: Act-3 D&D combat — mode-switch + composed NPCs on Qwen-7B` ([49ffcbd](https://github.com/maltelandgren/orate/commit/49ffcbd))

- `examples/d20/characters.py` — three NPC `@program`s (`aria_attack`, `hooded_figure_attack`, `borin_attack`), each with its own action set and damage cap. `@enter_combat` (mode_transition="combat") + `@exit_combat` (mode_transition="default").
- Demo run:
```
[ok]  @enter_combat('aria',)              → mode: combat
[ok]  @hooded_figure_attack('dagger', 'aria', 2)
[ok]  @aria_attack('longsword', 'hooded_figure', 3)
[ok]  @borin_attack('shield_bash', 'hooded_figure', 4)
[ok]  @exit_combat('victory',)            → mode: default
```

---

## What didn't land (and why)

- **`ends_turn=True` client-execution loop.** Marked stretch in the plan; we skipped it because the inline tools (`@done`, `@qed`, `@exit_combat`) achieved the demo goals. Today's `ends_turn` ends the assistant turn but doesn't yet hand structured args to a client for external execution + result injection. ~3 hr of work, scoped for after the deadline.
- **Algebra problem difficulty escalation.** Original plan was "escalate the system until Qwen-7B free-text reliably slips." We didn't run that experiment because the simpler 1-variable problem (`3x + 5 = 14`) gave a clean run that lets the predicate-verification beat read clearly. For the video, the contrast shot — free-text Qwen vs. constrained Qwen on the same problem — would need the harder problem; my recommendation in [§2 below].

## What you might want to look at first

1. The three `/tmp/*_run_*.log` files — those are the literal demo outputs the video should reproduce. Re-run the demo scripts to regenerate.
2. The branch's commit log: 7 commits, each self-contained with a long-form message. Reviewable in 10 minutes.
3. `tests/test_session_modes_typed.py` and `tests/test_legal_step_programs.py` — the new test files. Total +63 tests, all pass.

## Decisions to make before filming

1. **Algebra demo problem.** Currently `3x + 5 = 14` (one variable, two steps + done). Reads cleanly but the "free-text fails" contrast is weak — Qwen-7B solves this in free text most of the time. **Options:**
   - Stick with simple, sell the predicate-verification beat alone (easier shot, less compelling).
   - Escalate to a 2-variable system and accept 1–2 rejection events in the trace before the model lands the right step. The `[REJ]` events are visually informative.
   - Show a contrast shot: free-text Qwen on a slightly harder problem (we can pick one where it's known to slip ~50% of seeds) → fails. Same Qwen under `@algebra_step` → succeeds. Picking the right problem needs ~30 min of trial.
2. **D&D demo length.** The `act3_combat_demo` is hard-coded to one round per character + exit. Easy to extend (the model wants to keep going — see how it tries a second `@enter_combat` after exit). Up to you whether the video wants one round or two.
3. **Voiceover, music, GitHub URL.** Same questions as before, still unanswered.

## How to re-run demos in the morning

```bash
cd /Users/maltelandgren/code/Private/sandbox/orate-act4
.venv/bin/python examples/legal_steps/act4_algebra_demo.py
.venv/bin/python examples/legal_steps/act4_logic_demo.py
.venv/bin/python examples/d20/act3_combat_demo.py
```

Each takes ~30–60s on the 7B model after the initial load (~10s).

## How to merge feat/act-4-capabilities

```bash
cd /Users/maltelandgren/code/Private/sandbox/orate
git checkout main
git merge --ff-only feat/act-4-capabilities  # should be FF since main hasn't moved
git push
```

Or, since the branch is purely additive: `git push origin feat/act-4-capabilities:main` after a final review pass. The merge is conflict-free against main as of `f31f368`.
