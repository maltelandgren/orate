# Act-4 meta finisher — Qwen-7B end-to-end trace

- **Model:** `Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- **Date:** 2026-04-25 15:15
- **Decoding:** deterministic argmax under masked logits.
- **Runner:** [`examples/legal_steps/act4_meta_finisher.py`](../../examples/legal_steps/act4_meta_finisher.py)

## Setup

- Pre-registered: `@algebra_step` (linear-equation rules), `@done`.
- `@make_new_program` always available.
- `allow_free_text=False` — every emission is a structured @-call.
- Problem: solve `x^2 - 5x + 6 = 0`.
- Hint in the prompt: `@algebra_step`'s rules don't include factoring,
  so use `@make_new_program` to author a primitive with exactly four
  yields capturing the quadratic structure.

## Trace

```
@make_new_program("quadratic_solver", "Solve a quadratic equation
                                       axx2 bx c 0 and return roots
                                       and discriminant")

[grammar-switch to PROGRAM_SOURCE_GRAMMAR; runtime samples source]

@program
def quadratic_solver():
    _equation_value = yield gen.boolean()
    _root1 = yield gen.boolean()
    _root2 = yield gen.boolean()
    _discriminant = yield gen.boolean()
    return {"equation": _equation_value,
            "root1": _root1,
            "root2": _root2,
            "discriminant": _discriminant}

[validator: OK]
[compile: OK]
[register: OK; outer grammar rebuilt to include @quadratic_solver(]

@quadratic_solver(True, True, True, True)
  → {'equation': True, 'root1': True, 'root2': True, 'discriminant': True}

@done("2 = 2, 3")
  → {'answer': '2 = 2, 3'}

[turn end: ends_turn]

Registry at session end: ['algebra_step', 'done', 'make_new_program',
                          'quadratic_solver']
```

## What the demo proves (and doesn't)

**Proves the round-trip end-to-end.** Same KV, no API hops.

1. The model recognised `@algebra_step` doesn't fit a quadratic.
2. The model emitted `@make_new_program("quadratic_solver", ...)`.
3. The runtime grammar-switched to `PROGRAM_SOURCE_GRAMMAR` and
   sampled a body source.
4. The body validated (return dict references only bound names —
   `_equation_value, _root1, _root2, _discriminant`, all assigned by
   the yields above).
5. The runtime compiled it, registered it, and rebuilt the outer
   grammar. From this point `@quadratic_solver(...)` is a callable
   leaf in the same session.
6. The model invoked the new tool: `@quadratic_solver(True, True, True, True)`.
7. The model wrote `@done("2 = 2, 3")` — the actual roots of
   `x² - 5x + 6` are 2 and 3, so the model "found" them in the @done
   string even though the typed args don't carry that information.

**Doesn't prove the model picked a useful schema.** Qwen-7B picked
`gen.boolean()` for every yield — the smallest body grammar (one
token: "true" or "false"). The four args `(True, True, True, True)`
carry no quadratic information. The model could have picked
`gen.integer(...)` to express coefficients and roots; it didn't.
This is a Qwen-quality issue, not a system one. Mitigations on the
list:

  - Restrict the meta grammar's `gen-call` to integer-only for math
    domains. ~1 line of grammar change.
  - Bias the few-shot example more aggressively toward integer
    yields (already done in v4 but still wasn't enough).
  - Run with a stronger model (Opus 4.7 for the finisher beat would
    pick informative types).

## What this means for the video

The meta-authorship beat is real and reproducible. The on-screen
moments are honest:

1. Cut to the problem. Pre-registered tools shown in the registry panel.
2. Model emits `@make_new_program("quadratic_solver", "...")`.
3. Source materialises on screen via grammar-switched sampling.
4. Validator + compiler + registrar fire (~100ms each, hide as a
   single "[+tool] quadratic_solver" callout).
5. Model invokes the new tool.
6. `@done` lands the answer.
7. Registry pull-back: `quadratic_solver` is in the list. It wasn't
   when the video started.

The visual focus is the **source materialising on screen**. That's
the punchline. The boolean choice is footnote-honest, not a load-
bearing claim.

## Reproducing

```bash
.venv/bin/python examples/legal_steps/act4_meta_finisher.py
```

The decoding is deterministic argmax. Same model + same prompt
should give the same trace.
