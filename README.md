# orate

**Programmatic decoding over local LLM inference.** The developer writes a generator; the model is an oracle consulted at marked yield points. Structured output, tool calls, and agent control flow collapse into one primitive: `yield`.

> Pre-alpha. Built for *Built with Opus 4.7: a Claude Code hackathon* (Apr 21–26, 2026). Problem statement **#2 — Build For What's Next**.

## The argument, in four acts

**Act 1 — Schemas are a ceiling.** Declare a JSON schema, the model fills it. Schemas can't express "a prime number whose digits sum to 10" or "a word that is both a fruit and a color." Types are too weak to carry constraints that live at the value level.

**Act 2 — Predicates move the bar.** A `where=` predicate on a generation moves constraints from the *type* to the *property*. Grammar tightening on reject turns each failed sample into a search step. Correctness becomes a guarantee, not a hope.

**Act 3 — Programs subsume the rest.** A coroutine's `yield` is a decision point. So is a tool call. So is a sub-agent handoff. Why are these three different APIs? They are not. Control flow is first class, and `yield` is the only primitive you need:

```python
@program
def turn(world):
    reasoning = yield gen.string(max_len=120)
    kind = yield gen.choice(["attack", "speak"])
    match kind:
        case "attack":
            target = yield gen.choice(world.enemies)
            damage = yield tool.call(dice.roll, sides=20)
            return {"kind": "attack", "target": target, "damage": damage}
        case "speak":
            line = yield gen.string(max_len=140)
            return {"kind": "speak", "line": line}
```

One KV cache, one engine, one API. Structured output is the trivial case. Tool use is the next yield. Agents are just longer programs.

**Act 4 — The model writes the program.** For a new task, the model authors its own `@program` at runtime — a typed AST of DSL operations, constrained by the same grammar machinery. The program is validated against the task (e.g., the training examples of an ARC-AGI-2 puzzle), grammar-tightened on reject, retried with the grid diff injected as context. The mechanism that filtered scalar values in Act 2 now filters entire programs.

This is the uppercut. **Opus 4.7 is the first model that can reliably synthesize its own constraint programs at inference time.** Smaller models need a lot of hand-holding; Opus 4.7 tears through.

## Demo target

ARC-AGI-2 on a curated subset. The visible loop:

1. Opus 4.7 is shown three input→output grid demonstrations.
2. It emits a `@program` — a transformation rule over a small grid DSL, grammar-constrained.
3. The program is executed on the demonstrations. Mismatch → reject → grammar tightens, the offending case is surfaced in natural language, retry.
4. On convergence, the rule runs on the test input. Grid appears.

Honest framing: passing the demonstrations does not *guarantee* the test output — ARC is underspecified on purpose. What's guaranteed is that we never emit a rule *inconsistent with the demonstrations*.

## Install (dev)

```bash
pip install -e ".[dev]"              # kernel only
pip install -e ".[local,dev]"        # + llama-cpp-python + xgrammar (local inference)
pip install -e ".[api,dev]"          # + anthropic (Opus 4.7 as engine)
```

## Status

Kernel scaffold is in place. Engine adapters (XGrammar, Anthropic tool-use) land next. See commit history; this repo is a live build.

## License

MIT. See [LICENSE](LICENSE).
