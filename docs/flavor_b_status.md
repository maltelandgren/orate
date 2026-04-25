# Flavor-B unification status

**Branch:** `feat/flavor-b-full` (worktree at `/Users/maltelandgren/code/Private/sandbox/orate/.claude/worktrees/flavor-b-full`).
**Tests:** 234 / 234 non-model pass; ruff clean.
**Demos that ran clean on Qwen2.5-7B:** all four (Act-4 algebra Session-based, Act-4 algebra composer-based, Act-4 logic, Act-3 D&D combat).

> Top-line: the architecture refactor we discussed (two-tier `@program` + transition-based grammar composition + `gen.alternative` for composers) is shipped. The Session class is now formally a frozen instance of one specific composer pattern; the composer alternative produces an identical trace against the same model.

---

## What landed (in commit order)

### 1. `program: invocable flag — split @program into leaves vs composers` ([9a96bbe](https://github.com/maltelandgren/orate/commit/9a96bbe))

`@program` gains an `invocable: bool = True` kwarg. Default keeps every existing program a leaf (backwards compat). `invocable=False` declares a composer:

```python
@program(invocable=False)
def dnd():
    while True:
        action = yield gen.alternative([narration, diceroll])
        ...
```

`body_grammar.derive_*` rejects composers up front with `BodyGrammarError("…is a composer…")` — body grammar is meaningless for them since they're not embedded in another program's grammar.

### 2. `session: per-leaf grammars + transition-based driver` ([91c1b6b](https://github.com/maltelandgren/orate/commit/91c1b6b))

The big architectural shift. Before, `_build_outer_grammar` concatenated every leaf's body rules into one monolithic grammar. After, each `_RegistryEntry` carries its own self-contained `body_grammar` field; the outer grammar shrinks to just `text_chunk | "@a(" | "@b("`.

`Session.advance()` does two `sample_under` calls per `@`-emission: prefix, then body. The closing `)` is a literal `engine.append`. The `_handle_call(raw)` method is kept as a backwards-compat shim for tests; `advance()` calls `_dispatch(name, body_text)` directly.

End-to-end verified on Qwen2.5-7B against three Session-based demos — identical traces to before the refactor.

### 3. `gen: Alternative — composer-side leaf dispatch` ([0d15f77](https://github.com/maltelandgren/orate/commit/0d15f77))

`gen.alternative([leaves])` is the composer's primitive for expressing a runtime alternation over leaves. `Alternative.dispatch` does the same transition pattern Session does, but at the composer's scope:

1. Build prefix grammar from leaves' names.
2. `engine.sample_under(prefix)` → `"@<name>("`.
3. Build the picked leaf's body grammar; `engine.sample_under(body)`.
4. `engine.append(")")`.
5. `scan_typed_args(body_text, arg_types)` → typed positional args.
6. Drive the leaf's generator with those args via `.send()` — runs `where=` predicates, collects return value.
7. Return `Picked(name, args, value)`.

Three small refactors landed alongside:

- `body_grammar.scan_typed_args` (was `session._scan_typed_args` — natural home next to body grammar derivation, since it's the inverse).
- `_check_value_against_spec` and `_safe_predicate` added to `gen.py` (the predicate-verification helpers live with the Gen subclasses they switch on).
- `Alternative` and `Picked` exported from `orate`'s top-level `__init__`.

9 new unit tests in `test_alternative.py` using a stub engine.

### 4. `demo: act4_algebra_composer — agent loop via composer @program` ([9e1e45c](https://github.com/maltelandgren/orate/commit/9e1e45c))

The unification end-to-end on Qwen2.5-7B. The whole agent is:

```python
@program(invocable=False)
def solve():
    trace = []
    while True:
        action = yield gen.alternative([algebra_step, done])
        trace.append({"name": action.name, "args": action.args})
        if action.name == "done":
            return {"answer": action.value, "trace": trace}
```

Run as `solve().run(engine=engine)`. No Session, no registry, no `_build_outer_grammar`. Demo output:

```
[step 1]  @algebra_step('3x + 5 = 14', 'combine_like', '3x = 9')
[step 2]  @algebra_step('3x = 9', 'isolate_var', 'x = 3')
[step 3]  @done('x = 3',)
=== Final answer: {'answer': 'x = 3'}
=== Steps taken: 3
```

Identical trace to the Session-based demo. Different wiring.

### 5. `docs/design/two-tier-and-transitions.md`

Architectural ADR describing the decision, what collapsed, what stayed, the migration path, and four open questions for the next round (composer ↔ Session coexistence, composers as leaves, gen.alternative with free-text, make_new_program inside composers).

## What this means for the architecture

Before: `@program` was one tier (leaf only). The Session class held a Python loop that drove the agent; modes were a registry filter; `make_new_program` was special-cased grammar-switching.

After: `@program` is two tiers. Composers (`invocable=False`) own the agent loop in their bodies; leaves carry their own self-contained grammars. The library's job at runtime is to do grammar transitions at `@`-call boundaries — one mechanism, used uniformly for the outer Session loop, for `gen.alternative` inside composers, and for `make_new_program`'s synthesis sub-sample.

The Session class is preserved as a convenience — for "one composer, one loop" the existing API is fine. But it's no longer load-bearing; it's an ergonomic wrapper over `composer.run(engine=engine)`.

## How to verify in the morning

```bash
cd /Users/maltelandgren/code/Private/sandbox/orate/.claude/worktrees/flavor-b-full

# Unit tests
.venv/bin/python -m pytest -q --no-header \
  --ignore=tests/test_xgrammar_engine.py \
  --ignore=tests/test_session_mode.py \
  --ignore=tests/test_gen_against_local.py \
  --ignore=tests/test_arc_data.py \
  --ignore=tests/test_arc_dsl.py \
  --ignore=tests/test_arc_solve.py
# expect: 234 passed

# End-to-end demos (real Qwen-7B; ~30–60s each)
.venv/bin/python examples/legal_steps/act4_algebra_demo.py        # Session-based
.venv/bin/python examples/legal_steps/act4_algebra_composer.py    # composer-based (NEW)
.venv/bin/python examples/legal_steps/act4_logic_demo.py
.venv/bin/python examples/d20/act3_combat_demo.py
```

The composer-based demo is the headline result. Same model, same problem, same predicates — but the agent is one `@program(invocable=False)` body running on the engine directly. No Session class touches it.

## Decisions to make on review

The four open questions in `docs/design/two-tier-and-transitions.md`:

1. Composers retire the Session class, or coexist? My read: coexist. Session is fine ergonomic sugar for one-composer cases.
2. Can a composer be a leaf for another composer? Already works via Flavor-B sub-program yields; just not yet a public idiom.
3. `gen.alternative` with a free-text alternative? Useful for narrative composers; ~30 min to add via a `text_kwarg=`.
4. `make_new_program` inside a composer? Natural composer-native pattern: synthesis appends to a local `available` list, next iteration's `gen.alternative` exposes it. Needs a small refactor to expose synthesis as a callable.

None of these block anything; they're cleanups for the next round.

## How to merge

```bash
cd /Users/maltelandgren/code/Private/sandbox/orate
git checkout main
git merge --ff-only feat/flavor-b-full
git push
```

The branch is purely additive against main and conflict-free.
