# ADR: Two-tier `@program` and transition-based grammar composition

**Status:** shipped on `feat/flavor-b-full` (Apr 25).
**Supersedes:** the implicit single-tier model where every `@program` was a leaf, and the Session class held a Python loop owning the agent's control flow.

## Context

`orate` started with one kind of `@program`: a generator function whose body is a straight-line sequence of `var = yield gen.X(...)` yields, terminated by `return`. The body grammar is AST-derivable, so any program can be inlined into a parent's call-site grammar as `@name(args)`. The runtime mints a single monolithic outer grammar that concatenates every registered program's body rules under a shared `at_call` production.

The Session class wraps an engine + a registry of `@program`s + a Python `while` loop that samples the outer grammar repeatedly, dispatches `@`-emissions, and tracks state (mode, ends_turn, mode_transition).

Two pressures converged:

1. **`make_new_program`**, the bootstrap that lets the model author its own programs, already breaks the "one monolithic outer grammar" picture: when invoked, it does a *separate* `sample_under(PROGRAM_SOURCE_GRAMMAR)` call to get the synthesised source. We had two patterns coexisting: inline-everything-into-one-grammar (for normal leaves) and grammar-transition-at-boundary (for `make_new_program` only).

2. **Composers** (`@dnd`, `@combat` from the script discussion) need control flow — loops, conditionals, mutable state. The single-tier model rejected them because body-grammar derivation can't handle a `while True:`.

The user pushed back on my framing of "the loop has to live in the driver" (Python state outside any `@program`) by pointing out that with full Flavor B + control flow + sum types, the loop belongs *inside* a top-level `@program`. The Session class wasn't architecturally distinct from a frozen Python loop; it was just *the* loop, hard-coded.

## Decision

**Two tiers of `@program`:**

- **Leaf** (`@program` or `@program(invocable=True)`): bounded straight-line yields, derivable body grammar. Inlined as `@name(args)` in another grammar. Default for backwards compat — every existing `@program` is a leaf.
- **Composer** (`@program(invocable=False)`): arbitrary Python control flow, no body-grammar derivation. Run via `.run(engine=engine)`; orchestrates leaves via `gen.alternative([…])`.

**Transition-based grammar composition:**

Each leaf carries its own self-contained GBNF. The "outer grammar" (whether at the Session level or inside `gen.alternative`) is just a prefix-alternation:

```
root ::= text_chunk | "@a(" | "@b(" | ...
```

The matcher accepts the moment a prefix is consumed. The driver then samples the picked leaf's body under *its* grammar (a separate `sample_under` call), appends the literal `)`, and dispatches.

This is the same pattern `make_new_program` already used; we generalised it.

**`gen.alternative([leaves])` is the composer-side primitive:**

```python
@program(invocable=False)
def solve():
    while True:
        action = yield gen.alternative([algebra_step, done])
        if action.name == "done":
            return action.value
```

`Alternative.dispatch` does the same transition pattern at the composer's scope: build prefix grammar, sample, identify leaf, sample body, append `)`, parse args, drive the leaf's generator with the parsed args (running `where=` predicates), return `Picked(name, args, value)`.

## What collapsed

The Session class is now a frozen instance of one specific composer pattern. With composer + `gen.alternative`, the equivalent agent is:

```python
engine.begin_session(SYSTEM)
engine.append(USER_TURN)
result = solve().run(engine=engine)
```

For the `act4_algebra_demo`, the composer version produces an identical trace on Qwen-7B:

```
[step 1]  @algebra_step('3x + 5 = 14', 'combine_like', '3x = 9')
[step 2]  @algebra_step('3x = 9', 'isolate_var', 'x = 3')
[step 3]  @done('x = 3',)
=== Final answer: {'answer': 'x = 3'}
```

Mode switching becomes plain Python control flow. `make_new_program`'s registry mutation becomes a mutable Python list inside the composer's scope (e.g. `available.append(new_leaf)`). Nothing special.

## Why this is an architectural improvement

1. **One pattern, not two.** `make_new_program`'s grammar transition and a normal leaf's grammar transition are now identical mechanism. The runtime drives transitions at boundaries; leaves don't know they're being composed.

2. **Locality of compilation.** Adding a leaf only compiles that one body grammar. The outer grammar's recompile is a string concatenation of prefixes — cheap. No more "regenerate the entire monolithic grammar on every register()."

3. **Locality of state.** Each leaf's matcher is independent. One leaf cannot affect another's grammar via shared rule names.

4. **Mental model matches reality.** The user writes a leaf's schema in isolation. The library composes at runtime via prefix transitions. The user's "I just write the schema" expectation is now true.

5. **Composers are first-class.** Loops, conditionals, mutable state, sub-program yields — all available inside `@program(invocable=False)`. The agent's control flow lives in Python where Python belongs, owned by a `@program`.

## What stays

- `@program` as the public decorator (just gains an `invocable=` kwarg).
- `gen.choice/integer/string/boolean/struct/tool/alternative` as yield primitives.
- `body_grammar.derive_*` (now refuses composers up front).
- Predicate verification on every `@`-emission. The Session class still does it for its dispatch; `Alternative.dispatch` does it for composer dispatch.
- `Session` class itself — kept for the Session-based demos that still work; will retire when we port the remaining demos to composer form.

## Migration path for existing demos

- `examples/legal_steps/act4_algebra_demo.py` (Session-based) — unchanged, still works.
- `examples/legal_steps/act4_algebra_composer.py` (composer-based) — sibling, demonstrates the new pattern.
- `act4_logic_demo` and `act3_combat_demo` will get composer counterparts in subsequent commits. The combat demo's mode switching becomes a `if action.name == "enter_combat": yield combat_loop()` branch.

## Engineering shipped

```
program.py      +  invocable kwarg + is_invocable() helper
body_grammar.py +  scan_typed_args (moved from session.py)
                +  _reject_composer guard on derivation entry points
gen.py          +  Alternative + Picked + alternative() constructor
                +  _check_value_against_spec + _safe_predicate
                   (composer-side predicate verification, mirrors Session's)
session.py      ↻  per-leaf body_grammar field on _RegistryEntry
                ↻  _build_outer_grammar emits prefix-only alternation
                ↻  advance() drives transitions: sample prefix → sample body
                   → append ")" → _dispatch(name, body_text)
                +  _dispatch separated from _handle_call (the latter kept
                   as a backwards-compat shim for tests)
__init__.py     +  Alternative, Picked exported
tests           +  3 new body-grammar composer-rejection tests
                +  9 new gen.alternative tests
```

End-to-end verified on Qwen2.5-7B: Session-based algebra demo, Session-based logic demo, Session-based D&D combat demo, AND composer-based algebra demo all run cleanly under the new architecture. 234 unit tests pass; ruff clean.

## Open questions for the next round

1. **Do composers retire the Session class entirely, or do they coexist?** Argues for coexistence: the Session class is a simple ergonomic wrapper for "one composer + one loop"; users who don't want to write a composer body get it for free. Argues against: two ways to express the same thing is friction.
2. **Can a composer be a leaf for another composer?** Conceptually no (composers don't have call-site grammars). But you could have a leaf that *internally* yields a composer — the composer runs to completion within the leaf's body. This already works via Flavor-B sub-program yields; the question is whether to surface it as a public idiom.
3. **`gen.alternative` with free-text alternative?** Today it's leaves-only. Composers that need narration would benefit from `gen.alternative([narration_leaf, ...])` where `narration_leaf` is a leaf that takes a single `gen.string`. Or we add a `text=` kwarg that's syntactic sugar for the same thing.
4. **`make_new_program` inside a composer?** With composers having mutable Python state, `make_new_program` could append to the composer's local `available` list and the next `gen.alternative([*available])` includes the new leaf. This is the natural composer-native version of registry mutation. Needs a small refactor to lift the synthesis logic out of `_handle_make_new_program` into a callable composers can invoke.
