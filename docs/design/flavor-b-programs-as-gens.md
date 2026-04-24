# Design note — Flavor B: programs as first-class Gen specs

**Status:** Deferred, intentional. Good idea, wrong time to ship it.
**Decided:** 2026-04-24 during design discussion.
**Would re-open when:** the hackathon is over, or a demo script specifically needs composed validators and `yield from` isn't clean enough.

## The idea in one paragraph

A `@program` should be usable *in place of* a `Gen` spec — yielded like `gen.choice(...)` or `gen.integer(...)`. This turns every `@program` into a reusable, self-contained, introspectable constraint block that slots into a larger program the same way a helper function slots into a larger function. Composition becomes the default story for building libraries of named constraint shapes.

## What the API would look like

```python
from orate import program, gen

@program
def valid_email():
    local  = yield gen.string(pattern=r"[a-z0-9.]+")
    _at    = yield gen.choice(["@"])
    domain = yield gen.string(pattern=r"[a-z0-9]+\.[a-z]{2,}")
    return f"{local}@{domain}"

@program
def us_zip():
    digits = yield gen.string(pattern=r"[0-9]{5}")
    return digits

@program
def signup():
    name  = yield gen.string(max_len=40, description="full name")
    email = yield valid_email                          # <-- sub-program as Gen
    zip_  = yield us_zip                               # <-- another one
    age   = yield gen.integer(13, 120)
    return dict(name=name, email=email, zip=zip_, age=age)
```

Each `@program` can carry its own `description=`, `whole_program_retries=`, and
`reject_message=`. The outer program sees the composed result as the yield's
value. Source-in-prompt prettier too: the model sees composed programs the way
the author wrote them, recursively.

## Why it's lovely

1. **Constraint libraries become first-class.** An ecosystem of named
   `@program` specs (`email`, `us_address`, `uuid4`, `iso8601_date`,
   `grid_of_dimensions(h, w)`) — composable like types.
2. **Introspection is uniform.** The source-in-prompt machinery that already
   walks a top-level `@program` can recurse into composed sub-programs and
   show the full structure to the model.
3. **Semantics matches Python intuition.** A `@program` is a generator that
   returns a value; yielding it and getting that value back is what you'd
   expect from any other language with first-class generators.
4. **Retry budgets compose.** A brittle sub-constraint (`valid_email` with
   `whole_program_retries=3`) can retry *inside* a larger program without
   exhausting the outer budget.

## Why it's deferred

Not because it's hard — because it's easy to ship with subtle corners that
bite later.

### The four design questions that must be answered before shipping

#### 1. Retry-budget composition

If `valid_email` has `whole_program_retries=3` and `signup` has
`whole_program_retries=10`, and a yield inside `valid_email` rejects —

- **(a) inner-first:** consume `valid_email`'s 3 first; on its exhaustion
  bubble a `ProgramRejected` up to `signup`, consume one of its 10.
- **(b) outer-only:** ignore inner budget; any rejection walks back to
  `signup` immediately.
- **(c) combined:** sub-program inherits caller's remaining budget.

Leaning: **(a) inner-first.** Matches the function-call mental model and
keeps sub-programs self-contained. But it needs to be documented clearly
so users don't expect (b).

#### 2. Phase-B context isolation

When `valid_email`'s `gen.string` rejects and `_notify_reject` injects a note
into the engine session — does that note stay visible to yields *outside*
`valid_email`?

- **(a) global:** today's behavior, context accumulates across the whole run.
  Simple. But sub-program implementation details leak upward.
- **(b) scoped:** push/pop a context frame on sub-program entry/exit, drop
  inner notes when the sub-program completes.
- **(c) hybrid:** notes stay scoped by default; the sub-program can "publish"
  a summary note upward if it chooses.

Leaning: **(b) scoped.** Sub-programs are libraries; their failure modes are
not the caller's concern. Implementation: engine grows `push_context_scope()`
/ `pop_context_scope()`, runner bookends sub-program dispatch with those.

#### 3. Phase-C rewind scope

If the *outer* program rejects and rewinds, does the sub-program's state
get rewound too?

- Yes, cleanest. But means tracking invocation boundaries in the trace and
  re-running any sub-program calls from the top on the outer retry.
- Related: if the sub-program has already produced a return value that the
  outer program stored in a local variable, how is that local unwound? (Answer:
  outer Phase-C already re-invokes the whole outer body; locals are reset
  with it. Just need sub-program's engine-visible state to reset too.)

Leaning: **full rewind on outer retry**, matching "re-invoke the outer body"
semantics we already have.

#### 4. Verifiers on composed programs

If both `valid_email` and `signup` have attached verifiers — when does
`valid_email`'s verifier run?

- **(a) on sub-program return:** verifier fires immediately, local to the
  sub-program. A fail triggers inner Phase-C retry first.
- **(b) at outer boundary:** all verifiers (sub and outer) run once, when the
  outer program returns.

Leaning: **(a) on sub-program return.** Keeps sub-programs self-validating.
Matches the analogy "function checks its own postconditions before returning
to the caller."

### Other reasons it's deferred

- `yield from helper()` already covers ~80% of the practical use cases for
  factored helpers. You lose introspection-at-the-@program-boundary and the
  per-sub-program retry budget, but you keep composition.
- Nothing in the four-act pitch requires Flavor B. Acts 1-3 use `gen.*`
  directly; Act 4's meta-programming composes at the Program AST level, not
  at the `@program` level.
- In a hackathon window the cost of "looks simple on the whiteboard, eats a
  day in edge cases" is real.

## Implementation sketch (when we do ship it)

1. **Make `ProgramInvocation` satisfy the `Gen` protocol.** Add `dispatch(engine)`
   method that delegates to `run(engine=engine)`. No new class hierarchy.
2. **Runner recognizes `ProgramInvocation` in the yield dispatcher** (already
   half-there — the isinstance check in `_run_once` would broaden).
3. **Engine gets context scopes.** `push_context_scope()` + `pop_context_scope()`
   on engines that support injection; Mock just stacks lists; XGrammarEngine
   does the same on `_context`.
4. **Sub-program's `reject_program` + Phase-C retry stays local.** Only
   `GrammarExhausted` that exits the inner retry budget propagates up; at
   that point the outer's Phase-C catches and retries.
5. **Source-in-prompt recurses.** When building the prompt from a program's
   source, `inspect.getsource` on any yielded `@program` also gets walked
   and inlined (or referenced by name + its own source separately).

Rough scope: 150-250 LOC + tests + docs update. Probably half a day once
the design is locked.

## Related future work

- **Verifiers as first-class `@verifier`-decorated callables.** Already on the
  plan. Composes with Flavor B: a sub-program's verifier is itself a reusable
  `@verifier` that other sub-programs can import.
- **Layer 3 (static reduction):** program-level compile pass that fuses
  straight-line runs of yields into one compound grammar. Flavor B would
  let the compiler recurse into composed sub-programs.
- **"Spec libraries" documentation.** Once composition is solid,
  `orate.specs.email`, `orate.specs.dates`, `orate.specs.ids` become useful
  imports for the common patterns.
