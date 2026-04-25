# JIT grammar segmentation — the architectural endpoint for `@program`

**Status:** vision / future direction. Not implemented.
**Relates to:** [`two-tier-and-transitions.md`](two-tier-and-transitions.md) (the current pragmatic approximation).
**Predecessor in the original plan:** "Layer 3b: static reduction — fuse straight-line yields into one compound grammar." Generalises that idea to non-straight-line bodies.

## TL;DR

`@program` bodies are generator functions. Some bodies can be lowered into one big grammar; some can't. Today we draw a hard line — leaves are wholly static, composers are wholly dynamic — because that's the cheapest thing to ship.

The principled architecture is **block-level compilation with recompile boundaries**, modelled on what JIT compilers do for code. Walk the body, classify each yield as self-contained or dependent, group consecutive self-contained yields into segments, compile each segment as one grammar, recompile between segments. Number-of-segments is a property of the body, not a tier the user picks. Everything else falls out.

## What "self-contained" means

A yield is **self-contained** at its position in the body iff the grammar produced by its `gen.X(...)` spec can be determined from:

- The function's parameters
- Free variables / module-level constants captured in the closure
- Constants and literals in the spec call

It is **dependent** iff its grammar references:

- A name bound by a prior yield
- A mutable Python value updated by code between yields
- An external effect's result (an I/O yield, a `make_new_program` source synthesis)

Note: `where=` predicates are *not* part of the grammar — they're post-hoc verifiers driven by `body_iter.send()`. A cross-yield closure inside a `where=` is fine; it doesn't make the yield dependent. (This is why today's `algebra_step` is one fully-static segment despite its third yield's `where=lambda s: equivalent_under(rule, before, s)`.)

## The four boundary kinds

What forces a recompile between segments:

1. **Grammar-shape dependence.** `gen.choice(some_dynamic_list)` where the list mutates between yields, or `gen.integer(0, prior_yield)` where the bound depends on a sampled value. The matcher can't pre-build the grammar without knowing the prior value.
2. **Branch on a yielded value.** `if x == "a": yield gen.X else: yield gen.Y`. In principle this can be unioned into one grammar (`("a, " X_grammar | "b, " Y_grammar)`) for finite branches, but it's an analysis question — at some sophistication level you stop unioning and just recompile.
3. **External effects updating Python state.** `make_new_program` extending an `available` list; a network yield whose result the next yield depends on; a stateful tool call whose return shapes downstream grammar. The runtime *has* to pause, do the side effect, then build the next grammar.
4. **Unbounded iteration.** `while True: yield ...` where the loop count isn't statically known. A loop over a *static* alternation is regular and can be encoded; a loop over a state-dependent alternation can't.

Most real programs are one or zero boundaries deep. A composer like `dnd` — narrative loop with `make_new_program` and mode transitions — is N boundaries (one per `make_new_program` invocation, plus mode-change points).

## Runtime model

A compiled program is a sequence of **segments**, each holding:

- A pre-compiled GBNF for that segment's emission shape.
- The set of yields covered (so the runtime knows how many `.send()`s to drive after the segment's matcher accepts).
- The names whose values become known at segment exit (used by the *next* segment's compilation).

Execution:

1. Compile segment 0 (free-of-runtime-state by definition).
2. `engine.sample_under(segment_0_grammar)` — matcher emits the whole first segment in one shot.
3. Drive the body's generator with parsed values for segment 0's yields, collecting any yielded `Gen` specs to learn what was bound.
4. Compile segment 1 with those bindings now visible to closures and gen.X args.
5. `engine.sample_under(segment_1_grammar)`.
6. ...repeat until the body returns.

Per-yield overhead is paid only at segment boundaries. A program with 0 boundaries (the common case for tool-call-like leaves) is one matcher invocation, identical to today's leaves. A program with N boundaries is N+1 matcher invocations — far cheaper than per-yield resolution but as flexible.

## What collapses

- The `invocable=True/False` flag.
- The leaf vs composer mental model. Every `@program` is just a generator; the compiler decides how many segments to emit.
- The hand-coded `gen.alternative` versus body-grammar derivation distinction. Both reduce to "compile this segment."
- Most of `Session` — driving segment-by-segment is what Session does today, but for a flat list of yields. Generalised, the same loop handles any program.

## What stays

- `@program` decorator + ProgramInvocation as the top-level type.
- `gen.choice / integer / string / boolean / struct / tool` as yield primitives.
- Engine session protocol (`begin_session / append / sample_under`).
- Predicate verification (still post-hoc per yield within a segment; interleaved between segments).
- The transition pattern at `@`-call boundaries (one program calling another via `@name(args)`).

## What the analysis has to do

1. **Yield discovery.** Walk the AST, find every `yield` expression in the body — including those nested inside `if`/`while`/`for`. (Today's body_grammar walker only handles top-level straight-line yields.)
2. **Dataflow.** For each yield, compute the set of names its `gen.X(...)` spec call references. Compare with names bound by prior yields and names mutated by intervening statements.
3. **Segment formation.** Greedily extend the current segment until a yield's spec is dependent — that's a boundary. Start a new segment.
4. **Branch handling.** For `if`/`elif`/`else` chains where the condition is itself a recently-bound yield value with a finite domain, optionally union the branch grammars into one segment. (This is an optimisation, not strictly required — recompiling at the branch is the safe default.)
5. **Loop handling.** For loops with statically-determinable iteration count or static-body yields, encode as Kleene closure in one segment. Otherwise treat the loop as a recompile boundary on each iteration.
6. **Effect tracking.** Yields that produce side effects (e.g., `gen.tool(make_new_program)` adds to `available`) force a boundary unconditionally — anything downstream that reads the mutated state has to be recompiled.

This is a real compiler. It's not a weekend's work; it's months. But it generalises the body-grammar derivation we already have — that walker is just a degenerate case where every program is one segment or rejected.

## Connection to compiler theory

What we've described is approximately:

- **Basic blocks**: maximal sequences of yields with no internal control-flow boundaries.
- **Dataflow analysis**: which names does each yield's spec reference; which yields produce them.
- **Escape analysis** (light): which mutations leak across yield boundaries.
- **Region inference**: where can we safely fuse vs. where must we recompile.

The body grammar we derive today is a primitive form of this — already a CFG, just for one segment per program.

## Migration path

The current two-tier design is the right pragmatic stopping point for now. It captures the cheap case (entirely-static bodies → one segment) cleanly, and it explicitly opts the hard case (dynamic bodies) out with `invocable=False` so the user owns the loop in Python.

When the segmentation analysis lands:

- The `invocable=` flag becomes optional / informational. Bodies are analysed; segments are derived.
- `gen.alternative` becomes "the special case where every yield in a segment is a sub-program prefix-alternation." Implementation-wise the same; conceptually one corner of a more general system.
- `Session.advance` becomes "drive segment 0 of the top-level program, run any host effects between segments, compile & drive subsequent segments until the program returns."
- `make_new_program` becomes "an effect-producing yield that boundary-forces a recompile of the rest of the body" — no longer special-cased.

The change is mostly additive on top of today's body_grammar.py. The grammar-emission code already handles per-Gen-method derivation; what's needed is the surrounding analysis that figures out which yields belong in which segment.

## Open questions

1. **Where does the analysis run?** At decoration time (eager — `@program` does the AST walk and stores segments on the wrapper)? Or first-call time (lazy — analyse when `.run()` is first invoked)? Eager is simpler; lazy avoids paying for unused programs.
2. **How do we expose segments for inspection?** The user might want to debug "why didn't this fuse into one segment?" An introspection API (`fn.__orate_segments__` or similar) would be useful.
3. **What's the surface for branch unioning?** When the analyser unions an `if`/`else` into one segment vs. forces a recompile is an optimisation choice with grammar-size implications. Probably exposed as a heuristic threshold (max grammar size after union) plus a manual override.
4. **Mutual recursion between programs.** `@program A` yields `@program B` which yields `@program A`. Each invocation is one ProgramInvocation, so the recursion is bounded per-call, but cross-program effects (a yield in B mutating state read in A) need cross-program analysis.
5. **Can the model author segments?** A `make_new_program`-authored program is itself segmented at compile-time-of-synthesis. That should fall out for free, but the validator currently rejects control flow in synthesised sources. Once segmentation is in place, model-authored programs could have control flow too — which feels like a correct generalisation.

## Why we wrote this down

The two-tier design landed today is a pragmatic approximation. The next architecture round (post-hackathon, when there's time for compiler work) should aim straight at this. Storing the vision keeps the next person — or future-us — from re-deriving the same conclusion from scratch.

The unifying principle: **a `@program`'s body is a generator; the runtime's job is to compile as many of its yields as possible into one grammar shot, and recompile only where Python state genuinely matters.** Everything else is mechanism around that.
