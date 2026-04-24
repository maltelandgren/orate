"""Stress tests: edge cases and regressions for Layer 1 (witness enum),
Layer 3 (forward-checking), Layer 4 (source-in-prompt), and verifiers.

These complement the focused per-module tests in test_compile.py,
test_prompt.py, and test_verify.py by hitting boundary conditions,
combinatorial interactions, and the places where the four features
cross paths. Uses MockEngine (or a small spy wrapper around it) so
nothing here needs a real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from orate import (
    Accept,
    GrammarExhausted,
    ProgramRejected,
    Reject,
    build_prompt,
    gen,
    program,
    verifier,
)
from orate.compile import (
    DEFAULT_ENUM_BUDGET,
    enumerate_int,
)
from orate.engine.mock import MockEngine

# ---------------------------------------------------------------------------
# Spy engine: wraps MockEngine, counts each primitive sample call. Lets us
# assert things like "engine.sample_choice was never invoked" for the
# single-value accept-set case.
# ---------------------------------------------------------------------------


@dataclass
class SpyEngine:
    """Counts primitive calls; delegates to an inner MockEngine."""

    seed: int = 0
    choice_calls: int = 0
    int_calls: int = 0
    string_calls: int = 0
    bool_calls: int = 0
    last_choice_options: list[list[str]] = field(default_factory=list)
    _inner: MockEngine = field(init=False)
    _context: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._inner = MockEngine(seed=self.seed)

    def sample_choice(self, options):  # noqa: ANN001
        self.choice_calls += 1
        self.last_choice_options.append(list(options))
        return self._inner.sample_choice(options)

    def sample_int(self, min_val, max_val, *, excluded=None):  # noqa: ANN001
        self.int_calls += 1
        return self._inner.sample_int(min_val, max_val, excluded=excluded)

    def sample_string(self, *, max_len, pattern=None, excluded=None):  # noqa: ANN001
        self.string_calls += 1
        return self._inner.sample_string(max_len=max_len, pattern=pattern, excluded=excluded)

    def sample_bool(self):
        self.bool_calls += 1
        return self._inner.sample_bool()

    def inject_context(self, text: str) -> None:
        self._context.append(text)


# ---------------------------------------------------------------------------
# 1. Layer 1 boundaries: enumeration budget, single-value short-circuit,
#    empty accept set, predicate-raises semantics.
# ---------------------------------------------------------------------------


def test_layer1_enum_at_exact_budget_enumerates():
    """A domain of size == DEFAULT_ENUM_BUDGET is still enumerable (not None)."""
    size = DEFAULT_ENUM_BUDGET
    accepted = enumerate_int(0, size - 1, where=lambda x: x == size - 1)
    assert accepted == [size - 1]


def test_layer1_enum_one_over_budget_returns_none():
    """Size == budget + 1 falls back to rejection sampling (returns None)."""
    size = DEFAULT_ENUM_BUDGET + 1
    accepted = enumerate_int(0, size - 1, where=lambda x: x == 7)
    assert accepted is None


def test_layer1_single_value_accept_skips_engine():
    """Choice whose where= narrows to exactly one option must not call
    engine.sample_choice — the dispatcher short-circuits."""
    engine = SpyEngine(seed=0)

    @program
    def pick():
        c = yield gen.choice(["a", "b", "c", "d"], where=lambda x: x == "c")
        return c

    assert pick().run(engine=engine) == "c"
    assert engine.choice_calls == 0, "single-value accept set should skip sample_choice"


def test_layer1_single_value_int_skips_engine():
    """Same short-circuit for Int: only one value satisfies → engine untouched."""
    engine = SpyEngine(seed=0)

    @program
    def pick():
        n = yield gen.integer(1, 100, where=lambda x: x == 42)
        return n

    assert pick().run(engine=engine) == 42
    assert engine.choice_calls == 0
    assert engine.int_calls == 0


def test_layer1_empty_accept_set_raises_with_clear_message():
    """An unsatisfiable predicate must raise GrammarExhausted with a message
    that mentions the gen type and the domain — not a bare KeyError etc."""

    @program
    def pick():
        _ = yield gen.choice(["a", "b"], where=lambda _: False)
        return None

    with pytest.raises(GrammarExhausted) as exc_info:
        pick().run(engine=MockEngine(seed=0))
    msg = str(exc_info.value)
    assert "choice" in msg.lower()
    assert "satisf" in msg.lower()


def test_layer1_mixed_error_and_accept_keeps_only_clean_accepts():
    """Predicate `1/(x-3) > 0` on int[1,6]: x=3 raises (rejected), x=1,2
    give negative (rejected), x=4,5,6 give positive (accepted)."""
    accepted = enumerate_int(1, 6, where=lambda x: 1 / (x - 3) > 0)
    assert accepted == [4, 5, 6]


def test_layer1_tautology_predicate_accepts_full_domain():
    """where=lambda _: True is a no-op; every value is in the accept set."""
    engine = SpyEngine(seed=0)

    @program
    def pick():
        c = yield gen.choice(["a", "b", "c"], where=lambda _: True)
        return c

    result = pick().run(engine=engine)
    assert result in {"a", "b", "c"}
    # Engine was called exactly once with the full option set.
    assert engine.choice_calls == 1
    assert engine.last_choice_options[0] == ["a", "b", "c"]


def test_layer1_contradiction_predicate_raises_grammar_exhausted():
    """where=lambda _: False rejects everything → GrammarExhausted."""

    @program
    def pick():
        _ = yield gen.integer(1, 10, where=lambda _: False)
        return None

    with pytest.raises(GrammarExhausted):
        pick().run(engine=MockEngine(seed=0))


# ---------------------------------------------------------------------------
# 2. Layer 3 cross-field forward-checking: 3-field sum, mixed types,
#    partial references, non-enumerable fields, unsatisfiable predicates.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_layer3_three_field_sum_to_ten(seed):
    """a+b+c==10 over int[0,5]. Forward-checking should produce a valid
    triple in one pass — no struct-level rejection retries."""
    engine = SpyEngine(seed=seed)

    @program
    def triple():
        return (
            yield gen.struct(
                a=gen.integer(0, 5),
                b=gen.integer(0, 5),
                c=gen.integer(0, 5),
                where=lambda d: d["a"] + d["b"] + d["c"] == 10,
            )
        )

    result = triple().run(engine=engine)
    assert result["a"] + result["b"] + result["c"] == 10
    for v in result.values():
        assert 0 <= v <= 5
    # Zero struct-level reject notes because forward-checking closes the
    # predicate over each bound field.
    assert engine._context == []


def test_layer3_mixed_type_int_and_choice_cross_predicate():
    """x:int[1,5], color:choice(red/blue), cross: red→x<=3 else x>=3."""
    for seed in range(6):
        engine = MockEngine(seed=seed)

        @program
        def pick():
            return (
                yield gen.struct(
                    x=gen.integer(1, 5),
                    color=gen.choice(["red", "blue"]),
                    where=lambda d: (
                        (d["color"] == "red" and d["x"] <= 3)
                        or (d["color"] == "blue" and d["x"] >= 3)
                    ),
                )
            )

        r = pick().run(engine=engine)
        assert r["color"] in {"red", "blue"}
        assert 1 <= r["x"] <= 5
        if r["color"] == "red":
            assert r["x"] <= 3
        else:
            assert r["x"] >= 3


def test_layer3_partial_field_reference_ignores_unreferenced_field():
    """where: a==c, no mention of b. b should be free, c forced to a."""
    for seed in range(4):
        engine = MockEngine(seed=seed)

        @program
        def pick():
            return (
                yield gen.struct(
                    a=gen.integer(1, 4),
                    b=gen.integer(1, 4),
                    c=gen.integer(1, 4),
                    where=lambda d: d["a"] == d["c"],
                )
            )

        r = pick().run(engine=engine)
        assert r["a"] == r["c"]
        assert 1 <= r["b"] <= 4


def test_layer3_non_enumerable_field_falls_back_to_native_dispatch():
    """Struct mixing String (non-enumerable) + Int. Forward-checking can't
    narrow String, so it dispatches natively; the cross predicate is
    enforced at the struct level by the backup rejection loop."""
    engine = MockEngine(seed=0)

    @program
    def pick():
        return (
            yield gen.struct(
                name=gen.string(max_len=8),
                tag=gen.integer(1, 3),
                # Cross predicate only constrains tag, which IS enumerable.
                where=lambda d: d["tag"] >= 2,
            )
        )

    r = pick().run(engine=engine)
    assert isinstance(r["name"], str) and r["name"]
    assert r["tag"] in {2, 3}


def test_layer3_unsatisfiable_cross_predicate_raises():
    """No (a,b) in [0,3]x[0,3] can have a+b==100. Should exhaust."""

    @program
    def pick():
        return (
            yield gen.struct(
                a=gen.integer(0, 3),
                b=gen.integer(0, 3),
                where=lambda d: d["a"] + d["b"] == 100,
            )
        )

    with pytest.raises(GrammarExhausted):
        pick().run(engine=MockEngine(seed=0))


# ---------------------------------------------------------------------------
# 3. Verifier composition: chains, external state, yield-from, no-kwarg form.
# ---------------------------------------------------------------------------


def test_verifier_chain_only_failing_message_raised():
    """Three verifiers; first two accept, third rejects. The raised
    ProgramRejected should name only the third verifier."""

    @verifier
    def first_ok(_x):
        return Accept()

    @verifier
    def second_ok(_x):
        return Accept()

    @verifier
    def third_fails(_x):
        return Reject("third failed here")

    @program
    def pick():
        n = yield gen.integer(1, 3)
        yield first_ok(n)
        yield second_ok(n)
        yield third_fails(n)
        return n

    with pytest.raises(ProgramRejected) as exc_info:
        pick().run(engine=MockEngine(seed=0))
    msg = str(exc_info.value)
    assert "third_fails" in msg
    assert "third failed here" in msg
    assert "first_ok" not in msg
    assert "second_ok" not in msg


def test_verifier_with_closure_state_survives_retry():
    """Verifier closes over a list of already-seen values; each retry pushes
    a new sample and accepts when the sample is fresh. State must be
    stable across Phase-C retries (it's not re-initialized)."""
    seen: list[int] = []

    @verifier
    def distinct_from_seen(x):
        if x in seen:
            return Reject(f"{x} already seen")
        seen.append(x)
        return Accept()

    @program(whole_program_retries=20)
    def pick():
        n = yield gen.integer(1, 3)
        yield distinct_from_seen(n)
        return n

    # Run three times; on each run we expect a fresh value eventually.
    out = [pick().run(engine=MockEngine(seed=s)) for s in range(3)]
    assert len(set(out)) == 3
    assert set(out) == {1, 2, 3}


def test_verifier_yield_from_sub_generator_still_phase_c_catchable():
    """A @program that uses `yield from` to delegate to a helper which
    itself yields a verifier. Reject inside the sub-gen must still raise
    ProgramRejected that Phase-C can catch + retry."""
    attempts = {"n": 0}

    @verifier
    def needs_second_try(_x):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return Reject("first call always fails")
        return Accept()

    def validate(n):
        yield needs_second_try(n)

    @program(whole_program_retries=3)
    def pick():
        n = yield gen.integer(1, 5)
        yield from validate(n)
        return n

    result = pick().run(engine=MockEngine(seed=0))
    assert 1 <= result <= 5
    assert attempts["n"] == 2


def test_verifier_no_extra_kwargs_just_value():
    """Bare-form verifier — only the value, no context kwargs."""

    @verifier
    def is_even(x):
        return Accept() if x % 2 == 0 else Reject("odd")

    @program(whole_program_retries=30)
    def pick():
        n = yield gen.integer(1, 10)
        yield is_even(n)
        return n

    result = pick().run(engine=MockEngine(seed=0))
    assert result % 2 == 0


# ---------------------------------------------------------------------------
# 4. Source-in-prompt edge cases: helpers, special characters, long strings.
# ---------------------------------------------------------------------------


def _external_helper(n: int) -> int:
    """A helper used by a @program body; its source lives here, not in the body."""
    return n * 2


@program
def _program_using_external_helper():
    # The body calls into the helper below; introspection should show
    # only the body's source, not the helper's.
    raw = yield gen.integer(1, 5)
    return _external_helper(raw)


def test_prompt_helper_function_not_inlined():
    """Pinned behavior: build_prompt shows the @program body source only,
    not the source of helpers the body references. The helper function's
    body (its multiplication logic) must not appear in the rendered prompt."""
    out = build_prompt(_program_using_external_helper, user_prompt="do it")
    assert "_program_using_external_helper" in out
    assert "_external_helper(raw)" in out  # call site IS in the body
    # The helper's own def/body text must NOT be inlined. Check for a
    # snippet unique to the helper.
    assert "return n * 2" not in out
    assert "def _external_helper" not in out


@program
def _program_with_tricky_description():
    # Description contains quotes, backslashes, and a newline escape.
    x = yield gen.integer(
        1,
        5,
        description='pick a digit — "even" numbers are \\special\\; also\nnewline',
    )
    return x


def test_prompt_description_with_special_chars_renders():
    """Quotes, backslashes, and escaped newlines inside description= must
    not break the prompt. The description (or at least a distinguishing
    fragment) must appear in the output."""
    out = build_prompt(_program_with_tricky_description, user_prompt="ok")
    # The leading-comment annotation should carry at least the visible
    # prefix of the description through to the rendered source.
    assert "pick a digit" in out
    # The prompt must remain a valid triple-backtick python block.
    assert "```python" in out
    assert out.rstrip().endswith("Begin:")


# A >500-char description written as a single contiguous string literal
# (the only form _extract_description recognizes — ast.Constant). The
# literal includes a 600-char run so we can assert on-by-one-preserved.
_LONG_LITERAL = "long description marker " + ("q" * 600) + " end marker"


@program
def _program_with_long_description():
    y = yield gen.integer(0, 9, description=_LONG_LITERAL)  # name-ref; raw only
    return y


def test_prompt_very_long_description_preserved_in_raw_source():
    """A >500-char description is preserved in the rendered prompt when
    assigned through any path that ends up in the source text. Raw-mode
    rendering always shows the @program source verbatim, so if the
    description is bound to a module-level name, the name reference
    appears in the output (pinned behavior of the AST-only path)."""
    out_raw = build_prompt(_program_with_long_description, user_prompt="hi", source_mode="raw")
    # The name reference _LONG_LITERAL is present in the raw source.
    assert "_LONG_LITERAL" in out_raw
    # But the materialized 600-q run is NOT — raw mode shows source, not
    # resolved values. Pin this so a future "eagerly resolve descriptions"
    # change is surfaced as a failing test for review.
    assert ("q" * 600) not in out_raw


@program
def _program_with_inline_long_description():
    # A single string literal >500 chars, inlined directly so the AST
    # _extract_description path can see it as an ast.Constant and emit
    # it as a leading comment above the yield.
    y = yield gen.integer(  # noqa: F841
        0,
        9,
        description=(
            "inline very long description: "
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            " end marker"
        ),
    )
    return y


def test_prompt_very_long_inline_description_in_comment():
    """A >500-char description written as a single Python string (implicit
    string concatenation counts as one ast.Constant) renders as a leading
    comment in the annotated prompt, intact."""
    out = build_prompt(_program_with_inline_long_description, user_prompt="hi")
    # Leading-comment marker inserted by render_program_with_descriptions.
    assert "# inline very long description" in out
    # End marker of the literal survives to the output.
    assert "end marker" in out
    # Character-count sanity: more than 500 a's live in the prompt.
    assert ("a" * 500) in out


# ---------------------------------------------------------------------------
# 5. End-to-end integration: witness-enum + forward-check + verifier + prompt.
# ---------------------------------------------------------------------------


@verifier
def sum_is_even(pair):
    s = pair["a"] + pair["b"]
    if s % 2 == 0:
        return Accept()
    return Reject(f"sum {s} is odd")


@program(whole_program_retries=3)
def _integrated_program():
    # Witness-enum Int with a tight predicate.
    seed_n = yield gen.integer(
        1,
        20,
        where=lambda x: x % 3 == 0,
        description="a multiple of three",
    )
    # Forward-check struct with a cross predicate.
    pair = yield gen.struct(
        a=gen.integer(0, 5),
        b=gen.integer(0, 5),
        where=lambda d: d["a"] + d["b"] == 6,
        description="two small non-negative ints summing to six",
    )
    # Verifier: derives a fact about the struct, always accepts here
    # (6 is even) — the check is real; it's just satisfied.
    yield sum_is_even(pair)
    return {"seed": seed_n, "pair": pair}


def test_integrated_all_features_together():
    engine = SpyEngine(seed=0)
    result = _integrated_program().run(engine=engine)

    assert result["seed"] % 3 == 0 and 1 <= result["seed"] <= 20
    assert result["pair"]["a"] + result["pair"]["b"] == 6
    # Phase-B notes (rejection context) must be empty: witness enum + forward
    # checking together mean no sample was ever rejected. Verifier accepts,
    # so no Phase-C note either.
    assert engine._context == []
    # The engine was called only for actual pick-among-many cases, never
    # for tightening retries.
    assert engine.int_calls == 0, (
        "witness enum should have routed Int picks through sample_choice, not sample_int"
    )


def test_integrated_prompt_contains_all_descriptions():
    """The same integrated program, rendered: every description= appears
    as a leading comment in the annotated source block."""
    out = build_prompt(_integrated_program, user_prompt="please")
    assert "# a multiple of three" in out
    assert "# two small non-negative ints summing to six" in out
    assert "```python" in out


def test_phase_c_retry_budget_across_verifier_rejection():
    """Reject once, accept second time: program completes inside budget."""
    attempts = {"n": 0}

    @verifier
    def odd_once_then_ok(x):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return Reject(f"first call saw {x}; try again")
        return Accept()

    @program(whole_program_retries=2)
    def pick():
        n = yield gen.integer(1, 3)
        yield odd_once_then_ok(n)
        return n

    engine = MockEngine(seed=0)
    result = pick().run(engine=engine)
    assert 1 <= result <= 3
    assert attempts["n"] == 2
    # Phase-C should have injected exactly one reject note between attempts.
    assert any("first call saw" in note for note in engine._context)
