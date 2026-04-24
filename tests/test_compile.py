"""Layer 1 + Layer 3: witness enumeration + forward-checking on Struct.

These tests lock in that the compile step actually bypasses the
rejection loop for enumerable domains — no ``_context`` notes appear
because no yields were rejected. For unbounded domains (String) the
rejection path remains; separate tests cover that in
``test_phase_bc.py``.
"""

from __future__ import annotations

import pytest

from orate import GrammarExhausted, gen, program
from orate.compile import (
    compile_struct_field,
    enumerate_bool,
    enumerate_choice,
    enumerate_int,
)
from orate.engine.mock import MockEngine

# --- enumerate_* unit tests -----------------------------------------------


def test_enumerate_choice_filters_by_where():
    accepted = enumerate_choice(["a", "b", "c", "d"], where=lambda x: x in {"a", "c"})
    assert accepted == ["a", "c"]


def test_enumerate_choice_no_where_returns_all():
    assert enumerate_choice(["x", "y"], where=None) == ["x", "y"]


def test_enumerate_int_returns_full_set():
    assert enumerate_int(1, 5, where=None) == [1, 2, 3, 4, 5]


def test_enumerate_int_with_narrow_predicate():
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

    accepted = enumerate_int(1, 20, where=is_prime)
    assert accepted == [2, 3, 5, 7, 11, 13, 17, 19]


def test_enumerate_int_returns_none_for_oversized_domain():
    accepted = enumerate_int(0, 1_000_000, where=lambda x: x == 42)
    assert accepted is None


def test_enumerate_int_honors_custom_budget():
    accepted = enumerate_int(1, 100, where=lambda x: x % 2 == 0, budget=50)
    assert accepted is None
    accepted = enumerate_int(1, 40, where=lambda x: x % 2 == 0, budget=50)
    assert accepted == list(range(2, 41, 2))


def test_enumerate_bool_both():
    assert set(enumerate_bool(where=None)) == {True, False}


def test_enumerate_bool_only_true():
    assert enumerate_bool(where=lambda v: v is True) == [True]


def test_enumerate_bool_unsatisfiable():
    assert enumerate_bool(where=lambda v: v is None) == []


def test_safe_eval_swallows_predicate_errors():
    """Predicate that raises is treated as rejection, not library bug."""
    accepted = enumerate_int(1, 5, where=lambda x: 1 / (x - 3) > 0)
    # x=3 raises ZeroDivisionError → treated as rejection.
    # x=4,5: 1/1=1, 1/2=0.5 > 0 → accepted.
    # x=1,2: 1/-2=-0.5, 1/-1=-1 < 0 → rejected by the predicate itself.
    assert accepted == [4, 5]


# --- integration: witness-enum eliminates the retry loop ------------------


def test_choice_with_where_no_rejection_needed():
    """Witness enum compiles Choice's where= at dispatch time; the engine
    only sees the already-filtered candidates, so no Phase-B notes fire."""
    engine = MockEngine(seed=0)

    @program
    def pick():
        c = yield gen.choice(
            ["a", "b", "c", "d"],
            where=lambda x: x == "a",
            reject_message=lambda v: f"{v!r} is not 'a'",
        )
        return c

    assert pick().run(engine=engine) == "a"
    # If Layer 1 works, no rejections fired; no notes injected.
    assert engine._context == [], "witness enum should have bypassed rejection"


def test_integer_with_narrow_predicate_no_rejection_needed():
    """Same mechanism for Int: the prime-with-digit-sum-10 case compiles
    to {19, 37, 73} statically; the engine chooses among those three
    without any retry loop."""
    engine = MockEngine(seed=0)

    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

    def digit_sum(n: int) -> int:
        return sum(int(c) for c in str(abs(n)))

    @program
    def pick_prime_digitsum10():
        n = yield gen.integer(
            10,
            99,
            where=lambda v: is_prime(v) and digit_sum(v) == 10,
            reject_message=lambda v: f"{v} failed",
        )
        return n

    result = pick_prime_digitsum10().run(engine=engine)
    assert result in {19, 37, 73}
    assert engine._context == [], "Layer 1 should have eliminated retries"


def test_integer_domain_too_large_falls_back_to_rejection():
    """Domain above the enum budget → witness-enum returns None and the
    Int dispatcher falls back to today's rejection-sampling loop."""
    from orate.compile import enumerate_int

    assert enumerate_int(1, 20_000, where=lambda v: v == 7777) is None


def test_empty_accept_set_raises_grammar_exhausted():
    engine = MockEngine(seed=0)

    @program
    def pick_impossible():
        _ = yield gen.integer(1, 10, where=lambda v: v > 1000)
        return None

    with pytest.raises(GrammarExhausted):
        pick_impossible().run(engine=engine)


def test_bool_with_narrow_predicate_no_engine_call():
    """If exactly one bool satisfies the predicate, we skip the engine."""
    engine = MockEngine(seed=0)

    @program
    def only_true():
        b = yield gen.boolean(where=lambda v: v is True)
        return b

    assert only_true().run(engine=engine) is True


# --- Layer 3: forward-checking on Struct ----------------------------------


def test_struct_without_cross_field_predicate_unchanged():
    engine = MockEngine(seed=0)

    @program
    def pair():
        return (yield gen.struct(x=gen.integer(1, 3), y=gen.choice(["a", "b"])))

    result = pair().run(engine=engine)
    assert 1 <= result["x"] <= 3
    assert result["y"] in {"a", "b"}


def test_struct_forward_check_sum_eq_10():
    """Classic: x+y==10 with x, y in [0,10]. Once x is bound, y is forced."""
    engine = MockEngine(seed=0)

    @program
    def sum_to_ten():
        return (
            yield gen.struct(
                x=gen.integer(0, 10),
                y=gen.integer(0, 10),
                where=lambda d: d["x"] + d["y"] == 10,
            )
        )

    result = sum_to_ten().run(engine=engine)
    assert result["x"] + result["y"] == 10
    # Forward-checking means the struct-level rejection loop never
    # fires: every successful bind produces a consistent pair in one pass.
    assert engine._context == []


def test_struct_forward_check_gt_relation():
    """x < y, both in [1, 5]. Once x=3, y is narrowed to {4, 5}."""
    engine = MockEngine(seed=0)

    @program
    def ordered_pair():
        return (
            yield gen.struct(
                x=gen.integer(1, 5),
                y=gen.integer(1, 5),
                where=lambda d: d["x"] < d["y"],
            )
        )

    result = ordered_pair().run(engine=engine)
    assert result["x"] < result["y"]


def test_struct_forward_check_unsatisfiable_raises():
    engine = MockEngine(seed=0)

    @program
    def impossible():
        return (
            yield gen.struct(
                x=gen.integer(1, 3),
                y=gen.integer(1, 3),
                where=lambda d: d["x"] == 99 and d["y"] == 99,
            )
        )

    with pytest.raises(GrammarExhausted):
        impossible().run(engine=engine)


def test_compile_struct_field_narrows_y_after_x_bound():
    """Unit-level check that the primitive behaves as expected."""
    int_spec = gen.integer(0, 10)
    object.__setattr__(int_spec, "_field_name", "y")
    accepted = compile_struct_field(
        int_spec,
        bound={"x": 3},
        cross_field_where=lambda d: d["x"] + d["y"] == 10,
    )
    assert accepted == [7]


def test_compile_struct_field_returns_none_for_string_field():
    """String fields can't be enumerated; caller falls back to native dispatch."""
    str_spec = gen.string(max_len=10)
    object.__setattr__(str_spec, "_field_name", "name")
    assert compile_struct_field(str_spec, bound={}, cross_field_where=None) is None
