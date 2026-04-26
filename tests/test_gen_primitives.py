"""Coverage for the full gen.* surface against MockEngine.

MockEngine is noise-generator; these tests confirm *wiring*, not
quality. Any real-model coverage lives in tests/test_local_engine.py
(model-gated).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pytest

from orate import gen, program
from orate.engine.mock import MockEngine
from orate.gen import GrammarExhausted


def test_gen_integer_in_range():
    @program
    def pick():
        n = yield gen.integer(10, 20)
        return n

    result = pick().run(engine=MockEngine(seed=0))
    assert 10 <= result <= 20


def test_gen_integer_where_predicate_tightens():
    @program
    def pick_even():
        n = yield gen.integer(1, 20, where=lambda x: x % 2 == 0)
        return n

    for seed in range(8):
        result = pick_even().run(engine=MockEngine(seed=seed))
        assert result % 2 == 0 and 1 <= result <= 20


def test_gen_integer_exhausts_impossible_predicate():
    @program
    def pick_impossible():
        _ = yield gen.integer(1, 3, where=lambda x: x == 100)
        return "unreachable"

    with pytest.raises(GrammarExhausted):
        pick_impossible().run(engine=MockEngine(seed=0))


def test_gen_string_honors_max_len():
    @program
    def pick():
        s = yield gen.string(max_len=8)
        return s

    for seed in range(4):
        result = pick().run(engine=MockEngine(seed=seed))
        assert 1 <= len(result) <= 8


def test_gen_bool_samples_both_truth_values():
    @program
    def pick():
        b = yield gen.boolean()
        return b

    seen = {pick().run(engine=MockEngine(seed=s)) for s in range(20)}
    assert seen == {True, False}


def test_gen_struct_compound_dict():
    @program
    def person():
        p = yield gen.struct(
            name=gen.string(max_len=10),
            age=gen.integer(0, 120),
            is_student=gen.boolean(),
        )
        return p

    result = person().run(engine=MockEngine(seed=0))
    assert set(result.keys()) == {"name", "age", "is_student"}
    assert isinstance(result["name"], str)
    assert 0 <= result["age"] <= 120
    assert isinstance(result["is_student"], bool)


def test_gen_tool_runs_and_returns():
    def roll(sides: int) -> int:
        return sides  # deterministic for the test

    @program
    def combat():
        damage = yield gen.tool(roll, sides=20)
        return damage

    result = combat().run(engine=MockEngine(seed=0))
    assert result == 20


# ---- gen.datetime --------------------------------------------------------


def test_gen_datetime_default_range_is_today_hourly():
    """Default range produces 24 hourly slots from today's midnight."""

    @program
    def pick():
        dt = yield gen.datetime()
        return dt

    seen = {pick().run(engine=MockEngine(seed=s)) for s in range(8)}
    # All seen results must be datetime instances on the same date.
    assert all(isinstance(d, datetime) for d in seen)
    assert len({d.date() for d in seen}) == 1
    # Every result must be on the hour boundary (default granularity=60).
    assert all(d.minute == 0 and d.second == 0 for d in seen)


def test_gen_datetime_explicit_bounds_and_granularity():
    """Min/max + granularity produce exactly the right gridded slots."""

    @program
    def pick():
        dt = yield gen.datetime(
            min_dt=datetime(2026, 4, 26, 9, 0),
            max_dt=datetime(2026, 4, 26, 11, 0),
            granularity_minutes=30,
        )
        return dt

    seen = {pick().run(engine=MockEngine(seed=s)) for s in range(20)}
    # Expected slots: 9:00, 9:30, 10:00, 10:30, 11:00 (5 slots).
    expected = {
        datetime(2026, 4, 26, 9, 0),
        datetime(2026, 4, 26, 9, 30),
        datetime(2026, 4, 26, 10, 0),
        datetime(2026, 4, 26, 10, 30),
        datetime(2026, 4, 26, 11, 0),
    }
    assert seen <= expected
    # Across many seeds we should land in multiple slots.
    assert len(seen) >= 2


def test_gen_datetime_where_predicate_filters():
    """Single-arg where= filters the slot grid before the engine sees it."""

    def in_business_hours(dt: datetime) -> bool:
        return 9 <= dt.hour <= 17

    @program
    def pick():
        dt = yield gen.datetime(where=in_business_hours)
        return dt

    for seed in range(8):
        result = pick().run(engine=MockEngine(seed=seed))
        assert 9 <= result.hour <= 17


def test_gen_datetime_cross_field_where_constraint():
    """The video's exact pattern: end is 2 hours after start.

    First yield enumerates business hours; second yield closes over
    ``start`` and demands ``e - start == timedelta(hours=2)`` — the
    accept set narrows to exactly one slot, and dispatch returns it
    without engine sampling.
    """

    def in_business_hours(dt: datetime) -> bool:
        return 9 <= dt.hour <= 17

    @program
    def book_meeting(duration_h: int):
        start = yield gen.datetime(
            min_dt=datetime(2026, 4, 26, 0, 0),
            max_dt=datetime(2026, 4, 26, 23, 0),
            where=in_business_hours,
        )
        end = yield gen.datetime(
            min_dt=datetime(2026, 4, 26, 0, 0),
            max_dt=datetime(2026, 4, 26, 23, 0),
            where=lambda e: e - start == timedelta(hours=duration_h),
        )
        return {"start": start, "end": end}

    for seed in range(6):
        result = book_meeting(2).run(engine=MockEngine(seed=seed))
        assert 9 <= result["start"].hour <= 17
        assert result["end"] - result["start"] == timedelta(hours=2)


def test_gen_datetime_exhausts_on_impossible_predicate():
    """If no slot satisfies where=, dispatch raises."""

    @program
    def pick():
        _ = yield gen.datetime(
            min_dt=datetime(2026, 4, 26, 9, 0),
            max_dt=datetime(2026, 4, 26, 11, 0),
            where=lambda d: d.year == 1999,
        )
        return "unreachable"

    with pytest.raises(GrammarExhausted):
        pick().run(engine=MockEngine(seed=0))


def test_gen_datetime_rejects_inverted_bounds():
    """min_dt > max_dt is a programmer error."""

    @program
    def pick():
        _ = yield gen.datetime(
            min_dt=datetime(2026, 4, 26, 12, 0),
            max_dt=datetime(2026, 4, 26, 9, 0),
        )
        return None

    with pytest.raises(ValueError):
        pick().run(engine=MockEngine(seed=0))


def test_gen_datetime_grid_too_large_raises():
    """A range that would produce >10k slots is refused upfront.

    Five-year range at 1-minute granularity is ~2.6M slots — well past
    the enumeration budget. Caller's responsibility to tighten.
    """

    @program
    def pick():
        _ = yield gen.datetime(
            min_dt=datetime(2020, 1, 1),
            max_dt=datetime(2025, 1, 1),
            granularity_minutes=1,
        )
        return None

    with pytest.raises(GrammarExhausted, match="too large"):
        pick().run(engine=MockEngine(seed=0))


def test_gen_datetime_singleton_accept_set_skips_engine():
    """When where= narrows to exactly one slot, no sampler call happens."""

    @program
    def pick():
        dt = yield gen.datetime(
            min_dt=datetime(2026, 4, 26, 9, 0),
            max_dt=datetime(2026, 4, 26, 11, 0),
            where=lambda d: d == datetime(2026, 4, 26, 10, 0),
        )
        return dt

    # Same result regardless of seed — proof that the engine wasn't asked.
    results = {pick().run(engine=MockEngine(seed=s)) for s in range(10)}
    assert results == {datetime(2026, 4, 26, 10, 0)}


def test_gen_tool_unified_with_other_yields():
    """Act-3: tool calls share the same yield primitive as gen.* calls."""

    def lookup_enemy() -> str:
        return "dragon"

    @program
    def turn():
        kind = yield gen.choice(["attack", "speak"])
        if kind == "attack":
            target = yield gen.tool(lookup_enemy)
            damage = yield gen.integer(1, 10)
            return {"kind": "attack", "target": target, "damage": damage}
        else:
            return {"kind": "speak"}

    result = turn().run(engine=MockEngine(seed=1))
    assert result["kind"] in {"attack", "speak"}
    if result["kind"] == "attack":
        assert result["target"] == "dragon"
        assert 1 <= result["damage"] <= 10
