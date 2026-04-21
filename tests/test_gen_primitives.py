"""Coverage for the full gen.* surface against MockEngine.

MockEngine is noise-generator; these tests confirm *wiring*, not
quality. Any real-model coverage lives in tests/test_local_engine.py
(model-gated).
"""

from __future__ import annotations

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
