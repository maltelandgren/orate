from __future__ import annotations

import pytest

from orate import gen, program
from orate.engine.mock import MockEngine
from orate.gen import GrammarExhausted


def test_trivial_choice_runs():
    @program
    def pick_color():
        c = yield gen.choice(["red", "green", "blue"])
        return c

    result = pick_color().run(engine=MockEngine(seed=0))
    assert result in {"red", "green", "blue"}


def test_where_predicate_filters():
    @program
    def pick_even():
        n = yield gen.choice(["1", "2", "3", "4"], where=lambda x: int(x) % 2 == 0)
        return int(n)

    for seed in range(10):
        result = pick_even().run(engine=MockEngine(seed=seed))
        assert result % 2 == 0


def test_tightening_exhausts_with_unsatisfiable_predicate():
    @program
    def pick_impossible():
        _ = yield gen.choice(["a", "b", "c"], where=lambda x: x == "z")
        return "unreachable"

    with pytest.raises(GrammarExhausted):
        pick_impossible().run(engine=MockEngine(seed=0))


def test_multi_yield_program():
    @program
    def two_choices():
        first = yield gen.choice(["attack", "speak"])
        second = yield gen.choice(["loud", "quiet"])
        return (first, second)

    result = two_choices().run(engine=MockEngine(seed=42))
    assert result[0] in {"attack", "speak"}
    assert result[1] in {"loud", "quiet"}


def test_non_gen_yield_raises():
    @program
    def bad_program():
        x = yield 42  # not a Gen instance
        return x

    with pytest.raises(TypeError, match="yielded non-Gen value"):
        bad_program().run(engine=MockEngine(seed=0))
