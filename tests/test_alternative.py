"""Unit tests for ``gen.alternative`` — composer-side leaf dispatch.

These exercise the same transition pattern Session uses (sample
prefix → sample body → parse args → drive generator → collect return),
but at the composer's scope. A stub engine returns canned chunks so
we can simulate "the model picked X" deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from orate import gen, program
from orate.gen import Alternative, GrammarExhausted, Picked
from orate.program import ProgramRejected

# ---- stub engine -------------------------------------------------------


@dataclass
class _StubEngine:
    """Returns canned chunks for sample_under in order. Records appends."""

    canned: list[str]
    appended: list[str] = None  # type: ignore[assignment]
    calls: list[tuple[str, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.appended = []
        self.calls = []
        self._idx = 0

    def sample_under(self, grammar: str, max_tokens: int = 256) -> str:
        self.calls.append((grammar, max_tokens))
        if self._idx >= len(self.canned):
            raise RuntimeError(f"stub ran out of canned responses; last grammar:\n{grammar}")
        out = self.canned[self._idx]
        self._idx += 1
        return out

    def append(self, text: str) -> None:
        self.appended.append(text)

    def begin_session(self, prompt: str) -> None:  # noqa: ARG002
        pass


# ---- leaf @programs used as fixtures ----------------------------------


@program
def pick_color():
    c = yield gen.choice(["red", "green", "blue"])
    return {"chosen": c}


@program
def pick_count():
    n = yield gen.integer(0, 100)
    return {"count": n}


@program
def succ_pair():
    """Cross-yield where: the second value must equal first + 1."""
    n = yield gen.integer(0, 100)
    s = yield gen.integer(0, 100, where=lambda x: x == n + 1)
    return {"n": n, "s": s}


@program(invocable=False)
def some_composer():
    while True:
        c = yield gen.choice(["a", "b"])
        if c == "a":
            return None


# ---- happy paths -------------------------------------------------------


def test_alternative_picks_single_leaf():
    """One leaf in the alternation → model picks it (only choice)."""
    engine = _StubEngine(
        canned=[
            "@pick_color(",  # prefix sample
            "red",  # body sample
        ]
    )
    spec = gen.alternative([pick_color])
    picked = spec.dispatch(engine)
    assert isinstance(picked, Picked)
    assert picked.name == "pick_color"
    assert picked.args == ("red",)
    assert picked.value == {"chosen": "red"}
    # The closing ')' was appended after the body.
    assert ")" in "".join(engine.appended)


def test_alternative_picks_one_of_many():
    engine = _StubEngine(canned=["@pick_count(", "42"])
    picked = gen.alternative([pick_color, pick_count]).dispatch(engine)
    assert picked.name == "pick_count"
    assert picked.args == (42,)
    assert picked.value == {"count": 42}


def test_alternative_cross_yield_predicate_passes():
    engine = _StubEngine(canned=["@succ_pair(", "5, 6"])
    picked = gen.alternative([succ_pair]).dispatch(engine)
    assert picked.name == "succ_pair"
    assert picked.args == (5, 6)
    assert picked.value == {"n": 5, "s": 6}


# ---- predicate rejection -----------------------------------------------


def test_alternative_cross_yield_predicate_fails():
    engine = _StubEngine(canned=["@succ_pair(", "5, 9"])  # 9 != 5+1
    with pytest.raises(ProgramRejected, match="where="):
        gen.alternative([succ_pair]).dispatch(engine)


# ---- input validation --------------------------------------------------


def test_alternative_empty_program_list_raises():
    engine = _StubEngine(canned=[])
    with pytest.raises(GrammarExhausted, match="empty"):
        gen.alternative([]).dispatch(engine)


def test_alternative_rejects_composer():
    engine = _StubEngine(canned=[])
    with pytest.raises(TypeError, match="composer"):
        gen.alternative([pick_color, some_composer]).dispatch(engine)


def test_alternative_grammar_includes_only_named_leaves():
    """The prefix grammar emitted on sample_under should list each leaf's
    name verbatim — no inlined body rules from anywhere."""
    engine = _StubEngine(canned=["@pick_color(", "red"])
    gen.alternative([pick_color, pick_count]).dispatch(engine)
    # First call is the prefix grammar.
    prefix_grammar, _ = engine.calls[0]
    assert '"@pick_color("' in prefix_grammar
    assert '"@pick_count("' in prefix_grammar
    # No body rules inlined into the prefix grammar.
    assert "pick_color_body" not in prefix_grammar
    # Second call is the picked leaf's full body grammar.
    body_grammar, _ = engine.calls[1]
    assert "pick_color_body" in body_grammar


# ---- alternative as a Gen subclass usage -------------------------------


def test_alternative_constructor_returns_alternative_instance():
    spec = gen.alternative([pick_color, pick_count])
    assert isinstance(spec, Alternative)
    assert len(spec.programs) == 2


def test_alternative_inside_a_composer_body() -> None:
    """A composer yielding gen.alternative gets back a Picked from .send()."""

    @program(invocable=False)
    def loop_once() -> Iterator[gen.Gen]:  # type: ignore[name-defined]
        action = yield gen.alternative([pick_color])
        return action

    engine = _StubEngine(canned=["@pick_color(", "blue"])
    invocation = loop_once()
    result = invocation.run(engine=engine)
    assert isinstance(result, Picked)
    assert result.name == "pick_color"
    assert result.value == {"chosen": "blue"}
