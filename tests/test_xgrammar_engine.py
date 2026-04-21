"""Model-gated tests for the local XGrammar engine.

These load the 0.5B Qwen2.5 GGUF once per module and exercise the
full Engine protocol. Skipped entirely if the model or optional
dependencies aren't available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("llama_cpp")
pytest.importorskip("xgrammar")
pytest.importorskip("transformers")

from orate import gen  # noqa: E402
from orate.engine.xgrammar import (  # noqa: E402
    XGrammarEngine,
    _alternation_grammar,
    _gbnf_quote,
    _int_grammar,
)

MODEL_PATH = "/Users/maltelandgren/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

needs_model = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="local GGUF not available",
)


# ---- unit tests that don't need the model --------------------------


def test_gbnf_quote_escapes_specials():
    assert _gbnf_quote("foo") == '"foo"'
    assert _gbnf_quote('sa"id') == '"sa\\"id"'
    assert _gbnf_quote("back\\slash") == '"back\\\\slash"'


def test_alternation_grammar_empty_raises():
    with pytest.raises(ValueError):
        _alternation_grammar([])


def test_int_grammar_excludes():
    g = _int_grammar(1, 3, {2})
    assert '"1"' in g
    assert '"3"' in g
    assert '"2"' not in g


def test_int_grammar_all_excluded_raises():
    with pytest.raises(ValueError):
        _int_grammar(1, 2, {1, 2})


# ---- model-gated fixture -------------------------------------------


@pytest.fixture(scope="module")
def engine() -> XGrammarEngine:
    if not Path(MODEL_PATH).exists():
        pytest.skip("local GGUF not available")
    eng = XGrammarEngine(
        model_path=MODEL_PATH,
        n_ctx=1024,
        max_tokens_per_sample=16,
        seed=0,
    )
    eng.prime("You are a helpful assistant. Answer with the single word requested.\nAnswer: ")
    return eng


# ---- protocol coverage ---------------------------------------------


@needs_model
def test_sample_choice_returns_one_of_options(engine: XGrammarEngine) -> None:
    out = engine.sample_choice(["red", "blue", "green"])
    assert out in {"red", "blue", "green"}


@needs_model
def test_sample_int_in_range(engine: XGrammarEngine) -> None:
    out = engine.sample_int(1, 5)
    assert 1 <= out <= 5


@needs_model
def test_sample_int_excluded_is_never_returned(engine: XGrammarEngine) -> None:
    # Ask for 1..3 with 1 and 2 excluded: only 3 is grammar-legal.
    out = engine.sample_int(1, 3, excluded={1, 2})
    assert out == 3


@needs_model
def test_sample_bool_returns_bool(engine: XGrammarEngine) -> None:
    out = engine.sample_bool()
    assert isinstance(out, bool)


@needs_model
def test_sample_string_respects_charclass(engine: XGrammarEngine) -> None:
    out = engine.sample_string(max_len=6, pattern="[a-z]")
    assert len(out) >= 1
    assert len(out) <= 6
    assert all("a" <= c <= "z" for c in out)


# ---- ADR-0014 tightening end-to-end via gen.choice + where= --------


@needs_model
def test_choice_tightens_when_where_rejects(engine: XGrammarEngine) -> None:
    """Using the full dispatch path: gen.choice with a predicate that
    rejects "red" must never return "red", regardless of what the
    model's argmax would pick.
    """
    spec = gen.choice(["red", "blue", "green"], where=lambda o: o != "red")
    out = spec.dispatch(engine)
    assert out in {"blue", "green"}
    assert out != "red"


# ---- inject_context updates session state --------------------------


@needs_model
def test_inject_context_accumulates(engine: XGrammarEngine) -> None:
    start = len(engine._context)
    engine.inject_context("remember this")
    assert engine._context[-1] == "remember this"
    assert len(engine._context) == start + 1
    # Clean up so other tests aren't affected.
    engine._context.pop()
