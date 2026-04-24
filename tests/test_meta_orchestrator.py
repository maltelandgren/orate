"""Tests for synthesize_program / meta_solve using MockEngine's canned source path.

MockEngine cannot actually synthesize Python from a grammar — it's a
random sampler — so these tests exercise the orchestrator's control
flow (accept, retry-on-invalid, exhaust) by feeding pre-written
sources through ``canned_grammar_source``. Real-model coverage lives
in ``examples/smoke_meta.py`` (model-gated).
"""

from __future__ import annotations

import pytest

from orate import MetaProgramInvalid, meta_solve, synthesize_program
from orate.engine.mock import MockEngine

_VALID_SOURCE = (
    "@program\n"
    "def pick():\n"
    "    n = yield gen.integer(1, 10)\n"
    '    c = yield gen.choice(["red", "blue"])\n'
    '    return {"n": n, "c": c}\n'
)


_INVALID_SOURCE_SYNTAX = (
    "@program\n"
    "def pick(:\n"  # syntax error
    "    return n\n"
)


_INVALID_SOURCE_UNKNOWN_METHOD = (
    "@program\ndef pick():\n    n = yield gen.unknown(1, 10)\n    return n\n"
)


def test_synthesize_accepts_valid_source_first_try():
    engine = MockEngine(seed=0, canned_grammar_source=_VALID_SOURCE)
    compiled, source, trace = synthesize_program(engine, task="pick two things")
    assert source.strip().endswith("}")  # ends with dict literal
    assert len(trace) == 1
    assert trace[0]["status"] == "accepted"
    assert callable(compiled)
    # Invoking returns a ProgramInvocation.
    invocation = compiled()
    assert hasattr(invocation, "run")


def test_synthesize_raises_without_sample_grammar():
    class NoGrammarEngine:
        def prime(self, _):
            pass

    with pytest.raises(TypeError, match="sample_grammar"):
        synthesize_program(NoGrammarEngine(), task="anything")


def test_synthesize_exhausts_retries_on_invalid_source():
    engine = MockEngine(seed=0, canned_grammar_source=_INVALID_SOURCE_UNKNOWN_METHOD)
    with pytest.raises(MetaProgramInvalid, match="synthesis failed"):
        synthesize_program(engine, task="pick", max_retries=2)


def test_synthesize_injects_reject_context_between_attempts():
    engine = MockEngine(seed=0, canned_grammar_source=_INVALID_SOURCE_UNKNOWN_METHOD)
    with pytest.raises(MetaProgramInvalid):
        synthesize_program(engine, task="pick", max_retries=2)
    # Between attempts 1→2 and 2→3, a context note was injected.
    assert len(engine._context) == 2
    for note in engine._context:
        assert "previous synthesis was rejected" in note


def test_synthesize_syntax_error_reported():
    engine = MockEngine(seed=0, canned_grammar_source=_INVALID_SOURCE_SYNTAX)
    with pytest.raises(MetaProgramInvalid):
        synthesize_program(engine, task="pick", max_retries=0)


def test_meta_solve_end_to_end_with_mock():
    """Whole loop: synth → compile → run. MockEngine returns the canned
    source, then (during phase-2) picks values for each yield."""
    engine = MockEngine(seed=0, canned_grammar_source=_VALID_SOURCE)
    result = meta_solve(engine, task="pick a number and a color")
    assert isinstance(result.value, dict)
    assert set(result.value.keys()) == {"n", "c"}
    assert 1 <= result.value["n"] <= 10
    assert result.value["c"] in {"red", "blue"}
    assert result.synthesis_attempts == 1
    assert result.source == _VALID_SOURCE


def test_meta_solve_preserves_synthesis_trace():
    engine = MockEngine(seed=0, canned_grammar_source=_VALID_SOURCE)
    result = meta_solve(engine, task="pick two things")
    assert len(result.trace) == 1
    assert result.trace[0]["status"] == "accepted"
    assert result.trace[0]["source"] == _VALID_SOURCE


def test_synthesize_records_each_attempt_in_trace():
    engine = MockEngine(seed=0, canned_grammar_source=_INVALID_SOURCE_UNKNOWN_METHOD)
    with pytest.raises(MetaProgramInvalid):
        synthesize_program(engine, task="pick", max_retries=2)
    # The underlying MockEngine doesn't expose the trace, but the
    # orchestrator's internal logic appended 3 entries (initial + 2 retries).
    # Verified indirectly by the count of injected context notes.
    assert len(engine._context) == 2


def test_max_retries_zero_means_one_attempt():
    engine = MockEngine(seed=0, canned_grammar_source=_VALID_SOURCE)
    _compiled, _source, trace = synthesize_program(engine, task="pick", max_retries=0)
    assert len(trace) == 1
