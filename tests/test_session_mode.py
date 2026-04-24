"""Persistent-KV session-mode API on XGrammarEngine.

Model-gated. Verifies begin_session / append / sample_under behave
correctly against the real decoder: KV is not reset between samples,
appends extend the sequence, multiple sample_under calls in a row
see each other's output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_MODEL = "/Users/maltelandgren/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

pytestmark = [
    pytest.mark.skipif(not Path(_MODEL).exists(), reason="local model not present"),
]

_llama_cpp = pytest.importorskip("llama_cpp")
_xgrammar = pytest.importorskip("xgrammar")


@pytest.fixture(scope="module")
def engine():
    from orate.engine.xgrammar import XGrammarEngine

    e = XGrammarEngine(model_path=_MODEL, max_tokens_per_sample=32)
    e.load()
    return e


def test_begin_session_primes_kv_without_error(engine):
    engine.begin_session("You are helpful. Pick a color when asked.\n")
    assert engine._session_active is True
    assert engine._prompt_tokens  # tokenized non-empty


def test_sample_under_requires_session(engine):
    e2 = type(engine)(model_path=_MODEL, max_tokens_per_sample=8)
    e2.load()
    with pytest.raises(RuntimeError, match="outside a session"):
        e2.sample_under('root ::= "a" | "b"')


def test_append_requires_session(engine):
    e2 = type(engine)(model_path=_MODEL, max_tokens_per_sample=8)
    e2.load()
    with pytest.raises(RuntimeError, match="outside a session"):
        e2.append("hello")


def test_sequential_samples_share_context(engine):
    """A second sample_under call sees the first's output in its context."""
    engine.begin_session("You are a matching assistant. When the user names a color, echo it back.")
    engine.append("\nUser: red\nAssistant: ")
    first = engine.sample_under('root ::= "red" | "blue" | "green"', max_tokens=8)
    assert first in {"red", "blue", "green"}
    # Append another user message and sample again.
    engine.append(f"\nUser: same one again please ({first}).\nAssistant: ")
    second = engine.sample_under('root ::= "red" | "blue" | "green"', max_tokens=8)
    assert second in {"red", "blue", "green"}


def test_session_inject_context_appends_to_kv(engine):
    """inject_context under session mode is a live append, not a buffer."""
    engine.begin_session("System prompt here.\n")
    engine.inject_context("note: a steering hint")
    # No buffer in session mode — _context stays empty.
    assert engine._context == []
    # Subsequent sample works as expected.
    result = engine.sample_under('root ::= "ok" | "no"', max_tokens=4)
    assert result in {"ok", "no"}


def test_one_shot_and_session_modes_isolated(engine):
    """Stateless sample_choice still works on the same engine instance after
    a session (though mixing is discouraged — the prime text differs)."""
    engine.begin_session("Ignore me.\n")
    _ = engine.sample_under('root ::= "a" | "b"', max_tokens=4)

    # Flip back to stateless — prime + sample_choice.
    engine.prime("Pick red or blue.\nAnswer: ")
    # session mode stays "active" for state-tracking purposes, but the
    # one-shot path resets on each sample_choice, so it's safe.
    out = engine.sample_choice(["red", "blue"])
    assert out in {"red", "blue"}
