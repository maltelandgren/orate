"""Integration: legal-step @programs through Session._handle_call.

We don't run a real LLM here; instead we feed @-call strings directly
into the session's call-dispatch and verify that
- valid emissions are accepted (no 'rejected' marker)
- invalid emissions (bad arithmetic, illegal deduction) are rejected
  with a useful error message

This is the closest you can get to "the demo works" without booting a
Qwen-7B GGUF.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "examples"))

from legal_steps.algebra import algebra_step  # noqa: E402
from legal_steps.logic import inference_step  # noqa: E402

from orate import Session  # noqa: E402


class _StubEngine:
    def __init__(self) -> None:
        self.appended: list[str] = []

    def begin_session(self, prompt: str) -> None:
        self.appended.append(f"<system>{prompt}</system>")

    def append(self, text: str) -> None:
        self.appended.append(text)

    def sample_under(self, grammar: str, max_tokens: int) -> str:  # noqa: ARG002
        return ""


def _make_session(name: str, fn) -> Session:
    return Session(engine=_StubEngine(), programs={name: fn})


# ---- algebra ------------------------------------------------------------


def test_algebra_isolate_var_accepted():
    s = _make_session("algebra_step", algebra_step)
    events = s._handle_call('@algebra_step("x + y = 5", isolate_var, "x = 5 - y")')
    invoked = events[0]
    assert "rejected" not in invoked.result, invoked.result


def test_algebra_simplify_accepted():
    s = _make_session("algebra_step", algebra_step)
    events = s._handle_call(
        '@algebra_step("2(5 - y) + 3y = 12", simplify, "10 + y = 12")'
    )
    invoked = events[0]
    assert "rejected" not in invoked.result, invoked.result


def test_algebra_evaluate_accepted():
    s = _make_session("algebra_step", algebra_step)
    events = s._handle_call('@algebra_step("x = 5 - 2", evaluate, "x = 3")')
    invoked = events[0]
    assert "rejected" not in invoked.result, invoked.result


def test_algebra_arithmetic_error_rejected():
    """The famous LLM slip: 10 - 2y + 3y → '10 - y' instead of '10 + y'."""
    s = _make_session("algebra_step", algebra_step)
    events = s._handle_call(
        '@algebra_step("2(5 - y) + 3y = 12", simplify, "10 - y = 12")'
    )
    invoked = events[0]
    assert invoked.result.get("rejected") is True
    assert "where=" in invoked.result["error"]


def test_algebra_evaluate_rejects_non_numeric():
    s = _make_session("algebra_step", algebra_step)
    events = s._handle_call(
        '@algebra_step("x + y = 5", evaluate, "y = 5 - x")'
    )
    invoked = events[0]
    # rule says 'evaluate' but rhs isn't a pure number → rejected
    assert invoked.result.get("rejected") is True


def test_algebra_isolate_var_rejects_non_symbol_lhs():
    s = _make_session("algebra_step", algebra_step)
    events = s._handle_call(
        '@algebra_step("x + y = 5", isolate_var, "x + y = 5")'
    )
    invoked = events[0]
    assert invoked.result.get("rejected") is True


# ---- logic --------------------------------------------------------------


def test_logic_modus_ponens_accepted():
    s = _make_session("inference_step", inference_step)
    events = s._handle_call(
        '@inference_step("P -> Q; P", modus_ponens, "Q")'
    )
    invoked = events[0]
    assert "rejected" not in invoked.result, invoked.result


def test_logic_hypothetical_syllogism_accepted():
    s = _make_session("inference_step", inference_step)
    events = s._handle_call(
        '@inference_step("P -> Q; Q -> R", hypothetical_syllogism, "P -> R")'
    )
    invoked = events[0]
    assert "rejected" not in invoked.result, invoked.result


def test_logic_modus_ponens_wrong_conclusion_rejected():
    s = _make_session("inference_step", inference_step)
    events = s._handle_call(
        '@inference_step("P -> Q; P", modus_ponens, "R")'
    )
    invoked = events[0]
    assert invoked.result.get("rejected") is True


def test_logic_unrelated_premises_rejected():
    s = _make_session("inference_step", inference_step)
    events = s._handle_call(
        '@inference_step("P -> Q; R -> S", hypothetical_syllogism, "P -> S")'
    )
    invoked = events[0]
    assert invoked.result.get("rejected") is True


# ---- chained algebra: full system solve --------------------------------


def test_algebra_chain_solves_system():
    """Drives a full sequence of @algebra_step calls through the session.

    Models the exact chain the Act-4 demo expects Qwen-7B to produce.
    Every intermediate emission must pass predicate verification, end-
    to-end. This is the strongest test we can write without a real LLM.
    """
    s = _make_session("algebra_step", algebra_step)
    chain = [
        '@algebra_step("x + y = 5", isolate_var, "x = 5 - y")',
        '@algebra_step("2(5 - y) + 3y = 12", simplify, "10 + y = 12")',
        '@algebra_step("10 + y = 12", isolate_var, "y = 2")',
        '@algebra_step("x = 5 - 2", evaluate, "x = 3")',
    ]
    for call in chain:
        events = s._handle_call(call)
        invoked = events[0]
        assert "rejected" not in invoked.result, (call, invoked.result)
