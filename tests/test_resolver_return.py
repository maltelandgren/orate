"""Tests for client-resolved tool calls — the @roll shape.

A leaf @program can have ordinary Python after its yields. The runtime
predicate-checks the model's positional args (via the yields), then
runs the post-yield code as the resolver, captures the body's return
value, and surfaces it as the call's result — both as the
``ProgramInvoked.result`` and as text appended to the KV so the model
sees the resolved value on the next sample.
"""

from __future__ import annotations

import json
import random

from orate import Session, gen, program


class _StubEngine:
    def __init__(self) -> None:
        self.appended: list[str] = []

    def begin_session(self, prompt: str) -> None:  # noqa: ARG002
        pass

    def append(self, text: str) -> None:
        self.appended.append(text)

    def sample_under(self, grammar: str, max_tokens: int = 256) -> str:  # noqa: ARG002
        return ""


@program
def roll():
    """A skill check — yields gate the args, post-yield code rolls the d20."""
    skill = yield gen.choice(["perception", "stealth", "athletics"])
    dc = yield gen.integer(5, 25)
    d20 = random.randint(1, 20)
    return {
        "skill": skill,
        "dc": dc,
        "d20": d20,
        "success": d20 >= dc,
    }


def _make_session() -> tuple[Session, _StubEngine]:
    engine = _StubEngine()
    session = Session(
        engine=engine,
        programs={"roll": roll},
        system="test",
        allow_free_text=False,
        max_calls_per_turn=2,
        max_turn_tokens=200,
    )
    return session, engine


def test_roll_resolves_and_returns_dict() -> None:
    """The body's return value becomes the ProgramInvoked.result."""
    random.seed(42)
    session, _ = _make_session()
    events = session._handle_call("@roll(perception, 15)")
    [invoked] = events
    assert invoked.name == "roll"
    assert invoked.args == ("perception", 15)
    assert invoked.result["skill"] == "perception"
    assert invoked.result["dc"] == 15
    assert isinstance(invoked.result["d20"], int)
    assert 1 <= invoked.result["d20"] <= 20
    assert invoked.result["success"] is (invoked.result["d20"] >= 15)


def test_roll_result_appended_to_kv() -> None:
    """The resolved result is appended to the KV so the model sees it."""
    random.seed(0)
    session, engine = _make_session()
    session._handle_call("@roll(stealth, 10)")
    # Find the " → {...}\n" append.
    arrow = next(a for a in engine.appended if a.lstrip().startswith("→"))
    payload = arrow.strip().removeprefix("→").strip()
    parsed = json.loads(payload)
    assert parsed["skill"] == "stealth"
    assert parsed["dc"] == 10
    assert "d20" in parsed
    assert "success" in parsed


def test_roll_predicate_rejects_bad_skill() -> None:
    """An out-of-options skill is rejected at arg-parse, before the resolver."""
    session, engine = _make_session()
    events = session._handle_call("@roll(arcana, 15)")
    # arg-parse fails (typed scan rejects 'arcana'); session emits no
    # ProgramInvoked, but the rejection reason is in the KV tape.
    assert events == []
    rejection = next(
        a for a in engine.appended if "arg parse failed" in a or "rejected" in a
    )
    assert "perception" in rejection  # error message lists the legal choice set


def test_roll_predicate_rejects_dc_out_of_range() -> None:
    """A DC outside [5, 25] rejects."""
    session, _ = _make_session()
    events = session._handle_call("@roll(perception, 100)")
    [invoked] = events
    assert invoked.result.get("rejected") is True
    assert "range" in invoked.result["error"].lower() or "100" in invoked.result["error"]


@program
def algebra_step():
    """Existing-shape leaf with a return value (no post-yield code).

    The return value should still be surfaced — backward-compat.
    """
    before = yield gen.string(max_len=20)
    rule = yield gen.choice(["a", "b"])
    return {"before": before, "rule": rule}


def test_existing_program_return_value_used() -> None:
    """For all leaves, the body's return value populates result."""
    engine = _StubEngine()
    session = Session(
        engine=engine,
        programs={"algebra_step": algebra_step},
        system="test",
        allow_free_text=False,
    )
    events = session._handle_call('@algebra_step("x = 1", a)')
    [invoked] = events
    assert invoked.result == {"before": "x = 1", "rule": "a"}
