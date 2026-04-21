"""Phase B (reject-message context injection) + Phase C (whole-program retry).

Phase B: when a yield's predicate rejects and the Gen has a
reject_message, the engine's inject_context is called before the next
sample. Verified by inspecting MockEngine._context.

Phase C: @program(whole_program_retries=N) catches ProgramRejected and
GrammarExhausted at the program boundary, rewinds by re-invoking the
body, and injects a program-level reject message.
"""

from __future__ import annotations

import pytest

from orate import GrammarExhausted, ProgramRejected, gen, program, reject_program
from orate.engine.mock import MockEngine


def test_phase_b_reject_message_injected_on_predicate_fail():
    """Narrow predicate guarantees at least one rejection, so _context is non-empty."""
    engine = MockEngine(seed=0)

    @program
    def pick_17():
        n = yield gen.integer(
            1,
            20,
            where=lambda x: x == 17,
            reject_message=lambda v: f"{v} is not 17",
            max_retries=50,
        )
        return n

    result = pick_17().run(engine=engine)
    assert result == 17
    # With a single-value predicate over 20 candidates, rejections are
    # effectively certain. Every note in context should be properly formatted.
    assert len(engine._context) >= 1, "Phase-B did not inject on rejection"
    for note in engine._context:
        assert "is not 17" in note
        assert note.startswith("(note:")


def test_phase_b_string_reject_message_not_callable():
    """String form of reject_message (not a callable) also injects."""
    engine = MockEngine(seed=0)

    @program
    def pick_only_a():
        c = yield gen.choice(
            ["a", "b", "c", "d", "e"],
            where=lambda x: x == "a",
            reject_message="that was not 'a'",
        )
        return c

    result = pick_only_a().run(engine=engine)
    assert result == "a"
    # With 5 options and 1 accepted, seed=0 sampling picks "b" first.
    assert len(engine._context) >= 1
    for note in engine._context:
        assert "not 'a'" in note


def test_phase_b_no_reject_message_means_no_injection():
    engine = MockEngine(seed=0)

    @program
    def pick():
        c = yield gen.choice(["a", "b", "c"], where=lambda x: x == "a")
        return c

    pick().run(engine=engine)
    assert engine._context == []


def test_phase_c_whole_program_retry_on_program_rejected():
    engine = MockEngine(seed=0)
    attempts = {"n": 0}

    @program(whole_program_retries=3)
    def eventually_succeeds():
        attempts["n"] += 1
        _ = yield gen.choice(["x"])
        if attempts["n"] < 3:
            reject_program(f"attempt {attempts['n']} not good enough")
        return "finally"

    result = eventually_succeeds().run(engine=engine)
    assert result == "finally"
    assert attempts["n"] == 3
    # Program-level reject context injected twice (before attempts 2 and 3).
    assert len(engine._context) >= 2


def test_phase_c_raises_after_retries_exhausted():
    engine = MockEngine(seed=0)

    @program(whole_program_retries=2)
    def always_fails():
        _ = yield gen.choice(["x"])
        reject_program("nope")

    with pytest.raises(ProgramRejected, match="nope"):
        always_fails().run(engine=engine)


def test_phase_c_catches_grammar_exhausted():
    engine = MockEngine(seed=0)
    attempts = {"n": 0}

    @program(whole_program_retries=2)
    def impossible():
        attempts["n"] += 1
        _ = yield gen.choice(["x"], where=lambda v: False)
        return "unreachable"

    with pytest.raises(GrammarExhausted):
        impossible().run(engine=engine)
    # Body was re-invoked on each attempt.
    assert attempts["n"] == 3


def test_phase_c_custom_reject_message_callable():
    engine = MockEngine(seed=0)

    @program(
        whole_program_retries=1,
        reject_message=lambda attempt, exc: f"attempt {attempt} failed: {exc}",
    )
    def fails_then_succeeds():
        _ = yield gen.choice(["x"])
        if not hasattr(fails_then_succeeds, "_called"):
            fails_then_succeeds._called = True
            reject_program("first try")
        return "done"

    fails_then_succeeds().run(engine=engine)
    assert any("attempt 0 failed" in note for note in engine._context)


def test_phase_c_no_retries_by_default():
    engine = MockEngine(seed=0)

    @program
    def fails_immediately():
        _ = yield gen.choice(["x"])
        reject_program("no retry requested")

    with pytest.raises(ProgramRejected):
        fails_immediately().run(engine=engine)


def test_program_trace_records_attempts():
    engine = MockEngine(seed=0)
    attempts = {"n": 0}

    @program(whole_program_retries=3)
    def succeeds_second_try():
        attempts["n"] += 1
        _ = yield gen.choice(["x"])
        if attempts["n"] < 2:
            reject_program("nope")
        return "ok"

    invocation = succeeds_second_try()
    invocation.run(engine=engine)
    assert len(invocation.trace) == 2
    assert invocation.trace[0]["status"] == "rejected"
    assert invocation.trace[1]["status"] == "ok"
