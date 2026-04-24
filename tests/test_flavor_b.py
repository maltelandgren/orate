"""Flavor B — yielding a ProgramInvocation from a @program body.

Minimal semantics: yielded ProgramInvocation is run via ``.run(engine=...)``
on the outer engine; the return value becomes the yield's result. The inner
invocation manages its own whole_program_retries; the outer only sees
rejections the inner couldn't absorb.

Also covers the ``ends_turn`` metadata flag plumbed by the @program
decorator through to ProgramInvocation.
"""

from __future__ import annotations

import pytest

from orate import ProgramRejected, gen, program, reject_program
from orate.engine.mock import MockEngine


def test_yielding_bare_gen_still_works():
    """Regression: a @program that only yields Gen values is unaffected."""
    engine = MockEngine(seed=0)

    @program
    def pick():
        c = yield gen.choice(["a", "b", "c"])
        return c

    result = pick().run(engine=engine)
    assert result in {"a", "b", "c"}


def test_yielding_program_invocation_runs_subprogram_and_returns_value():
    engine = MockEngine(seed=0)

    @program
    def inner():
        c = yield gen.choice(["inner-x"])
        return f"inner:{c}"

    @program
    def outer():
        sub = yield inner()
        return f"outer[{sub}]"

    result = outer().run(engine=engine)
    assert result == "outer[inner:inner-x]"


def test_subprogram_yields_fire_against_same_engine():
    """The inner program's yields should hit the outer's engine, observed
    via the MockEngine's _context (used for Phase-B injection) and its
    sampling record. Easiest witness: count how many yields resolve on
    the shared engine by driving several inner choices and asserting the
    returned values could only have come from the engine we passed in."""
    engine = MockEngine(seed=0)

    @program
    def inner():
        a = yield gen.choice(["A1"])
        b = yield gen.choice(["B1"])
        return (a, b)

    @program
    def outer():
        outer_pick = yield gen.choice(["O1"])
        sub = yield inner()
        return (outer_pick, sub)

    result = outer().run(engine=engine)
    assert result == ("O1", ("A1", "B1"))
    # Sanity: the same MockEngine instance was used; we know this because
    # the inner's deterministic singleton choices returned the canned values
    # — if a fresh engine had been built it would still work, so we also
    # assert via context injection in the next test.


def test_subprogram_program_rejected_propagates_to_outer_phase_c():
    """The inner program has no retries; its ProgramRejected propagates up
    and the outer's Phase-C retry catches it."""
    engine = MockEngine(seed=0)
    inner_calls = {"n": 0}

    @program
    def inner_always_rejects():
        inner_calls["n"] += 1
        _ = yield gen.choice(["x"])
        reject_program("inner no good")

    outer_attempts = {"n": 0}

    @program(whole_program_retries=2)
    def outer():
        outer_attempts["n"] += 1
        if outer_attempts["n"] < 3:
            _ = yield inner_always_rejects()
        _ = yield gen.choice(["final"])
        return "done"

    result = outer().run(engine=engine)
    assert result == "done"
    # Outer ran 3 times; each of the first 2 called the inner body once.
    assert outer_attempts["n"] == 3
    assert inner_calls["n"] == 2


def test_subprogram_absorbs_its_own_rejections_without_bothering_outer():
    """If the inner has its own whole_program_retries big enough, the outer
    never sees a ProgramRejected."""
    engine = MockEngine(seed=0)
    inner_calls = {"n": 0}

    @program(whole_program_retries=3)
    def inner_eventually_succeeds():
        inner_calls["n"] += 1
        _ = yield gen.choice(["x"])
        if inner_calls["n"] < 3:
            reject_program(f"inner attempt {inner_calls['n']} fails")
        return "inner-ok"

    outer_attempts = {"n": 0}

    @program  # no retries on outer
    def outer():
        outer_attempts["n"] += 1
        sub = yield inner_eventually_succeeds()
        return sub

    result = outer().run(engine=engine)
    assert result == "inner-ok"
    assert outer_attempts["n"] == 1  # outer only ran once
    assert inner_calls["n"] == 3  # inner absorbed its own 2 rejections


def test_yielding_non_gen_non_program_invocation_raises_type_error():
    engine = MockEngine(seed=0)

    @program
    def bad():
        yield 42  # not a Gen, not a ProgramInvocation
        return "unreachable"

    with pytest.raises(TypeError) as excinfo:
        bad().run(engine=engine)
    msg = str(excinfo.value)
    assert "Gen" in msg
    assert "ProgramInvocation" in msg


def test_ends_turn_flag_sets_invocation_field():
    @program(ends_turn=True)
    def finalize():
        _ = yield gen.choice(["x"])
        return "final"

    invocation = finalize()
    assert invocation.ends_turn is True


def test_ends_turn_defaults_to_false():
    @program
    def no_flag():
        _ = yield gen.choice(["x"])
        return "ok"

    assert no_flag().ends_turn is False

    @program(whole_program_retries=2)
    def no_flag_with_retries():
        _ = yield gen.choice(["x"])
        return "ok"

    assert no_flag_with_retries().ends_turn is False


def test_ends_turn_preserved_across_nested_phase_c_retries():
    """An inner invocation's ends_turn is a stable field on the ProgramInvocation
    object and must be readable regardless of how many times the outer's
    Phase-C loop rewinds (each rewind re-invokes the body, which builds a
    fresh inner invocation — but all of them carry the flag)."""
    engine = MockEngine(seed=0)
    seen_ends_turn: list[bool] = []

    @program(ends_turn=True)
    def inner():
        _ = yield gen.choice(["x"])
        return "inner-ok"

    outer_attempts = {"n": 0}

    @program(whole_program_retries=2)
    def outer():
        outer_attempts["n"] += 1
        sub_invocation = inner()
        seen_ends_turn.append(sub_invocation.ends_turn)
        if outer_attempts["n"] < 3:
            _ = yield sub_invocation
            reject_program("force retry")
        _ = yield sub_invocation
        return "outer-ok"

    result = outer().run(engine=engine)
    assert result == "outer-ok"
    # 3 outer attempts, each built a fresh inner invocation; every one
    # carried ends_turn=True.
    assert seen_ends_turn == [True, True, True]


def test_subprogram_phase_b_context_lands_on_shared_engine():
    """Independent witness that inner yields hit the outer's engine: a
    Phase-B reject_message injection on an inner Gen should appear in
    the outer engine's _context buffer."""
    engine = MockEngine(seed=0)

    @program
    def inner():
        s = yield gen.string(
            max_len=6,
            where=lambda x: "q" in x,
            reject_message="inner wants a 'q'",
            max_retries=40,
        )
        return s

    @program
    def outer():
        sub = yield inner()
        return sub

    result = outer().run(engine=engine)
    assert "q" in result
    assert any("inner wants a 'q'" in note for note in engine._context)


def test_program_rejected_from_outer_after_subprogram_rejection_exhaustion():
    """Inner exhausts its own retries, outer has none — ProgramRejected leaks."""
    engine = MockEngine(seed=0)

    @program
    def inner_rejects():
        _ = yield gen.choice(["x"])
        reject_program("inner final")

    @program
    def outer():
        _ = yield inner_rejects()
        return "unreachable"

    with pytest.raises(ProgramRejected, match="inner final"):
        outer().run(engine=engine)
