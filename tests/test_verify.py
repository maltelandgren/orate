"""First-class verifiers: Accept/Reject, the @verifier decorator, and
the runtime dispatch that turns a Reject into a Phase-C-catchable
ProgramRejected."""

from __future__ import annotations

import pytest

from orate import (
    Accept,
    ProgramRejected,
    Reject,
    gen,
    program,
    verifier,
)
from orate.engine.mock import MockEngine

# --- decorator + factory shape --------------------------------------------


def test_verifier_decorator_bare():
    @verifier
    def always_ok(_x):
        return Accept()

    call = always_ok(42)
    assert call.name == "always_ok"
    assert call.args == (42,)


def test_verifier_decorator_with_description():
    @verifier(description="checks non-negativity")
    def nonneg(x):
        return Accept() if x >= 0 else Reject("negative")

    call = nonneg(5)
    assert call.description == "checks non-negativity"
    assert call.name == "nonneg"


def test_verifier_preserves_wrapped_function():
    @verifier
    def myfn(x):
        return Accept()

    # The decorated factory has __wrapped__ pointing at the original.
    assert callable(myfn.__wrapped__)


# --- integration: verifier inside @program --------------------------------


def test_verifier_accept_lets_program_continue():
    engine = MockEngine(seed=0)

    @verifier
    def nonneg(x):
        return Accept() if x >= 0 else Reject(f"{x} is negative")

    @program
    def pick():
        n = yield gen.integer(1, 10)
        yield nonneg(n)
        return n

    result = pick().run(engine=engine)
    assert 1 <= result <= 10


def test_verifier_reject_raises_program_rejected():
    engine = MockEngine(seed=0)

    @verifier
    def always_fails(_x):
        return Reject("synthetic failure")

    @program
    def pick():
        n = yield gen.integer(1, 10)
        yield always_fails(n)
        return n

    with pytest.raises(ProgramRejected, match="synthetic failure"):
        pick().run(engine=engine)


def test_verifier_reject_triggers_phase_c_retry():
    """When the outer @program has retries and the verifier rejects,
    the program rewinds and retries. Context note is injected."""
    engine = MockEngine(seed=0)
    attempts = {"n": 0}

    @verifier
    def needs_two_attempts(_x):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return Reject("first try rejected by verifier")
        return Accept()

    @program(whole_program_retries=3)
    def pick():
        n = yield gen.integer(1, 10)
        yield needs_two_attempts(n)
        return n

    result = pick().run(engine=engine)
    assert 1 <= result <= 10
    assert attempts["n"] == 2
    # Phase-C injected a note between attempt 1 and attempt 2.
    assert any("first try rejected" in note for note in engine._context)


def test_multiple_verifiers_all_must_accept():
    engine = MockEngine(seed=0)

    @verifier
    def positive(x):
        return Accept() if x > 0 else Reject("not positive")

    @verifier
    def less_than_7(x):
        return Accept() if x < 7 else Reject("not less than 7")

    @program(whole_program_retries=30)
    def pick():
        n = yield gen.integer(1, 10)
        yield positive(n)
        yield less_than_7(n)
        return n

    result = pick().run(engine=engine)
    assert 0 < result < 7


def test_verifier_returning_non_accept_reject_is_typeerror():
    engine = MockEngine(seed=0)

    @verifier
    def returns_bool(_x):
        return True  # type: ignore[return-value]

    @program
    def pick():
        _ = yield gen.integer(1, 3)
        yield returns_bool(1)
        return "done"

    with pytest.raises(TypeError, match="must return Accept"):
        pick().run(engine=engine)


def test_verifier_name_in_reject_message():
    engine = MockEngine(seed=0)

    @verifier
    def max_five(x):
        if x > 5:
            return Reject(f"{x} exceeds 5")
        return Accept()

    @program
    def pick():
        n = yield gen.integer(6, 10)
        yield max_five(n)
        return n

    try:
        pick().run(engine=engine)
    except ProgramRejected as e:
        assert "max_five" in str(e)
        assert "exceeds 5" in str(e)
    else:
        raise AssertionError("expected ProgramRejected")


def test_verifier_composition_with_context_kwargs():
    """Verifiers take arbitrary context via **kwargs."""
    engine = MockEngine(seed=0)

    @verifier
    def in_range(x, *, lo, hi):
        if lo <= x <= hi:
            return Accept()
        return Reject(f"{x} outside [{lo}, {hi}]")

    @program
    def pick():
        n = yield gen.integer(1, 100)
        yield in_range(n, lo=1, hi=100)  # always passes
        return n

    result = pick().run(engine=engine)
    assert 1 <= result <= 100
