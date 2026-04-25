"""Tests for predicate-bound model authoring.

Meta-authored ``@program``s can include ``where=<lib_predicate>(<bound_args>)``
in their gen calls. The grammar admits this; the validator checks
the predicate name is in the library and args are previously
bound; the compiler exec's the source with the predicate library
in scope so the closures resolve.
"""

from __future__ import annotations

from orate import gen, program
from orate.engine.mock import MockEngine
from orate.meta import (
    PROGRAM_SOURCE_GRAMMAR,  # noqa: F401  — import-time sanity
    MetaProgramInvalid,
    compile_program_source,
    validate_program_source,
)
from orate.meta_predicates import META_PREDICATES, equivalent_under, is_prime


# ---- predicate-library smoke ----------------------------------------------


def test_is_prime_library_entry_is_curried() -> None:
    """is_prime() returns a callable that takes the candidate."""
    check = is_prime()
    assert check(2) is True
    assert check(3) is True
    assert check(4) is False
    assert check(11) is True
    assert check(15) is False
    assert check(0) is False
    assert check(-7) is False


def test_equivalent_under_curried() -> None:
    """equivalent_under(rule, before) returns a check on the candidate."""
    check = equivalent_under("simplify", "2x + 3 = 7")
    assert check("2x = 4") is True
    assert check("x = 99") is False  # not equivalent


def test_meta_predicates_dict_lists_all() -> None:
    """The library dict matches what the validator allows by name."""
    assert "is_prime" in META_PREDICATES
    assert "equivalent_under" in META_PREDICATES
    assert "factors_to" in META_PREDICATES


# ---- validation ------------------------------------------------------------


def test_validate_accepts_where_with_no_args() -> None:
    src = (
        "@program\n"
        "def find_prime():\n"
        "    n = yield gen.integer(0, 99, where=is_prime())\n"
        "    return n\n"
    )
    errors = validate_program_source(src)
    assert errors == [], errors


def test_validate_accepts_where_with_bound_args() -> None:
    src = (
        "@program\n"
        "def step():\n"
        '    rule = yield gen.choice(["simplify"])\n'
        '    before = yield gen.string(max_len=20)\n'
        "    after = yield gen.string(max_len=40, where=equivalent_under(rule, before))\n"
        "    return after\n"
    )
    errors = validate_program_source(src)
    assert errors == [], errors


def test_validate_rejects_unknown_predicate() -> None:
    src = (
        "@program\n"
        "def f():\n"
        "    n = yield gen.integer(0, 99, where=banana())\n"
        "    return n\n"
    )
    errors = validate_program_source(src)
    assert any("banana" in e and "library" in e for e in errors), errors


def test_validate_rejects_self_reference_in_where() -> None:
    """``where=is_prime(n)`` inside ``n = yield ...`` fails — n isn't bound yet."""
    src = (
        "@program\n"
        "def f():\n"
        "    n = yield gen.integer(0, 99, where=is_prime(n))\n"
        "    return n\n"
    )
    errors = validate_program_source(src)
    assert any("not bound" in e for e in errors), errors


def test_validate_rejects_where_with_unbound_arg() -> None:
    src = (
        "@program\n"
        "def f():\n"
        "    n = yield gen.integer(0, 99, where=equivalent_under(banana, n))\n"
        "    return n\n"
    )
    errors = validate_program_source(src)
    assert any("banana" in e or "not bound" in e for e in errors), errors


def test_validate_rejects_non_call_where() -> None:
    src = (
        "@program\n"
        "def f():\n"
        "    n = yield gen.integer(0, 99, where=42)\n"
        "    return n\n"
    )
    errors = validate_program_source(src)
    assert any("predicate call" in e for e in errors), errors


# ---- compile + exec --------------------------------------------------------


def test_compile_with_where_clause_returns_callable() -> None:
    src = (
        "@program\n"
        "def find_prime():\n"
        "    n = yield gen.integer(0, 99, where=is_prime())\n"
        "    return n\n"
    )
    compiled = compile_program_source(src)
    assert callable(compiled)
    invocation = compiled()
    assert hasattr(invocation, "run")


def test_compiled_program_runs_with_predicate_filtering() -> None:
    """End-to-end: model-authored is_prime predicate actually filters.

    The engine's witness-enumeration path narrows the int domain to
    primes. We can't easily check the masked sample without a real
    sampler, but we can confirm the callable runs and the predicate
    gets exercised at instantiation time.
    """
    src = (
        "@program\n"
        "def find_prime():\n"
        "    n = yield gen.integer(2, 20, where=is_prime())\n"
        "    return n\n"
    )
    compiled = compile_program_source(src)
    # Pre-flight: the body's first yield carries a where=callable.
    inv = compiled()
    body = inv.body(*inv.args, **inv.kwargs)
    spec = next(body)
    assert isinstance(spec, gen.Int)
    assert spec.where is not None
    # The host's library actually got injected — it returns True only on primes.
    assert spec.where(7) is True
    assert spec.where(8) is False
    assert spec.where(13) is True
    assert spec.where(15) is False


def test_compiled_program_runs_against_mock_engine_with_predicate() -> None:
    """Run a meta-authored predicate-bound program against MockEngine.

    The MockEngine's int sampler hits witness enumeration, which
    pre-filters via the predicate. The returned value should satisfy
    the predicate.
    """
    src = (
        "@program\n"
        "def find_prime():\n"
        "    n = yield gen.integer(2, 20, where=is_prime())\n"
        "    return n\n"
    )
    compiled = compile_program_source(src)
    engine = MockEngine(seed=42)
    result = compiled().run(engine=engine)
    # Whatever was sampled, it must be prime.
    assert is_prime()(result), f"expected prime, got {result}"


def test_curried_two_arg_predicate_compiles() -> None:
    """A two-arg predicate (rule, before) over a bound name resolves."""
    src = (
        "@program\n"
        "def step():\n"
        '    rule = yield gen.choice(["simplify"])\n'
        '    before = yield gen.string(max_len=20)\n'
        "    after = yield gen.string(max_len=40, where=equivalent_under(rule, before))\n"
        "    return after\n"
    )
    compiled = compile_program_source(src)
    assert callable(compiled)
