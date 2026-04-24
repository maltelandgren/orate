"""Tests for orate.meta — Act-4 true-meta-programming.

Covers: grammar compiles (gated on XGrammar), AST validator rejects
the full escape-hatch inventory, compiler raises ``MetaProgramInvalid``
on bad source, the returned callable is drivable against a MockEngine,
and a handful of hand-crafted source strings match/don't match the
grammar via XGrammar's matcher.
"""

from __future__ import annotations

import pytest

from orate import (
    PROGRAM_SOURCE_GRAMMAR,
    MetaProgramInvalid,
    ProgramInvocation,
    compile_program_source,
    validate_program_source,
)
from orate.engine.mock import MockEngine

VALID_SOURCE = '''@program
def example():
    n = yield gen.integer(10, 99)
    color = yield gen.choice(["red", "blue", "green"])
    flag = yield gen.boolean()
    s = yield gen.string(max_len=20)
    return {"a": n, "b": color, "c": flag, "d": s}
'''


# ---- validator -----------------------------------------------------------


def test_valid_source_has_no_errors():
    assert validate_program_source(VALID_SOURCE) == []


def test_syntax_error_returned_as_single_error():
    errors = validate_program_source("@program\ndef (\n")
    assert len(errors) == 1
    assert "syntax" in errors[0].lower()


def test_missing_program_decorator_rejected():
    source = '''def f():
    x = yield gen.boolean()
    return x
'''
    errors = validate_program_source(source)
    assert errors
    assert any("@program" in e for e in errors)


def test_extra_decorator_rejected():
    source = '''@staticmethod
@program
def f():
    x = yield gen.boolean()
    return x
'''
    errors = validate_program_source(source)
    assert errors
    assert any("decorator" in e for e in errors)


def test_program_decorator_with_arguments_rejected():
    source = '''@program(whole_program_retries=3)
def f():
    x = yield gen.boolean()
    return x
'''
    errors = validate_program_source(source)
    assert errors


def test_unknown_gen_method_rejected():
    source = '''@program
def f():
    x = yield gen.unknown_method()
    return x
'''
    errors = validate_program_source(source)
    assert errors
    assert any("unknown_method" in e for e in errors)


def test_call_to_open_rejected():
    # `open` isn't in any allowed position; both the call-walker and
    # the unbound-name walker should flag it.
    source = '''@program
def f():
    x = yield gen.boolean()
    y = open("/etc/passwd")
    return x
'''
    errors = validate_program_source(source)
    assert errors


def test_call_to_dunder_import_rejected():
    source = '''@program
def f():
    x = yield gen.boolean()
    y = __import__("os")
    return x
'''
    errors = validate_program_source(source)
    assert errors


def test_import_statement_rejected():
    source = '''@program
def f():
    import os
    x = yield gen.boolean()
    return x
'''
    errors = validate_program_source(source)
    assert errors
    assert any("import" in e for e in errors)


def test_attribute_access_on_non_gen_rejected():
    source = '''@program
def f():
    x = yield gen.boolean()
    y = x.upper
    return x
'''
    errors = validate_program_source(source)
    assert errors
    assert any("attribute" in e.lower() for e in errors)


def test_subscript_rejected():
    source = '''@program
def f():
    x = yield gen.choice(["a", "b"])
    return x
'''
    # valid baseline
    assert validate_program_source(source) == []
    # now with a subscript
    bad = '''@program
def f():
    x = yield gen.choice(["a", "b"])
    y = x[0]
    return x
'''
    errors = validate_program_source(bad)
    assert errors
    assert any("subscript" in e.lower() for e in errors)


def test_non_dict_non_name_return_rejected():
    # Return a list: neither a Name nor a Dict.
    source = '''@program
def f():
    x = yield gen.boolean()
    return [x]
'''
    errors = validate_program_source(source)
    assert errors
    assert any("return" in e.lower() for e in errors)


def test_unbound_name_in_return_rejected():
    source = '''@program
def f():
    x = yield gen.boolean()
    return ghost
'''
    errors = validate_program_source(source)
    assert errors
    assert any("ghost" in e for e in errors)


def test_unbound_name_in_return_dict_rejected():
    source = '''@program
def f():
    x = yield gen.boolean()
    return {"x": x, "y": phantom}
'''
    errors = validate_program_source(source)
    assert errors
    assert any("phantom" in e for e in errors)


def test_function_with_arguments_rejected():
    source = '''@program
def f(arg):
    x = yield gen.boolean()
    return x
'''
    errors = validate_program_source(source)
    assert errors
    assert any("argument" in e.lower() for e in errors)


def test_gen_string_positional_rejected():
    # gen.string must be keyword-only (max_len=).
    source = '''@program
def f():
    x = yield gen.string(20)
    return x
'''
    errors = validate_program_source(source)
    assert errors


def test_gen_choice_with_non_string_rejected():
    source = '''@program
def f():
    x = yield gen.choice([1, 2, 3])
    return x
'''
    errors = validate_program_source(source)
    assert errors


def test_gen_integer_negative_rejected():
    source = '''@program
def f():
    x = yield gen.integer(-5, 10)
    return x
'''
    errors = validate_program_source(source)
    assert errors


def test_collects_multiple_errors():
    # Two distinct bad things: disallowed method AND unbound return.
    source = '''@program
def f():
    x = yield gen.unknown_method()
    return nowhere
'''
    errors = validate_program_source(source)
    assert len(errors) >= 2


# ---- compiler ------------------------------------------------------------


def test_compile_invalid_raises():
    with pytest.raises(MetaProgramInvalid):
        compile_program_source("not even python @@@")


def test_compile_missing_decorator_raises():
    source = '''def f():
    x = yield gen.boolean()
    return x
'''
    with pytest.raises(MetaProgramInvalid):
        compile_program_source(source)


def test_compile_valid_returns_callable():
    compiled = compile_program_source(VALID_SOURCE)
    assert callable(compiled)


def test_compile_then_invoke_returns_invocation():
    compiled = compile_program_source(VALID_SOURCE)
    inv = compiled()
    assert isinstance(inv, ProgramInvocation)


def test_compiled_program_runs_against_mock_engine():
    compiled = compile_program_source(VALID_SOURCE)
    inv = compiled()
    result = inv.run(engine=MockEngine(seed=42))
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b", "c", "d"}
    assert 10 <= result["a"] <= 99
    assert result["b"] in {"red", "blue", "green"}
    assert isinstance(result["c"], bool)
    assert isinstance(result["d"], str)


def test_compiled_program_with_bare_name_return():
    source = '''@program
def choose():
    pick = yield gen.choice(["alpha", "beta"])
    return pick
'''
    compiled = compile_program_source(source)
    out = compiled().run(engine=MockEngine(seed=1))
    assert out in {"alpha", "beta"}


# ---- grammar (gated on xgrammar) -----------------------------------------


def test_grammar_compiles_with_xgrammar():
    xgrammar = pytest.importorskip("xgrammar")
    ti = xgrammar.TokenizerInfo(encoded_vocab=["a", "b", "c"], vocab_size=3)
    compiler = xgrammar.GrammarCompiler(ti)
    # Must compile without raising.
    compiler.compile_grammar(PROGRAM_SOURCE_GRAMMAR)


def _matcher():
    xgrammar = pytest.importorskip("xgrammar")
    ti = xgrammar.TokenizerInfo(encoded_vocab=["a", "b", "c"], vocab_size=3)
    compiler = xgrammar.GrammarCompiler(ti)
    compiled = compiler.compile_grammar(PROGRAM_SOURCE_GRAMMAR)
    return xgrammar.GrammarMatcher(compiled)


def test_grammar_accepts_full_valid_program():
    m = _matcher()
    assert m.accept_string(VALID_SOURCE) is True
    assert m.is_completed() is True


def test_grammar_accepts_minimal_program():
    m = _matcher()
    src = '''@program
def f():
    x = yield gen.boolean()
    return x
'''
    assert m.accept_string(src) is True
    assert m.is_completed() is True


def test_grammar_rejects_unquoted_string_in_choice():
    m = _matcher()
    bad = '''@program
def f():
    x = yield gen.choice([red])
    return x
'''
    assert m.accept_string(bad) is False


def test_grammar_rejects_disallowed_gen_method():
    m = _matcher()
    bad = '''@program
def f():
    x = yield gen.unknown()
    return x
'''
    assert m.accept_string(bad) is False


def test_grammar_rejects_missing_decorator():
    m = _matcher()
    bad = '''def f():
    x = yield gen.boolean()
    return x
'''
    assert m.accept_string(bad) is False
