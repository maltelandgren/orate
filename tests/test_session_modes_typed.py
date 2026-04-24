"""Unit tests for Session's mode-aware registry + typed-arg parsing.

No model: a stub engine satisfies the begin_session / append /
sample_under interface so we can construct a Session and inspect its
bookkeeping without running real inference.
"""

from __future__ import annotations

import pytest

from orate import gen, program
from orate.body_grammar import ArgType, derive_call_arg_types
from orate.session import DEFAULT_MODE, Session, _scan_typed_args


class _StubEngine:
    """Minimum surface to construct a Session without real inference."""

    def __init__(self) -> None:
        self.appended: list[str] = []
        self.primed: str = ""

    def begin_session(self, prompt: str) -> None:
        self.primed = prompt

    def append(self, text: str) -> None:
        self.appended.append(text)

    def sample_under(self, grammar: str, max_tokens: int) -> str:  # noqa: ARG002
        return ""


# ---- programs used as fixtures ------------------------------------------


@program
def greet():
    g = yield gen.choice(["hi", "hello"])
    return g


@program(mode_transition="combat")
def enter_combat():
    target = yield gen.choice(["alpha", "bravo"])
    return target


@program(mode_transition="default")
def exit_combat():
    reason = yield gen.choice(["killed", "fled"])
    return reason


@program
def attack():
    n = yield gen.integer(1, 20)
    return n


@program
def speak():
    line = yield gen.string(max_len=80)
    return line


@program
def mixed_args():
    name = yield gen.string(max_len=20)
    n = yield gen.integer(0, 100)
    flag = yield gen.boolean()
    return (name, n, flag)


# ---- mode-aware registry ------------------------------------------------


def test_default_mode_visible_at_construction():
    s = Session(engine=_StubEngine(), programs={"greet": greet})
    assert s.active_mode == DEFAULT_MODE
    assert "greet" in s._outer_grammar


def test_mode_scoped_program_hidden_in_default():
    s = Session(engine=_StubEngine(), programs={"greet": greet})
    s.register("attack", attack, mode="combat")
    assert "greet" in s._outer_grammar
    assert '"@attack(' not in s._outer_grammar


def test_mode_scoped_program_visible_in_active_mode():
    s = Session(engine=_StubEngine(), programs={"greet": greet})
    s.register("attack", attack, mode="combat")
    s.set_mode("combat")
    assert s.active_mode == "combat"
    assert '"@attack(' in s._outer_grammar
    # Unscoped 'greet' still visible
    assert '"@greet(' in s._outer_grammar


def test_mode_transition_fires_after_invocation():
    s = Session(engine=_StubEngine(), programs={"enter_combat": enter_combat})
    assert s.active_mode == DEFAULT_MODE
    s._handle_call("@enter_combat(alpha)")
    assert s.active_mode == "combat"


def test_round_trip_default_combat_default():
    s = Session(engine=_StubEngine(), programs={"enter_combat": enter_combat})
    s.register("exit_combat", exit_combat, mode="combat")
    s.register("attack", attack, mode="combat")

    # Default → combat
    s._handle_call("@enter_combat(alpha)")
    assert s.active_mode == "combat"
    assert '"@attack(' in s._outer_grammar
    assert '"@exit_combat(' in s._outer_grammar

    # Combat → default
    s._handle_call("@exit_combat(killed)")
    assert s.active_mode == DEFAULT_MODE
    assert '"@attack(' not in s._outer_grammar


def test_make_new_program_visible_in_every_mode():
    s = Session(engine=_StubEngine(), programs={"enter_combat": enter_combat})
    assert '"@make_new_program(' in s._outer_grammar
    s.set_mode("combat")
    assert '"@make_new_program(' in s._outer_grammar


# ---- arg-type derivation -----------------------------------------------


def test_derive_arg_types_choice():
    types = derive_call_arg_types(greet)
    assert len(types) == 1
    assert types[0].kind == "choice"
    assert types[0].options == ("hi", "hello")


def test_derive_arg_types_integer():
    types = derive_call_arg_types(attack)
    assert len(types) == 1
    assert types[0].kind == "integer"
    assert (types[0].lo, types[0].hi) == (1, 20)


def test_derive_arg_types_string():
    types = derive_call_arg_types(speak)
    assert types[0].kind == "string"
    assert types[0].max_len == 80


def test_derive_arg_types_mixed():
    types = derive_call_arg_types(mixed_args)
    assert [t.kind for t in types] == ["string", "integer", "boolean"]


# ---- typed scanning -----------------------------------------------------


def test_scan_typed_args_integer():
    assert _scan_typed_args("42", [ArgType(kind="integer", lo=0, hi=100)]) == (42,)


def test_scan_typed_args_negative_integer():
    assert _scan_typed_args("-7", [ArgType(kind="integer", lo=-10, hi=10)]) == (-7,)


def test_scan_typed_args_boolean_true():
    assert _scan_typed_args("true", [ArgType(kind="boolean")]) == (True,)


def test_scan_typed_args_boolean_false():
    assert _scan_typed_args("false", [ArgType(kind="boolean")]) == (False,)


def test_scan_typed_args_string():
    assert _scan_typed_args('"hello"', [ArgType(kind="string", max_len=20)]) == ("hello",)


def test_scan_typed_args_string_with_escaped_quote():
    assert _scan_typed_args(
        r'"a \"b\" c"', [ArgType(kind="string", max_len=20)]
    ) == ('a "b" c',)


def test_scan_typed_args_string_with_escaped_backslash():
    assert _scan_typed_args(
        r'"x \\ y"', [ArgType(kind="string", max_len=20)]
    ) == ("x \\ y",)


def test_scan_typed_args_string_with_comma_content():
    """Naive split on ', ' would fail here; type-driven scan must not."""
    assert _scan_typed_args(
        '"hello, world"', [ArgType(kind="string", max_len=20)]
    ) == ("hello, world",)


def test_scan_typed_args_choice_bare():
    """Body grammar emits choices bare, not JSON-quoted."""
    types = [ArgType(kind="choice", options=("red", "blue"))]
    assert _scan_typed_args("red", types) == ("red",)


def test_scan_typed_args_choice_picks_longest_match():
    """When one option is a prefix of another, prefer the longer match."""
    types = [ArgType(kind="choice", options=("foo", "foobar"))]
    assert _scan_typed_args("foobar", types) == ("foobar",)


def test_scan_typed_args_mixed():
    types = [
        ArgType(kind="string", max_len=10),
        ArgType(kind="integer", lo=0, hi=100),
        ArgType(kind="boolean"),
    ]
    assert _scan_typed_args('"foo", 42, true', types) == ("foo", 42, True)


def test_scan_typed_args_arity_mismatch_raises():
    with pytest.raises(ValueError, match="trailing unparsed"):
        _scan_typed_args("42, 99", [ArgType(kind="integer")])


def test_scan_typed_args_missing_separator_raises():
    with pytest.raises(ValueError, match="', '"):
        _scan_typed_args('42 99', [ArgType(kind="integer"), ArgType(kind="integer")])


def test_scan_typed_args_unterminated_string_raises():
    with pytest.raises(ValueError, match="unterminated"):
        _scan_typed_args('"abc', [ArgType(kind="string", max_len=20)])


# ---- end-to-end through Session._handle_call ---------------------------


def test_handle_call_decodes_typed_args():
    s = Session(engine=_StubEngine(), programs={"mixed_args": mixed_args})
    events = s._handle_call('@mixed_args("foo", 42, true)')
    # ProgramInvoked event with parsed args.
    invoked = events[0]
    assert invoked.args == ("foo", 42, True)


# ---- predicate verification --------------------------------------------


@program
def chained_predicate():
    """Cross-yield where: second value must equal first + 1."""
    n = yield gen.integer(0, 100)
    succ = yield gen.integer(0, 100, where=lambda x: x == n + 1)
    return {"n": n, "succ": succ}


@program
def out_of_range_value():
    """For testing range-violation on a parsed arg."""
    n = yield gen.integer(0, 10)
    return n


def test_predicate_passes_for_valid_chain():
    s = Session(engine=_StubEngine(), programs={"chained_predicate": chained_predicate})
    events = s._handle_call("@chained_predicate(5, 6)")
    invoked = events[0]
    # Successful call: result has emitted_args, no rejection
    assert "rejected" not in invoked.result


def test_predicate_rejects_invalid_chain():
    s = Session(engine=_StubEngine(), programs={"chained_predicate": chained_predicate})
    events = s._handle_call("@chained_predicate(5, 9)")  # 9 != 5+1
    invoked = events[0]
    assert invoked.result.get("rejected") is True
    assert "where=" in invoked.result["error"]


def test_scanner_rejects_unknown_choice():
    """Bare-word scanner returns empty events for unknown choices."""
    s = Session(engine=_StubEngine(), programs={"greet": greet})
    events = s._handle_call("@greet(weird)")
    # Scanner couldn't match → arg parse failed → empty events list.
    assert events == []
    # Engine got a session note about it.
    assert any("arg parse failed" in t for t in s.engine.appended)
