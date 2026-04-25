"""Tests for ``orate.body_grammar.derive_body_grammar``.

We verify the structural shape of the emitted GBNF (for precision) and
– when XGrammar is available – that the emitted grammar actually
compiles under ``Grammar.from_ebnf``. Each program defined here
exercises one corner of the supported shape, plus all the unsupported
shapes from the spec.
"""

from __future__ import annotations

import pytest

from orate import BodyGrammarError, derive_body_grammar, gen, program
from orate.body_grammar import derive_body_grammar_rules


def _compile_with_xgrammar(grammar_body: str, root_name: str) -> None:
    """Wrap a body-grammar fragment in a root rule and compile it.

    The body-grammar module returns only the body rule + helpers. To
    feed XGrammar we add a minimal ``root ::= <name>_body``.
    """
    xgrammar = pytest.importorskip("xgrammar")
    full = f"root ::= {root_name}\n{grammar_body}"
    # Will raise on invalid GBNF.
    xgrammar.Grammar.from_ebnf(full)


# ---- shape tests ---------------------------------------------------


def test_single_choice_yields_alternation_of_literals():
    @program
    def pick_color():
        c = yield gen.choice(["red", "blue", "green"])
        return c

    body = derive_body_grammar(pick_color)
    assert "pick_color_body" in body
    # All three options present as quoted literals.
    assert '"red"' in body
    assert '"blue"' in body
    assert '"green"' in body
    assert "|" in body


def test_multiple_yields_comma_joined():
    @program
    def many():
        a = yield gen.choice(["x"])
        b = yield gen.boolean()
        c = yield gen.integer(1, 3)
        return {"a": a, "b": b, "c": c}

    body = derive_body_grammar(many)
    # Exactly two comma separators because three fragments.
    assert body.count('", "') == 2
    # The three fragments appear in order.
    body_line = [line for line in body.splitlines() if line.startswith("many_body ::=")][0]
    xi = body_line.index('"x"')
    ti = body_line.index('"true"')
    one_i = body_line.index('"1"')
    assert xi < ti < one_i


def test_integer_small_range_flat_alternation():
    @program
    def pick_n():
        n = yield gen.integer(3, 6)
        return n

    body = derive_body_grammar(pick_n)
    # Exactly four literals, no helper rule (under _INT_FLAT_MAX).
    for v in ("3", "4", "5", "6"):
        assert f'"{v}"' in body
    assert "pick_n_int_" not in body


def test_integer_large_range_uses_digit_dfa():
    @program
    def p():
        n = yield gen.integer(0, 9999)
        return n

    rules = derive_body_grammar_rules(p)
    # A helper rule should exist for the DFA.
    assert any(name.startswith("p_int_") for name in rules)
    # The helper rule should reference a digit char class (some variant
    # of ``[0-9]`` / ``[1-9]``) rather than listing every value.
    body_str = derive_body_grammar(p)
    assert "[0-9]" in body_str or "[1-9]" in body_str


def test_integer_large_range_compiles_under_xgrammar():
    @program
    def huge():
        n = yield gen.integer(0, 9999)
        return n

    _compile_with_xgrammar(derive_body_grammar(huge), "huge_body")


def test_boolean_emits_true_false_alternation():
    @program
    def flag():
        b = yield gen.boolean()
        return b

    body = derive_body_grammar(flag)
    assert '"true"' in body
    assert '"false"' in body


def test_string_max_len_emits_recursive_chars_rule():
    @program
    def name():
        s = yield gen.string(max_len=3)
        return s

    body = derive_body_grammar(name)
    # The string rule wraps a recursive chars-rest rule between quotes.
    # The cap (max_len) is enforced post-sample by the Session driver,
    # not in the grammar — so the helper compiles to a tight 2-state DFA.
    # Older shape was `char char? char? ...` which compiled to a
    # degenerate length-tracking automaton that hung on long strings.
    helper_lines = [line for line in body.splitlines() if line.startswith("name_str_")]
    assert helper_lines, f"expected name_str_* helper rules in: {body}"
    chars_rule = next(
        (line for line in helper_lines if "_chars" in line.split("::=")[0]),
        None,
    )
    assert chars_rule is not None, (
        f"expected a name_str_*_chars recursive rule in: {body}"
    )
    # Recursive shape: <chars> ::= <char_class> <chars> | ""
    assert "_chars" in chars_rule
    assert '""' in chars_rule  # the empty alternative is the recursion's base case
    # The wrapping rule opens + closes with a quote.
    wrap_rule = next(
        line for line in helper_lines if "_chars" not in line.split("::=")[0]
    )
    assert '"\\""' in wrap_rule


def test_string_with_explicit_pattern_uses_it():
    @program
    def slug():
        s = yield gen.string(max_len=5, pattern="[a-z]+")
        return s

    body = derive_body_grammar(slug)
    assert "[a-z]" in body


def test_empty_yield_sequence_gives_empty_body():
    @program
    def nothing():
        return None

    body = derive_body_grammar(nothing)
    # Rule resolves to the empty string literal.
    assert 'nothing_body ::= ""' in body


# ---- unsupported-shape tests --------------------------------------


def test_branches_rejected():
    @program
    def branchy():
        n = yield gen.integer(1, 10)
        if n > 5:
            n = yield gen.integer(1, 5)
        return n

    with pytest.raises(BodyGrammarError, match="If"):
        derive_body_grammar(branchy)


def test_loops_rejected():
    @program
    def loopy():
        for _ in range(3):
            n = yield gen.integer(1, 10)
        return n

    with pytest.raises(BodyGrammarError, match="For"):
        derive_body_grammar(loopy)


def test_yield_from_rejected():
    @program
    def delegator():
        n = yield from iter([gen.integer(1, 10)])
        return n

    with pytest.raises(BodyGrammarError, match="yield from"):
        derive_body_grammar(delegator)


def test_flavor_b_subprogram_yield_rejected():
    @program
    def sub():
        n = yield gen.integer(1, 3)
        return n

    @program
    def outer():
        # Yielding an arbitrary call (not gen.<method>) — Flavor-B style.
        n = yield sub()
        return n

    with pytest.raises(BodyGrammarError, match="non-gen call|sub-program"):
        derive_body_grammar(outer)


def test_dynamic_choice_options_rejected():
    options = ["a", "b"]

    @program
    def dyn():
        c = yield gen.choice(options)  # options is a bare name, not a list literal
        return c

    with pytest.raises(BodyGrammarError, match="list literal"):
        derive_body_grammar(dyn)


def test_non_literal_integer_bounds_rejected():
    lo = 1

    @program
    def badint():
        n = yield gen.integer(lo, 10)
        return n

    with pytest.raises(BodyGrammarError, match="integer literal"):
        derive_body_grammar(badint)


# ---- namespacing / collision avoidance ----------------------------


def test_per_program_rule_prefixing_avoids_collisions():
    @program
    def alpha():
        n = yield gen.integer(0, 999)
        _s = yield gen.string(max_len=2)
        return n

    @program
    def beta():
        n = yield gen.integer(0, 999)
        _s = yield gen.string(max_len=2)
        return n

    a = derive_body_grammar_rules(alpha)
    b = derive_body_grammar_rules(beta)
    # Rule namespaces are disjoint.
    assert set(a).isdisjoint(set(b))
    # Roots match the program name.
    assert "alpha_body" in a
    assert "beta_body" in b
    # Helpers carry the program name as prefix.
    assert all(k.startswith("alpha_") for k in a)
    assert all(k.startswith("beta_") for k in b)


# ---- end-to-end: grammar must compile under xgrammar --------------


def test_full_program_grammar_compiles_under_xgrammar():
    @program
    def trip():
        n = yield gen.integer(1, 10)
        c = yield gen.choice(["red", "blue"])
        b = yield gen.boolean()
        return {"n": n, "c": c, "b": b}

    _compile_with_xgrammar(derive_body_grammar(trip), "trip_body")


def test_string_grammar_compiles_under_xgrammar():
    @program
    def s():
        v = yield gen.string(max_len=4, pattern="[a-z]+")
        return v

    _compile_with_xgrammar(derive_body_grammar(s), "s_body")


def test_empty_body_compiles_under_xgrammar():
    @program
    def e():
        return None

    _compile_with_xgrammar(derive_body_grammar(e), "e_body")


# ---- two-tier: composers refuse derivation -------------------------


def test_composer_rejected_by_derive_body_grammar():
    @program(invocable=False)
    def dnd():
        # Composers can have control flow; they shouldn't be passed to
        # body-grammar derivation in the first place. The derivation
        # raises early with a pointed message rather than failing later
        # on the loop.
        while True:
            x = yield gen.choice(["a", "b"])
            if x == "a":
                return None

    with pytest.raises(BodyGrammarError, match="composer"):
        derive_body_grammar(dnd)


def test_composer_rejected_by_derive_call_arg_types():
    from orate.body_grammar import derive_call_arg_types

    @program(invocable=False)
    def dnd():
        n = yield gen.integer(0, 10)
        return n

    with pytest.raises(BodyGrammarError, match="composer"):
        derive_call_arg_types(dnd)


def test_invocable_default_is_true():
    @program
    def leaf():
        x = yield gen.boolean()
        return x

    # Every existing @program is a leaf; derivation works without
    # special-casing.
    body = derive_body_grammar(leaf)
    assert "leaf_body" in body
