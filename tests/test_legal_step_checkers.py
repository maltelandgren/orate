"""Unit tests for examples/legal_steps/checkers.py.

Covers algebra equivalence + per-rule sanity, and the small
propositional inference rules used in Act-4 Beat 2. These predicates
are the gate that makes the demo's "legal step only" claim real.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package alias.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "examples"))

from legal_steps.checkers import derivable_under, equivalent_under  # noqa: E402

# ---- algebra: positive cases -------------------------------------------


def test_isolate_var_simple():
    assert equivalent_under("isolate_var", "x + y = 5", "x = 5 - y")


def test_isolate_var_other_side():
    assert equivalent_under("isolate_var", "10 + y = 12", "y = 2")


def test_simplify_implicit_mult():
    assert equivalent_under("simplify", "2(5 - y) + 3y = 12", "10 + y = 12")


def test_evaluate_numeric_rhs():
    assert equivalent_under("evaluate", "x = 5 - 2", "x = 3")


def test_substitute_passthrough():
    # substitute is "any equivalent transformation" without per-rule constraint
    assert equivalent_under("substitute", "x + y = 5", "y = 5 - x")


def test_combine_like_terms():
    assert equivalent_under("combine_like", "2x + 3x = 10", "5x = 10")


# ---- algebra: negative cases -------------------------------------------


def test_arithmetic_error_rejected():
    # 10 - 2y + 3y = 10 + y, not 10 - y. The infamous LLM slip.
    assert not equivalent_under("simplify", "2(5 - y) + 3y = 12", "10 - y = 12")


def test_evaluate_requires_number():
    # Equivalent equation, but no numeric reduction → not "evaluate".
    assert not equivalent_under("evaluate", "x + y = 5", "y = 5 - x")


def test_isolate_var_not_alone_lhs():
    # Equivalent but lhs isn't a single Symbol.
    assert not equivalent_under("isolate_var", "x + y = 5", "x + y = 5")


def test_unknown_rule_rejected():
    assert not equivalent_under("teleport", "x = 5", "x = 5")


def test_unparseable_rejected():
    assert not equivalent_under("simplify", "x = ?+?", "x = 0")


def test_no_equals_sign_rejected():
    assert not equivalent_under("simplify", "x + y", "x + y")


# ---- logic: positive cases ---------------------------------------------


def test_modus_ponens():
    assert derivable_under("modus_ponens", ["P -> Q", "P"], "Q")


def test_modus_ponens_compound():
    assert derivable_under("modus_ponens", ["A & B -> C", "A & B"], "C")


def test_modus_tollens():
    assert derivable_under("modus_tollens", ["P -> Q", "~Q"], "~P")


def test_hypothetical_syllogism():
    assert derivable_under(
        "hypothetical_syllogism",
        ["P -> Q", "Q -> R"],
        "P -> R",
    )


def test_conjunction():
    assert derivable_under("conjunction", ["P", "Q"], "P & Q")


def test_simplification_left():
    assert derivable_under("simplification", ["P & Q"], "P")


def test_simplification_right():
    assert derivable_under("simplification", ["P & Q"], "Q")


# ---- logic: negative cases ---------------------------------------------


def test_modus_ponens_wrong_consequent():
    assert not derivable_under("modus_ponens", ["P -> Q", "P"], "R")


def test_modus_ponens_no_premise():
    assert not derivable_under("modus_ponens", ["P -> Q"], "Q")


def test_modus_tollens_wrong_negation():
    assert not derivable_under("modus_tollens", ["P -> Q", "Q"], "~P")


def test_hypothetical_syllogism_unrelated():
    # P -> Q and R -> S do not chain.
    assert not derivable_under(
        "hypothetical_syllogism",
        ["P -> Q", "R -> S"],
        "P -> S",
    )


def test_conjunction_missing_arm():
    assert not derivable_under("conjunction", ["P"], "P & Q")


def test_unknown_logic_rule():
    assert not derivable_under("disjunction_intro", ["P"], "P | Q")
