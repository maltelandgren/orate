"""Predicates for the Act-4 legal-step demos.

Two checkers, both parameterised by a *rule* label (string) and a
pair of expressions/premises (input → output). Returning True means the
output is a *legal* transformation of the input under the named rule;
returning False means it is not — in a @program, this becomes a
``where=`` predicate that the runner enforces token-by-token (or
post-hoc, as the orate Session does for body-grammar emissions).

`equivalent_under` (algebra)
----------------------------
Parses two equations of the form "lhs = rhs" using SymPy with implicit
multiplication enabled (so "2(5-y)" parses), then checks that the two
equations are mathematically equivalent. Some rule labels add light
sanity checks: ``evaluate`` requires a numeric RHS, ``isolate_var``
requires the LHS to be a single Symbol.

`derivable_under` (propositional logic)
---------------------------------------
A small dispatch table over the inference rules used in the demo:
``modus_ponens``, ``modus_tollens``, ``hypothetical_syllogism``,
``conjunction``, ``simplification``. Premises and conclusion are
plain text in the form ``P``, ``P -> Q``, ``~P``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

__all__ = [
    "ALGEBRA_RULES",
    "LOGIC_RULES",
    "derivable_under",
    "equivalent_under",
    "parse_equation",
    "parse_proposition",
]


# ---- algebra ------------------------------------------------------------

ALGEBRA_RULES: tuple[str, ...] = (
    "substitute",
    "simplify",
    "combine_like",
    "isolate_var",
    "evaluate",
)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def parse_equation(text: str) -> tuple[sympy.Expr, sympy.Expr]:
    """Parse "lhs = rhs" into a pair of SymPy expressions.

    Raises ValueError on shape mismatch and propagates SympifyError /
    SyntaxError on un-parseable subexpressions. We strip whitespace,
    enable implicit multiplication so ``2(5-y)`` parses naturally, and
    refuse strings without a single ``=``.
    """
    if text.count("=") != 1:
        raise ValueError(f"expected a single '=' in equation; got {text!r}")
    lhs_s, rhs_s = text.split("=", 1)
    lhs = parse_expr(lhs_s.strip(), transformations=_TRANSFORMATIONS)
    rhs = parse_expr(rhs_s.strip(), transformations=_TRANSFORMATIONS)
    return lhs, rhs


def equivalent_under(rule: str, before: str, after: str) -> bool:
    """Return True if ``after`` is a legal transformation of ``before``.

    *Legal* means:
      1. Both sides parse as equations.
      2. ``after`` and ``before`` describe the same solution set.
         This admits both same-form equivalence (``p1 - p2 == 0`` after
         simplify) and scalar multiplication of either side by a
         nonzero constant — so ``2x = 8`` ↔ ``x = 4`` counts as legal.
      3. Per-rule sanity:
         - ``evaluate``: ``after``'s RHS (or LHS) must be a pure Number.
         - ``isolate_var``: ``after``'s LHS must be a single Symbol.
         - other rules: equivalence alone is sufficient.

    Returns False on any parse / evaluation failure rather than
    propagating exceptions — predicate semantics demand a clean bool.
    """
    if rule not in ALGEBRA_RULES:
        return False
    try:
        lhs1, rhs1 = parse_equation(before)
        lhs2, rhs2 = parse_equation(after)
    except (ValueError, SyntaxError, sympy.SympifyError, TypeError):
        return False

    p1 = sympy.simplify(lhs1 - rhs1)
    p2 = sympy.simplify(lhs2 - rhs2)

    same_polynomial = sympy.simplify(p1 - p2) == 0
    scalar_multiple = False
    if not same_polynomial and p2 != 0:
        try:
            ratio = sympy.simplify(p1 / p2)
            scalar_multiple = ratio.is_number and ratio != 0
        except (sympy.SympifyError, TypeError, ZeroDivisionError):
            scalar_multiple = False

    if not (same_polynomial or scalar_multiple):
        return False

    if rule == "evaluate" and not (lhs2.is_number or rhs2.is_number):
        # The point of "evaluate" is to land a numeric value.
        return False
    return not (rule == "isolate_var" and not isinstance(lhs2, sympy.Symbol))


# ---- propositional logic -----------------------------------------------

LOGIC_RULES: tuple[str, ...] = (
    "modus_ponens",
    "modus_tollens",
    "hypothetical_syllogism",
    "conjunction",
    "simplification",
)


# Tokenise a proposition. We handle the connectives the demo uses:
# ``->`` (implies), ``&`` (and), ``|`` (or), ``~`` (not). Identifiers
# are letters/digits. We don't fully parse — just normalise whitespace
# and strip outer parentheses.
_WS_RE = re.compile(r"\s+")


def parse_proposition(text: str) -> str:
    """Normalise a proposition string for equality comparison.

    Whitespace collapsed, outer matching parens stripped, ``->`` and
    ``=>`` unified to ``->``. Returns the normalised string. Two
    propositions are considered equal iff their normalised forms match.
    """
    s = _WS_RE.sub("", text)
    s = s.replace("=>", "->")
    while s.startswith("(") and s.endswith(")") and _matches_outer(s):
        s = s[1:-1]
    return s


def _matches_outer(s: str) -> bool:
    """True if the leading '(' matches the trailing ')'."""
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


def _split_implication(p: str) -> tuple[str, str] | None:
    """Split ``A -> B`` at the top level. None if no top-level ``->``."""
    depth = 0
    i = 0
    while i < len(p) - 1:
        c = p[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and c == "-" and p[i + 1] == ">":
            return parse_proposition(p[:i]), parse_proposition(p[i + 2 :])
        i += 1
    return None


def _split_conjunction(p: str) -> tuple[str, str] | None:
    """Split ``A & B`` at the top level (left-associative; we just need any split)."""
    depth = 0
    for i, c in enumerate(p):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and c == "&":
            return parse_proposition(p[:i]), parse_proposition(p[i + 1 :])
    return None


def _negation_of(p: str) -> str:
    """Return the negation of p, normalised."""
    if p.startswith("~"):
        return parse_proposition(p[1:])
    return parse_proposition(f"~{p}")


def _is_negation_of(a: str, b: str) -> bool:
    """True if a = ~b or b = ~a (after normalisation)."""
    return _negation_of(a) == parse_proposition(b) or _negation_of(b) == parse_proposition(a)


def derivable_under(
    rule: str,
    premises: Iterable[str],
    conclusion: str,
) -> bool:
    """Return True if ``conclusion`` follows from ``premises`` under ``rule``.

    Supported rules:

    - ``modus_ponens``: from ``P -> Q`` and ``P``, conclude ``Q``.
    - ``modus_tollens``: from ``P -> Q`` and ``~Q``, conclude ``~P``.
    - ``hypothetical_syllogism``: from ``P -> Q`` and ``Q -> R``,
      conclude ``P -> R``.
    - ``conjunction``: from ``P`` and ``Q``, conclude ``P & Q``.
    - ``simplification``: from ``P & Q``, conclude ``P`` (or ``Q``).

    Premises are matched as multisets — order doesn't matter as long
    as every position in the rule has a witness.
    """
    if rule not in LOGIC_RULES:
        return False
    try:
        prems = [parse_proposition(p) for p in premises]
        conc = parse_proposition(conclusion)
    except (ValueError, TypeError):
        return False

    if rule == "modus_ponens":
        # need P -> Q and P, conclude Q
        for impl in prems:
            split = _split_implication(impl)
            if split is None:
                continue
            antecedent, consequent = split
            if antecedent in prems and consequent == conc:
                return True
        return False

    if rule == "modus_tollens":
        # need P -> Q and ~Q, conclude ~P
        for impl in prems:
            split = _split_implication(impl)
            if split is None:
                continue
            antecedent, consequent = split
            for p in prems:
                if _is_negation_of(p, consequent) and _is_negation_of(conc, antecedent):
                    return True
        return False

    if rule == "hypothetical_syllogism":
        # need P -> Q and Q -> R, conclude P -> R
        split_conc = _split_implication(conc)
        if split_conc is None:
            return False
        p_outer, r_outer = split_conc
        for a in prems:
            sa = _split_implication(a)
            if sa is None:
                continue
            pa, qa = sa
            if pa != p_outer:
                continue
            for b in prems:
                if b is a:
                    continue
                sb = _split_implication(b)
                if sb is None:
                    continue
                qb, rb = sb
                if qb == qa and rb == r_outer:
                    return True
        return False

    if rule == "conjunction":
        split = _split_conjunction(conc)
        if split is None:
            return False
        left, right = split
        return left in prems and right in prems

    if rule == "simplification":
        # need P & Q, conclude P (or Q)
        for p in prems:
            split = _split_conjunction(p)
            if split is None:
                continue
            left, right = split
            if conc in (left, right):
                return True
        return False

    return False
