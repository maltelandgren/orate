"""Host-provided predicate library for model-authored ``@program``s.

The model can write ``where=<name>(<bound_args>)`` in a meta-authored
program body. Each predicate here is a *factory*: it takes any
previously-yielded names that the source mentions and returns a
callable ``(candidate) -> bool`` that the runtime invokes during
predicate verification.

Convention:
    where=is_prime()                       # no bound args
    where=equivalent_under(rule, before)   # two bound args

The library function (``is_prime`` / ``equivalent_under`` / …) returns
a closure over the bound args; the closure takes the candidate value
and returns whether it satisfies the predicate. This matches the way
hand-authored ``where=lambda s: equivalent_under(rule, before, s)``
works — the meta path just curries the bound args at source-write
time so the model doesn't have to author the lambda itself.

Predicates added here become callable from any model-authored
program. The grammar / validator restricts the model to names from
this library; arbitrary callables are rejected.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

__all__ = [
    "META_PREDICATES",
    "coprime_with",
    "digit_sum_eq",
    "divides",
    "divisible_by",
    "equivalent_under",
    "factors_to",
    "gt",
    "is_palindrome",
    "is_prime",
    "is_square",
    "length_eq",
    "lt",
    "multiplies_to",
    "sums_to",
]


# ---- arithmetic -----------------------------------------------------------


def is_prime() -> Callable[[int], bool]:
    """Curried: ``where=is_prime()`` checks the candidate is prime.

    Candidate must be an int. Negative numbers and 0/1 return False.
    """

    def _check(n: int) -> bool:
        try:
            n = int(n)
        except (TypeError, ValueError):
            return False
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    return _check


def digit_sum_eq(target: int) -> Callable[[int], bool]:
    """Curried: ``where=digit_sum_eq(target)`` checks digits-of-candidate sum to target."""

    def _check(n: int) -> bool:
        try:
            n = int(n)
            target_int = int(target)
        except (TypeError, ValueError):
            return False
        return sum(int(d) for d in str(abs(n))) == target_int

    return _check


def lt(bound: int) -> Callable[[int], bool]:
    """Curried: ``where=lt(bound)`` checks candidate < bound."""

    def _check(n: int) -> bool:
        try:
            return int(n) < int(bound)
        except (TypeError, ValueError):
            return False

    return _check


def gt(bound: int) -> Callable[[int], bool]:
    """Curried: ``where=gt(bound)`` checks candidate > bound."""

    def _check(n: int) -> bool:
        try:
            return int(n) > int(bound)
        except (TypeError, ValueError):
            return False

    return _check


def multiplies_to(target: int, other: int) -> Callable[[int], bool]:
    """Curried: ``where=multiplies_to(target, other)`` checks
    ``candidate * other == target``. Useful for factoring problems."""

    def _check(n: int) -> bool:
        try:
            return int(n) * int(other) == int(target)
        except (TypeError, ValueError):
            return False

    return _check


def sums_to(target: int, other: int) -> Callable[[int], bool]:
    """Curried: ``where=sums_to(target, other)`` checks
    ``candidate + other == target``. Useful for Goldbach-style searches."""

    def _check(n: int) -> bool:
        try:
            return int(n) + int(other) == int(target)
        except (TypeError, ValueError):
            return False

    return _check


def divides(target: int) -> Callable[[int], bool]:
    """Curried: ``where=divides(target)`` checks ``target % candidate == 0``
    (i.e. the candidate is a non-zero divisor of ``target``)."""

    def _check(n: int) -> bool:
        try:
            n = int(n)
            if n == 0:
                return False
            return int(target) % n == 0
        except (TypeError, ValueError):
            return False

    return _check


def divisible_by(divisor: int) -> Callable[[int], bool]:
    """Curried: ``where=divisible_by(divisor)`` checks
    ``candidate % divisor == 0`` (i.e. ``divisor`` evenly divides
    the candidate)."""

    def _check(n: int) -> bool:
        try:
            d = int(divisor)
            if d == 0:
                return False
            return int(n) % d == 0
        except (TypeError, ValueError):
            return False

    return _check


def is_square() -> Callable[[int], bool]:
    """Curried: ``where=is_square()`` checks the candidate is a
    non-negative perfect square."""

    def _check(n: int) -> bool:
        try:
            n = int(n)
            if n < 0:
                return False
            r = math.isqrt(n)
            return r * r == n
        except (TypeError, ValueError):
            return False

    return _check


def is_palindrome() -> Callable[[Any], bool]:
    """Curried: ``where=is_palindrome()`` checks the candidate, read
    as a string, reads the same forward and backward. Works for both
    integer and string candidates."""

    def _check(v: Any) -> bool:
        s = str(v)
        return len(s) >= 1 and s == s[::-1]

    return _check


def coprime_with(other: int) -> Callable[[int], bool]:
    """Curried: ``where=coprime_with(other)`` checks
    ``gcd(candidate, other) == 1``."""

    def _check(n: int) -> bool:
        try:
            return math.gcd(int(n), int(other)) == 1
        except (TypeError, ValueError):
            return False

    return _check


def length_eq(target: int) -> Callable[[Any], bool]:
    """Curried: ``where=length_eq(target)`` checks ``len(str(candidate)) == target``.
    Convenient for digit-count constraints on integers."""

    def _check(v: Any) -> bool:
        try:
            return len(str(v)) == int(target)
        except (TypeError, ValueError):
            return False

    return _check


# ---- algebra (delegates to examples/legal_steps/checkers.py) -------------


def equivalent_under(rule: str, before: str) -> Callable[[str], bool]:
    """Curried: ``where=equivalent_under(rule, before)`` checks the
    candidate ``after`` is algebraically equivalent to ``before`` under
    ``rule`` (via SymPy, accepting same-form OR scalar-multiple).

    The implementation lives in examples/legal_steps/checkers.py;
    importing lazily so meta_predicates.py doesn't pull SymPy into
    every Session at import time.
    """

    def _check(after: str) -> bool:
        try:
            from examples.legal_steps.checkers import equivalent_under as impl  # noqa: PLC0415
        except ImportError:
            return False
        try:
            return bool(impl(rule, before, after))
        except Exception:  # noqa: BLE001
            return False

    return _check


def factors_to(target: str) -> Callable[[str], bool]:
    """Curried: ``where=factors_to(target)`` checks the candidate
    string parses as a polynomial factorisation that expands to
    ``target`` (modulo whitespace + ordering).

    Accepts strings like ``"(x - 2)(x - 3)"`` and verifies that
    expanding them gives ``target`` as a polynomial.
    """

    def _check(candidate: str) -> bool:
        try:
            import sympy  # noqa: PLC0415
            from sympy.parsing.sympy_parser import (  # noqa: PLC0415
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
        except ImportError:
            return False

        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        try:
            cand_expr = parse_expr(str(candidate), transformations=transformations)
            target_expr = parse_expr(str(target), transformations=transformations)
        except (SyntaxError, ValueError, TypeError):
            return False

        try:
            return bool(sympy.simplify(sympy.expand(cand_expr) - target_expr) == 0)
        except Exception:  # noqa: BLE001
            return False

    return _check


# ---- registry --------------------------------------------------------------

# The validator consults this dict to decide which predicate names the
# model is allowed to reference in a ``where=`` clause. Adding a new
# predicate here makes it accessible to model-authored programs.

META_PREDICATES: dict[str, Any] = {
    "is_prime": is_prime,
    "digit_sum_eq": digit_sum_eq,
    "lt": lt,
    "gt": gt,
    "multiplies_to": multiplies_to,
    "sums_to": sums_to,
    "divides": divides,
    "divisible_by": divisible_by,
    "is_square": is_square,
    "is_palindrome": is_palindrome,
    "coprime_with": coprime_with,
    "length_eq": length_eq,
    "equivalent_under": equivalent_under,
    "factors_to": factors_to,
}
