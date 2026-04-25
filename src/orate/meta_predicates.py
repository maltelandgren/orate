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

from collections.abc import Callable
from typing import Any

__all__ = [
    "META_PREDICATES",
    "digit_sum_eq",
    "equivalent_under",
    "factors_to",
    "is_prime",
    "lt",
    "gt",
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
    "equivalent_under": equivalent_under,
    "factors_to": factors_to,
}
