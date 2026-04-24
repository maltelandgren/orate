"""Compile a Gen's `where=` predicate into a tightened accept set.

The dominant strategy — witness enumeration — is trivial and fast: for
any finite domain small enough to enumerate (default budget: 10k
values), evaluate the predicate on every candidate and return the
accepted list. The Gen's dispatcher can then feed the narrowed set to
the engine directly (compiled as a grammar alternation), so the engine
cannot emit a rejected value and the runtime retry loop never fires.

Empirical probe (bench/probe_constraint_strategies.py) showed this
dominates LMQL-style AST pattern matching and z3 SMT for every
predicate orate's users actually write — including opaque Python
helpers like `is_prime(x)` and `digit_sum(x) == 10` that SMT cannot
model. Predicates over unbounded domains (long strings, huge integer
ranges) return None here and fall back to today's rejection-sampling +
tightening path in the Gen's dispatcher.

Forward-checking across struct fields is the same primitive applied
sequentially: after field A is sampled, close the cross-field
predicate over A's value and compile field B's remaining domain. See
`compile_struct_field` below.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

DEFAULT_ENUM_BUDGET = 10_000


def _safe_eval(predicate: Callable[[Any], bool], value: Any) -> bool:
    """Run a single-field predicate, treating exceptions as rejection.

    For a Gen's own ``where=``, a predicate that crashes on a given
    value cannot be satisfied by that value — crash is rejection.
    """
    try:
        return bool(predicate(value))
    except Exception:
        return False


def _safe_eval_tristate(
    predicate: Callable[[Any], bool],
    value: Any,
) -> bool | None:
    """Run a cross-field predicate. Returns True/False, or None on error.

    Used by forward-checking, where a raise typically means "the
    predicate referenced a sibling field that hasn't been bound yet" —
    in that case the caller should NOT narrow the domain on this
    iteration (we can't make a decision). Distinct from single-field
    ``_safe_eval``, where raising is treated as rejection.
    """
    try:
        return bool(predicate(value))
    except Exception:
        return None


def enumerate_choice(
    options: Sequence[str],
    where: Callable[[str], bool] | None,
) -> list[str]:
    """Compile a Choice's accept set. Always enumerable — options is finite."""
    if where is None:
        return list(options)
    return [o for o in options if _safe_eval(where, o)]


def enumerate_int(
    min_val: int,
    max_val: int,
    where: Callable[[int], bool] | None,
    *,
    budget: int = DEFAULT_ENUM_BUDGET,
) -> list[int] | None:
    """Compile an Int's accept set, or None if the domain exceeds budget.

    Returning None signals the caller (Gen.dispatch) to fall back to
    today's rejection-sampling loop; the domain is too large to
    enumerate at compile time.
    """
    size = max_val - min_val + 1
    if size <= 0:
        return []
    if size > budget:
        return None
    candidates = range(min_val, max_val + 1)
    if where is None:
        return list(candidates)
    return [v for v in candidates if _safe_eval(where, v)]


def enumerate_bool(where: Callable[[bool], bool] | None) -> list[bool]:
    """Compile a Bool's accept set. Two values; always enumerable."""
    if where is None:
        return [True, False]
    return [v for v in (True, False) if _safe_eval(where, v)]


def compile_struct_field(
    field_spec: Any,
    bound: dict[str, Any],
    cross_field_where: Callable[[dict], bool] | None,
    *,
    budget: int = DEFAULT_ENUM_BUDGET,
) -> list[Any] | None:
    """Forward-check: narrow this field's domain given already-bound siblings.

    Closes ``cross_field_where`` over ``bound``, producing a single-variable
    predicate on the candidate value. Combines it with the field's own
    ``where=`` (if any) and returns the enumerated accept set.

    Returns None if the field's domain isn't enumerable (e.g. String) —
    caller falls back to the field's native dispatch.

    Typical usage from ``Struct.dispatch``::

        after sampling x=3 in struct(x=int[0,10], y=int[0,10], where=x+y==10):
        compile_struct_field(y_spec, bound={'x': 3}, cross_field_where=...)
        -> [7]
    """
    # Local import to avoid a cycle with gen.py (compile imports gen
    # types; gen dispatches call compile).
    from orate.gen import Bool, Choice, Int

    # Build a combined predicate: field.where AND cross_field_where(bound ∪ {field: v}).
    # When the cross predicate raises (typically because it references a
    # sibling that's not yet bound), don't narrow — we can't decide yet.
    def combined(value: Any, _name: str) -> bool:
        if cross_field_where is not None:
            trial = dict(bound)
            trial[_name] = value
            result = _safe_eval_tristate(cross_field_where, trial)
            if result is False:
                return False
            # result is True or None (couldn't evaluate) → keep candidate.
        return True

    if isinstance(field_spec, Choice):
        own = enumerate_choice(field_spec.options, field_spec.where)
        name = _field_name_hint(field_spec, bound)
        return [o for o in own if combined(o, name)]

    if isinstance(field_spec, Int):
        own = enumerate_int(
            field_spec.min_val,
            field_spec.max_val,
            field_spec.where,
            budget=budget,
        )
        if own is None:
            return None
        name = _field_name_hint(field_spec, bound)
        return [v for v in own if combined(v, name)]

    if isinstance(field_spec, Bool):
        own = enumerate_bool(field_spec.where)
        name = _field_name_hint(field_spec, bound)
        return [v for v in own if combined(v, name)]

    return None


def _field_name_hint(field_spec: Any, bound: dict[str, Any]) -> str:
    """Return the struct-field name we're compiling for, or a fallback.

    Struct.dispatch knows the name and should pass it explicitly in a
    future refactor; for now we leave this hook — callers pass the
    field name via the outer Struct iteration.
    """
    return getattr(field_spec, "_field_name", "_")
