"""Empirical probe: which constraint-strengthening technique handles which predicate?

Three techniques, six representative predicates. For each cell we answer:
  - can the technique STATICALLY narrow the domain?
  - how much narrowing (|accepted set| / |original domain|)?
  - wall time
  - any surprise
"""

from __future__ import annotations

import ast
import inspect
import time
from typing import Any, Callable


# --- predicate zoo --------------------------------------------------------


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


PREDICATES: dict[str, tuple[tuple[int, int], Callable[[int], bool]]] = {
    "eq17":        ((1, 20),    lambda x: x == 17),
    "gt50":        ((1, 100),   lambda x: x > 50),
    "mod7_eq_3":   ((1, 100),   lambda x: x % 7 == 3),
    "prime":       ((2, 100),   is_prime),
    "prime_ds10":  ((10, 99),   lambda x: is_prime(x) and digit_sum(x) == 10),
    "wide_eq":    ((1, 10000),  lambda x: x == 7777),  # huge domain, single-value
}


# --- technique 1: witness enumeration --------------------------------------


def witness_enumerate(domain_range, predicate, *, budget: int = 10_000):
    """Evaluate predicate on every value in domain. Return accepted set + timing."""
    lo, hi = domain_range
    size = hi - lo + 1
    if size > budget:
        return None  # too big
    t0 = time.perf_counter()
    accepted = [v for v in range(lo, hi + 1) if predicate(v)]
    dt = time.perf_counter() - t0
    return accepted, dt


# --- technique 2: AST pattern-match compiler (LMQL-style) ------------------


def try_lmql_style_compile(predicate):
    """Return 'what we can statically prove' or None.

    Pattern-match: x == C, x != C, x < C, x > C, x in S, x % k == r,
    and chains of AND over those.
    """
    try:
        src = inspect.getsource(predicate).strip()
    except (OSError, TypeError):
        return None

    # Extract the lambda body: `lambda x: BODY` or `def f(x): return BODY`
    try:
        tree = ast.parse(src.strip().rstrip(","))
    except SyntaxError:
        return None

    lambda_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambda_node = node
            break
    if lambda_node is None:
        return None  # named function or more complex — skip

    body = lambda_node.body
    if not isinstance(body, ast.Compare):
        if isinstance(body, ast.BoolOp) and isinstance(body.op, ast.And):
            parts = [_compare_summary(c) for c in body.values]
            return parts if all(parts) else None
        return None
    return [_compare_summary(body)] if _compare_summary(body) else None


def _compare_summary(node):
    """Describe one Compare node in a form the compiler could use."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    op = node.ops[0]
    left = node.left
    right = node.comparators[0]
    if not (isinstance(left, ast.Name) and isinstance(right, ast.Constant)):
        return None
    op_name = type(op).__name__
    return f"x {op_name} {right.value}"


# --- technique 3: z3 SMT oracle --------------------------------------------


def z3_check_numeric_range(domain_range, op_name: str, constant):
    """For simple predicates, ask z3 which values are feasible.

    Returns (feasible_values, wall_time) if z3 can answer; None if predicate
    shape isn't directly expressible.
    """
    import z3

    lo, hi = domain_range
    x = z3.Int("x")

    op_map = {
        "Eq": x == constant,
        "NotEq": x != constant,
        "Lt": x < constant,
        "Gt": x > constant,
        "LtE": x <= constant,
        "GtE": x >= constant,
    }
    if op_name not in op_map:
        return None

    t0 = time.perf_counter()
    # Enumerate all feasible values under the constraint.
    # (For ranges this small, z3 is overkill — but we're validating the tool.)
    feasible = []
    for v in range(lo, hi + 1):
        s = z3.Solver()
        s.add(x == v, op_map[op_name], lo <= x, x <= hi)
        if s.check() == z3.sat:
            feasible.append(v)
    dt = time.perf_counter() - t0
    return feasible, dt


def z3_check_compound(domain_range, *, must_be_prime=False, digit_sum_eq=None):
    """Test z3's practical speed on 'prime and digit_sum==10' via bit-level encoding.

    z3 CAN model primality but it's slow — we let z3 enumerate via
    repeated check-sat-model loops here. Instructive for honest measurement.
    """
    import z3

    lo, hi = domain_range
    feasible = []

    for v in range(lo, hi + 1):
        if must_be_prime:
            if v < 2:
                continue
            if any(v % i == 0 for i in range(2, int(v**0.5) + 1)):
                continue
        if digit_sum_eq is not None:
            if digit_sum(v) != digit_sum_eq:
                continue
        feasible.append(v)
    # z3 isn't providing value here — primality is decidable, just not in
    # SMT-LIB's standard theory. This is the honest answer: z3 does not
    # help for predicates with opaque Python semantics unless you model
    # the semantics in SMT yourself (which for primality is expensive).
    return feasible


# --- cross-field propagation probe -----------------------------------------


def forward_check_sum_eq_constant():
    """After sampling x=3 in struct(x=int[0,10], y=int[0,10], x+y==10),
    what's the narrowed domain of y?"""
    x_domain = range(0, 11)
    y_domain = range(0, 11)
    target = 10
    # Before any binding:
    before = [(x, y) for x in x_domain for y in y_domain if x + y == target]
    # After x=3:
    x = 3
    narrowed_y = [y for y in y_domain if x + y == target]
    # General: for each x, what's y's allowed set?
    per_x = {x: [y for y in y_domain if x + y == target] for x in x_domain}
    return {
        "pairs_before": len(before),
        "narrowed_y_after_x=3": narrowed_y,
        "per_x_cardinality": {x: len(ys) for x, ys in per_x.items()},
    }


# --- run it ----------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("TECHNIQUE 1 — WITNESS ENUMERATION")
    print("=" * 80)
    print(f"{'predicate':<15} {'domain_size':>11} {'accepted':>9} {'time_ms':>10}  verdict")
    for name, (dom, pred) in PREDICATES.items():
        size = dom[1] - dom[0] + 1
        result = witness_enumerate(dom, pred)
        if result is None:
            print(f"{name:<15} {size:>11} {'--':>9} {'--':>10}  domain too big for enumeration")
            continue
        accepted, dt = result
        density = len(accepted) / size
        verdict = (
            "TRIVIAL — compile to {accepted} directly"
            if len(accepted) <= 20
            else "alternation over sparse set, still a win"
        )
        verdict = verdict.format(accepted=accepted[:5])
        print(f"{name:<15} {size:>11} {len(accepted):>9} {dt*1000:>10.2f}  {verdict}")

    print()
    print("=" * 80)
    print("TECHNIQUE 2 — LMQL-STYLE AST PATTERN MATCH")
    print("=" * 80)
    print(f"{'predicate':<15} pattern-match result")
    for name, (_, pred) in PREDICATES.items():
        summary = try_lmql_style_compile(pred)
        print(f"{name:<15} {summary}")

    print()
    print("=" * 80)
    print("TECHNIQUE 3 — z3 SMT ORACLE")
    print("=" * 80)
    # Pick two predicates that z3 CAN express and one it can't.
    print(f"gt50 on [1,100]  → {z3_check_numeric_range((1, 100), 'Gt', 50)[1]*1000:.1f}ms, "
          f"{len(z3_check_numeric_range((1,100),'Gt',50)[0])} feasible")
    print(f"eq17 on [1,20]   → {z3_check_numeric_range((1, 20), 'Eq', 17)[1]*1000:.1f}ms, "
          f"feasible = {z3_check_numeric_range((1,20),'Eq',17)[0]}")
    print("primality        → z3 does NOT help directly; primality is not in the "
          "standard SMT signature. Use witness enumeration instead.")

    print()
    print("=" * 80)
    print("CROSS-FIELD PROPAGATION PROBE: struct(x+y==10)")
    print("=" * 80)
    result = forward_check_sum_eq_constant()
    print(f"Pairs satisfying x+y=10 in [0,10]^2: {result['pairs_before']}")
    print(f"After binding x=3, y's domain shrinks to: {result['narrowed_y_after_x=3']}")
    print(f"Per-x domain sizes of y: {result['per_x_cardinality']}")


if __name__ == "__main__":
    main()
