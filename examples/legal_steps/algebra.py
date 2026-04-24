"""@algebra_step — one legal algebraic transformation.

Three yields:
  1. ``before`` — the equation we're transforming
  2. ``rule``   — which algebraic move is being applied
  3. ``after``  — the result, predicate-verified equivalent to before

The ``where=`` lambda on yield 3 closes over both ``rule`` and
``before`` (already bound by the time the third Gen spec is built),
so the Session's predicate-verification path can re-check
:func:`equivalent_under` on whatever the model emits.

Example call (model emits this; Session decodes and verifies):

    @algebra_step("x + y = 5", "isolate_var", "x = 5 - y")
    @algebra_step("2(5-y) + 3y = 12", "simplify", "10 + y = 12")
    @algebra_step("10 + y = 12", "isolate_var", "y = 2")
    @algebra_step("x = 5 - 2", "evaluate", "x = 3")

If the model emits a (rule, after) pair where ``after`` is not
algebraically equivalent to ``before`` under ``rule``, the call is
rejected — a session-level note is appended to the KV and the next
sample is taken under the same outer grammar.
"""

from __future__ import annotations

from orate import gen, program

from .checkers import equivalent_under


@program
def algebra_step():
    """One legal algebraic transformation: (before, rule, after).

    Note: the choice list is inlined as a literal so the body grammar
    can extract the options. Keep in sync with ``ALGEBRA_RULES`` in
    ``checkers.py``.
    """
    before = yield gen.string(
        max_len=80,
        description="the current equation, e.g. '2x + 3y = 12'",
    )
    rule = yield gen.choice(
        ["substitute", "simplify", "combine_like", "isolate_var", "evaluate"],
        description=(
            "the algebraic move: substitute / simplify / combine_like / "
            "isolate_var / evaluate"
        ),
    )
    after = yield gen.string(
        max_len=80,
        description=(
            "the resulting equation, mathematically equivalent to "
            "'before' under 'rule'"
        ),
        where=lambda s: equivalent_under(rule, before, s),
    )
    return {"before": before, "rule": rule, "after": after}
