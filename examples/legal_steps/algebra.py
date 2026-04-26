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


@program(ends_turn=True)
def done():
    """Signal that the solve is complete and surface the final answer.

    Takes a single integer — the value of the variable. The model
    emits ``@done(3)`` rather than a free-text string. This sidesteps
    the regex-fragile parsing of phrases like ``'x = 3'`` vs ``'3'``;
    the answer is just an int the bench can compare directly.
    """
    x = yield gen.integer(-9999, 9999)
    return {"x": x}


@program
def algebra_step():
    """One legal algebraic transformation: (before, rule, after).

    Note: the choice list is inlined as a literal so the body grammar
    can extract the options. Keep in sync with ``ALGEBRA_RULES`` in
    ``checkers.py``.

    The string slot for equations uses a math-only ``pattern``: digits,
    lowercase letters (variable names), basic operators, spaces,
    equals, and parens for distribution. This blocks the model from
    hallucinating English-language commentary inside the ``after``
    slot — e.g. ``"3 - 2x = 0 (subtract 5 from both sides)"`` — which
    it would otherwise do under the default printable-ASCII char
    class. The pattern must be a string LITERAL (the body-grammar
    deriver scans the AST), so it's inlined verbatim in both yields.
    """
    before = yield gen.string(
        max_len=30,
        pattern="[0-9a-z +\\-*/=()]",
        description="the current equation, e.g. '2x + 3y = 12'",
    )
    rule = yield gen.choice(
        ["simplify", "combine_like", "isolate_var", "evaluate"],
        description=(
            "the algebraic move: simplify / combine_like / isolate_var / evaluate"
        ),
    )
    after = yield gen.string(
        max_len=30,
        pattern="[0-9a-z +\\-*/=()]",
        description=(
            "the resulting equation, mathematically equivalent to "
            "'before' under 'rule'"
        ),
        where=lambda s: equivalent_under(rule, before, s),
    )
    return {"before": before, "rule": rule, "after": after}
