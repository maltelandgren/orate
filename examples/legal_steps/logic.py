"""@inference_step — one legal propositional deduction.

Three yields:
  1. ``premises`` — semicolon-separated proposition list, e.g.
                    ``"P -> Q; P"``
  2. ``rule``     — which inference rule is being applied
  3. ``conclusion`` — the derived proposition, predicate-verified
                    derivable from ``premises`` under ``rule``

Example calls:

    @inference_step("P -> Q; P", "modus_ponens", "Q")
    @inference_step("Q -> R; Q", "modus_ponens", "R")
    @inference_step("P -> Q; Q -> R", "hypothetical_syllogism", "P -> R")
"""

from __future__ import annotations

from orate import gen, program

from .checkers import derivable_under


def _split_premises(s: str) -> list[str]:
    return [p.strip() for p in s.split(";") if p.strip()]


@program
def inference_step():
    """One legal propositional deduction step.

    Note: the choice list is inlined as a literal so the body grammar
    can extract the options. Keep in sync with ``LOGIC_RULES`` in
    ``checkers.py``.
    """
    premises = yield gen.string(
        max_len=120,
        description=(
            "semicolon-separated premises in propositional logic, "
            "e.g. 'P -> Q; P'"
        ),
    )
    rule = yield gen.choice(
        [
            "modus_ponens",
            "modus_tollens",
            "hypothetical_syllogism",
            "conjunction",
            "simplification",
        ],
        description=(
            "the inference rule: modus_ponens / modus_tollens / "
            "hypothetical_syllogism / conjunction / simplification"
        ),
    )
    conclusion = yield gen.string(
        max_len=60,
        description="the derived proposition",
        where=lambda s: derivable_under(rule, _split_premises(premises), s),
    )
    return {
        "premises": premises,
        "rule": rule,
        "conclusion": conclusion,
    }
