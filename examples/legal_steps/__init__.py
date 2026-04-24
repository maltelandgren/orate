"""Act-4 legal-step demos.

Two ``@program`` primitives whose ``where=`` predicates make
"the model can only emit legal steps" load-bearing:

- :mod:`legal_steps.algebra` — :func:`algebra_step` enforces algebraic
  equivalence under one of {substitute, simplify, combine_like,
  isolate_var, evaluate}. Powered by SymPy.

- :mod:`legal_steps.logic` — :func:`inference_step` enforces
  derivability under one of {modus_ponens, modus_tollens,
  hypothetical_syllogism, conjunction, simplification}.

Run the demos:

    .venv/bin/python examples/legal_steps/algebra_demo.py
    .venv/bin/python examples/legal_steps/logic_demo.py
"""

from .algebra import algebra_step
from .checkers import (
    ALGEBRA_RULES,
    LOGIC_RULES,
    derivable_under,
    equivalent_under,
)
from .logic import inference_step

__all__ = [
    "ALGEBRA_RULES",
    "LOGIC_RULES",
    "algebra_step",
    "derivable_under",
    "equivalent_under",
    "inference_step",
]
