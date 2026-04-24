"""Act-3 D&D combat demo.

Three composable NPC ``@program``s + ``@enter_combat`` /
``@exit_combat`` mode-transition tools demonstrate the session's
mode-aware registry: in narrative mode only the entry tool is
visible; once the model invokes ``@enter_combat``, the grammar
swaps to expose each character's stat-sheet-derived action program.

Run:
    .venv/bin/python examples/d20/act3_combat_demo.py
"""

from .characters import (
    aria_attack,
    borin_attack,
    enter_combat,
    exit_combat,
    hooded_figure_attack,
)

__all__ = [
    "aria_attack",
    "borin_attack",
    "enter_combat",
    "exit_combat",
    "hooded_figure_attack",
]
