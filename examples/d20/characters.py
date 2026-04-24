"""NPC stat-sheets, encoded as @program bodies.

Each character's available actions, target set, and damage cap come
from its own program — no central switch statement. The combat
grammar is the textual concatenation of these programs' body rules,
exposed only in the session's "combat" mode.

This is the Act-3 picture: the schema for what a character can do
*in this fight* is built from that character's stat sheet, and the
session's outer grammar updates when combat begins / ends.
"""

from __future__ import annotations

from orate import gen, program


@program(mode_transition="combat")
def enter_combat():
    """Begin a combat encounter. Flips the session into 'combat' mode."""
    initiator = yield gen.choice(
        ["aria", "hooded_figure", "borin"],
        description="who initiates the encounter",
    )
    return {"initiator": initiator}


@program(mode_transition="default")
def exit_combat():
    """End combat. Returns the session to the default (narrative) mode."""
    outcome = yield gen.choice(
        ["victory", "flee", "stalemate"],
        description="how the fight resolved",
    )
    return {"outcome": outcome}


@program
def aria_attack():
    """Aria the bard — high mobility, short blade, supportive options."""
    action = yield gen.choice(
        ["longsword", "vicious_mockery", "hold"],
        description="aria's action this round",
    )
    target = yield gen.choice(
        ["hooded_figure", "borin", "self"],
        description="target of aria's action",
    )
    damage = yield gen.integer(0, 6, description="damage dealt (capped at 6)")
    return {
        "actor": "aria",
        "action": action,
        "target": target,
        "damage": damage,
    }


@program
def hooded_figure_attack():
    """The hooded figure — sneaky rogue with daggers and a shadow step."""
    action = yield gen.choice(
        ["dagger", "shadow_step", "retreat"],
        description="hooded_figure's action this round",
    )
    target = yield gen.choice(
        ["aria", "borin", "self"],
        description="target of hooded_figure's action",
    )
    damage = yield gen.integer(0, 4, description="damage dealt (capped at 4)")
    return {
        "actor": "hooded_figure",
        "action": action,
        "target": target,
        "damage": damage,
    }


@program
def borin_attack():
    """Borin the dwarven cleric — heavy armor, warhammer, intimidating."""
    action = yield gen.choice(
        ["warhammer", "shield_bash", "intimidate"],
        description="borin's action this round",
    )
    target = yield gen.choice(
        ["hooded_figure", "aria", "self"],
        description="target of borin's action",
    )
    damage = yield gen.integer(0, 8, description="damage dealt (capped at 8)")
    return {
        "actor": "borin",
        "action": action,
        "target": target,
        "damage": damage,
    }
