"""Act 3 — Tool calls are just yields. So are agent handoffs. So are structs.

Structured output, tool use, and agent control flow are three different
APIs today. They do not need to be. A yield is a point where the
program pauses and a decision comes back from somewhere — the model,
a tool, a sub-agent, a cache. The source of the decision is the
runner's problem, not the author's.

Below is a single @program that mixes all four: gen.choice, a struct,
a tool call, and a conditional sub-decision based on the tool result.
One KV cache, one engine, one primitive.

In the Anthropic API, this would be three distinct API contracts
(tool-use, JSON mode, agent loop). In orate it's one generator.
"""

from __future__ import annotations

import random

from orate import gen, program
from orate.engine.mock import MockEngine

# ---- tools ---------------------------------------------------------------


def roll_dice(sides: int) -> int:
    """A non-LLM tool: deterministic (seeded) for the demo."""
    random.seed(42)
    return random.randint(1, sides)


def lookup_enemy_weakness(enemy: str) -> str:
    """A non-LLM tool: fake lookup table."""
    table = {"dragon": "ice", "goblin": "fire", "ghost": "silver"}
    return table.get(enemy, "none")


# ---- unified @program ----------------------------------------------------


@program
def combat_turn():
    """One program, four yield types:

    1. gen.choice  — structured-output decision
    2. gen.struct  — compound decision (one grammar, lowered on XGrammar)
    3. gen.tool    — tool call (no separate API)
    4. gen.integer — follow-up decision that *depends on* the tool result
    """
    action = yield gen.choice(["attack", "defend", "cast_spell"])

    if action == "attack":
        target = yield gen.choice(["dragon", "goblin", "ghost"])
        # Tool call is a yield. No "tool-use API" to set up; no
        # round-trip through a separate endpoint.
        weakness = yield gen.tool(lookup_enemy_weakness, enemy=target)
        # A compound struct for the attack params — one logical yield.
        attack = yield gen.struct(
            weapon=gen.choice(["sword", "bow", "staff"]),
            stance=gen.choice(["aggressive", "defensive"]),
        )
        # A tool-sourced value steers the next sampled choice's range.
        # (This is the "bonus damage when exploiting a weakness" rule.)
        bonus = 5 if weakness != "none" else 0
        damage = yield gen.integer(1 + bonus, 10 + bonus)
        return {
            "action": "attack",
            "target": target,
            "weakness_exploited": weakness,
            "attack": attack,
            "damage": damage,
        }

    if action == "cast_spell":
        spell = yield gen.choice(["fireball", "heal", "shield"])
        # Tool-driven dice roll determines spell power.
        power = yield gen.tool(roll_dice, sides=20)
        return {"action": "cast_spell", "spell": spell, "power": power}

    # defend
    stance = yield gen.choice(["brace", "parry", "evade"])
    return {"action": "defend", "stance": stance}


def main() -> None:
    print("Act 3 — structured output, tool calls, and compounds share one primitive.")
    print()
    print("One @program. Four yield types. No separate tool-use API.")
    print()

    for seed in range(5):
        engine = MockEngine(seed=seed)
        result = combat_turn().run(engine=engine)
        print(f"  seed={seed}: {result}")

    print()
    print("Notice: the tool call is just a yield. The struct is just a yield.")
    print("The runner does not care which is which. One primitive, one KV cache.")
    print("Continue: act_04_arc_sketch.py  (the uppercut)")


if __name__ == "__main__":
    main()
