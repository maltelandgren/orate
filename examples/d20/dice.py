"""Dice tools — client-resolved skill checks.

The model emits ``@roll(skill, dc)`` choosing the skill and the DC; the
runtime predicate-checks both against the choice list / DC range, then
the program body computes the d20 server-side and returns the resolved
result. The Session driver appends that result to the KV, so on the
next sample the model sees::

    @roll(perception, 15) → {"skill": "perception", "dc": 15, "d20": 17, "success": true}

…and can continue narrating from there. Tool calls and structured
output are the same primitive — one yield stream, one KV.
"""

from __future__ import annotations

import random

from orate import gen, program


@program
def roll():
    """One skill check. Model picks the skill + DC; runtime rolls the d20.

    The client-side roll happens *after* predicate verification: the
    yields gate the (skill, dc) emission to legal values, then plain
    Python computes the result and returns it. The runtime treats the
    return value as the resolved tool result and feeds it back into
    the KV for the next sample to read.

    Note: gen.choice options are inlined as a list literal — the body
    grammar derivation reads the source AST and needs the choice list
    to be a literal, not a name reference, so it can extract the
    accept set into the call-site grammar.
    """
    skill = yield gen.choice(
        ["perception", "stealth", "athletics", "persuasion", "insight", "investigation"],
        description="which skill the character is using",
    )
    dc = yield gen.integer(
        5,
        25,
        description="difficulty class — 5 trivial, 15 moderate, 25 nearly impossible",
    )
    # Server-resolved. Anything that isn't a yield runs as ordinary
    # Python after predicate verification.
    d20 = random.randint(1, 20)
    return {
        "skill": skill,
        "dc": dc,
        "d20": d20,
        "success": d20 >= dc,
    }
