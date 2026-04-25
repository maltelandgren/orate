"""Act-3 demo: @narrate + @roll + @meta + @enter_combat + 3 NPC turns + @exit_combat.

The narrative-mode session exposes four tools: in-character
``@narrate``, client-resolved ``@roll``, out-of-character ``@meta``
(same body shape as @narrate but a different name + purpose), and
``@enter_combat`` for the mode transition. Combat mode swaps to the
three composed NPC programs + ``@exit_combat``. One KV from start
to finish; two mode switches; no API hops.

The @narrate / @meta pair shows that *multiple same-shape outputs
are welcome*: the body grammar is identical (one string yield), but
the tool name carries the purpose. No need to wrap outputs in XML
tags and parse post-generation; the call-prefix is the signal.

Each character's program is its own stat sheet — the action set,
target list, and damage cap come from the program body, not a
central switch.

Run:
    .venv/bin/python examples/d20/act3_full_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make examples/ importable as a package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from d20.characters import (  # noqa: E402
    aria_attack,
    borin_attack,
    enter_combat,
    exit_combat,
    hooded_figure_attack,
)
from d20.dice import meta, narrate, roll  # noqa: E402
from orate import (  # noqa: E402
    FreeText,
    NewProgramRegistered,
    ProgramInvoked,
    Session,
    TurnEnded,
)
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402


def _pick_model() -> str:
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


SYSTEM = """\
You output ONLY @-calls. No prose, no markdown.

Tools available in NARRATIVE mode:
  @narrate("...")               — one short sentence of in-character
                                  DM narration describing the scene.
  @roll(skill, dc)              — d20 skill check; the runtime rolls
                                  and returns the result. skill ∈
                                  {perception, stealth, athletics,
                                  persuasion, insight, investigation}.
                                  dc ∈ [5, 25].
  @meta("...")                  — one short OUT-of-character comment
                                  (DM aside, table reaction). Same
                                  body shape as @narrate, different
                                  purpose; pick the right tool.
  @enter_combat(initiator)     — begin combat. initiator ∈ {aria,
                                  hooded_figure, borin}.

Tools available in COMBAT mode (visible only after @enter_combat):
  @aria_attack(action, target, damage)
  @hooded_figure_attack(action, target, damage)
  @borin_attack(action, target, damage)
  @exit_combat(outcome)

Worked example:

@narrate("The tavern is dim and smells of woodsmoke.")
@roll(perception, 13)
@meta("That low roll won't catch much.")
@narrate("Aria glances around but misses the glint of metal in the booth.")
@enter_combat(hooded_figure)
@hooded_figure_attack(dagger, aria, 3)
@aria_attack(longsword, hooded_figure, 5)
@borin_attack(warhammer, hooded_figure, 6)
@exit_combat(victory)
"""


PROBLEM = """\
Setup: Aria the bard and Borin the dwarven cleric enter the Wandering
Goose tavern. A hooded figure sits in the back booth with a hand on a
hilt.

Run this scene as @-calls:
1. @narrate(...) — describe the tavern as the party enters.
2. @roll(perception, 13) — Aria sizes up the hooded figure.
3. @meta(...) — one short comment reacting to the roll result.
4. @narrate(...) — narrate what Aria notices given the roll.
5. @enter_combat(hooded_figure) — the figure draws.
6. @hooded_figure_attack, @aria_attack, @borin_attack — one round each.
7. @exit_combat(victory).
"""


def _render(event, mode: str) -> None:
    if isinstance(event, FreeText):
        text = event.text.strip()
        if text:
            print(f"[narrate]   {text}")
    elif isinstance(event, NewProgramRegistered):
        print(f"[+tool]     {event.name}")
    elif isinstance(event, ProgramInvoked):
        if event.result.get("rejected"):
            print(f"[REJ]       @{event.name}{event.args}")
            print(f"            → {event.result['error']}")
        else:
            print(f"[{mode}]    @{event.name}{event.args}")
            print(f"            → {event.result}")
    elif isinstance(event, TurnEnded):
        print(f"[turn end: {event.reason}]")


def main() -> None:
    model = _pick_model()
    print(f"=== Loading {Path(model).name} ===")
    engine = XGrammarEngine(
        model_path=model,
        max_tokens_per_sample=512,
        n_ctx=16384,
    )

    session = Session(
        engine=engine,
        programs={
            "narrate": narrate,
            "roll": roll,
            "meta": meta,
            "enter_combat": enter_combat,
        },
        system=SYSTEM,
        max_turn_tokens=4096,
        max_calls_per_turn=12,
        allow_free_text=False,
    )
    # Combat-mode programs (visible only inside combat).
    session.register("aria_attack", aria_attack, mode="combat")
    session.register("hooded_figure_attack", hooded_figure_attack, mode="combat")
    session.register("borin_attack", borin_attack, mode="combat")
    session.register("exit_combat", exit_combat, mode="combat")

    print()
    print("=" * 72)
    print("ACT 3 — narrative tools → combat (mode switch) → narrative.")
    print("=" * 72)
    print(PROBLEM)
    print("-" * 72)

    session.user(PROBLEM)
    for event in session.advance():
        _render(event, session.active_mode[:5])

    print()
    print("-" * 72)
    print(f"Final mode: {session.active_mode}")


if __name__ == "__main__":
    main()
