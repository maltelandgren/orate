"""Act-3 demo: narrative → combat → narrative on a single KV.

Setup:
- Narrative mode (default) exposes ``@enter_combat`` only.
- Combat mode exposes the three composed NPC action programs plus
  ``@exit_combat``.
- ``@enter_combat`` is decorated with ``mode_transition="combat"``;
  the runtime swaps the outer grammar after the call.

The model is asked to start a fight, play out one round per character,
then exit combat.

Run:
    .venv/bin/python examples/d20/act3_combat_demo.py
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
You output ONLY @-calls. No prose. No markdown. No commentary.

Available tools (narrative mode):
  @enter_combat(initiator)   — begin combat. initiator is one of:
                               aria, hooded_figure, borin.

Available tools (combat mode — visible only after enter_combat):
  @aria_attack(action, target, damage)
  @hooded_figure_attack(action, target, damage)
  @borin_attack(action, target, damage)
  @exit_combat(outcome)

Each character has its own action set + damage cap (the runtime
enforces these). action and target are bare identifiers; damage is
a bare integer. Strings (when needed) are JSON-quoted.

Worked example:
@enter_combat(aria)
@hooded_figure_attack(dagger, aria, 3)
@aria_attack(longsword, hooded_figure, 4)
@borin_attack(warhammer, hooded_figure, 6)
@exit_combat(victory)
"""


PROBLEM = """\
A hooded figure draws a blade in the tavern. Aria, Borin, and the
hooded figure square off. Initiate combat, play one round (each
character acts once), then end combat.
"""


def _render(event) -> None:
    if isinstance(event, FreeText):
        text = event.text.strip()
        if text:
            print(f"[text]   {text!r}")
    elif isinstance(event, NewProgramRegistered):
        print(f"[+tool]  {event.name}")
    elif isinstance(event, ProgramInvoked):
        if event.result.get("rejected"):
            print(f"[REJ]    @{event.name}{event.args}")
            print(f"         → {event.result['error']}")
        else:
            print(f"[ok]     @{event.name}{event.args}")
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
        programs={"enter_combat": enter_combat},
        system=SYSTEM,
        max_turn_tokens=4096,
        max_calls_per_turn=12,
        allow_free_text=False,
    )
    # Combat-mode programs (only visible inside combat).
    session.register("aria_attack", aria_attack, mode="combat")
    session.register("hooded_figure_attack", hooded_figure_attack, mode="combat")
    session.register("borin_attack", borin_attack, mode="combat")
    session.register("exit_combat", exit_combat, mode="combat")

    print()
    print("=" * 72)
    print("ACT 3 — narrative → combat → narrative on one KV.")
    print("=" * 72)
    print(PROBLEM)
    print("-" * 72)

    session.user(PROBLEM)
    for event in session.advance():
        _render(event)
        if isinstance(event, ProgramInvoked) and event.name in {"enter_combat", "exit_combat"}:
            print(f"           [mode: {session.active_mode}]")

    print()
    print("-" * 72)
    print(f"Final mode: {session.active_mode}")
    print(f"Registry at session end: {sorted(session.registry.keys())}")


if __name__ == "__main__":
    main()
