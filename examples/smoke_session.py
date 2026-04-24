"""Act-4 final form: the Session loop, end-to-end on Qwen2.5-7B.

Persistent KV. Accumulating tool registry. Every @call the model emits
is either a new program defined via @make_new_program (which triggers
a source-synthesis sub-sample, validates, compiles, registers,
rebuilds the outer grammar — all on the same KV) or an invocation of
a program defined earlier in the session.

    .venv/bin/python examples/smoke_session.py
"""

from __future__ import annotations

from pathlib import Path

from orate import FreeText, NewProgramRegistered, ProgramInvoked, Session, TurnEnded
from orate.engine.xgrammar import XGrammarEngine


def _pick_model() -> str:
    for candidate in [
        "/Users/maltelandgren/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "/Users/maltelandgren/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No local Qwen2.5 GGUF found.")


SYSTEM = """\
You are a tool-using assistant.

You live inside a single continuous session. Your output is constrained
by a grammar: at each step you emit either plain text OR a tool call
of the form `@toolname(args)`. When you emit `@make_new_program("name",
"description")`, the runtime will prompt you to author a new @program
whose body it will compile into a tool; once compiled, the tool becomes
callable by name in any later step of this same session. All previously
registered tools remain available.

Keep your prose short. Use tools when they help you structure output.
"""


def _render(event) -> None:
    if isinstance(event, FreeText):
        print(f"[text]  {event.text!r}")
    elif isinstance(event, NewProgramRegistered):
        print(f"[+tool] {event.name}\n--source--\n{event.source}\n----------")
    elif isinstance(event, ProgramInvoked):
        print(f"[call]  @{event.name}({event.args}) → {event.result}")
    elif isinstance(event, TurnEnded):
        print(f"[turn end: {event.reason}]")


def main() -> None:
    model = _pick_model()
    print(f"Loading {Path(model).name}...")
    engine = XGrammarEngine(model_path=model, max_tokens_per_sample=512, n_ctx=16384)

    session = Session(engine=engine, system=SYSTEM, max_turn_tokens=1024)

    print()
    print("=" * 72)
    print("TURN 1 — user asks for a plan. Expect the model to define tools.")
    print("=" * 72)
    session.user(
        "Help me plan a medieval-knight-themed birthday party. Design a tool "
        "or two if it helps you produce structured output."
    )
    for event in session.advance():
        _render(event)

    print()
    print("=" * 72)
    print("TURN 2 — user asks the model to use the tool(s) it just defined.")
    print("=" * 72)
    session.user("Now use the tool(s) you defined to pick final details.")
    for event in session.advance():
        _render(event)

    print()
    print("-" * 72)
    print(f"Registry at session end: {sorted(session.registry.keys())}")


if __name__ == "__main__":
    main()
