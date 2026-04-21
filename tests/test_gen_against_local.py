"""End-to-end: a @program with multiple yields against the real engine.

Model-gated. Confirms the coroutine-driven runner, ADR-0014
tightening, and the XGrammar engine all compose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("llama_cpp")
pytest.importorskip("xgrammar")
pytest.importorskip("transformers")

from orate import gen  # noqa: E402
from orate.engine.xgrammar import XGrammarEngine  # noqa: E402
from orate.program import program  # noqa: E402

MODEL_PATH = "/Users/maltelandgren/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

needs_model = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="local GGUF not available",
)


@program
def _pick_two():
    """Two yields: one choice, one int. Enough to prove the loop works."""
    color = yield gen.choice(["red", "blue", "green"])
    number = yield gen.integer(1, 3)
    return (color, number)


@needs_model
def test_program_with_two_yields_runs_end_to_end() -> None:
    eng = XGrammarEngine(
        model_path=MODEL_PATH,
        n_ctx=1024,
        max_tokens_per_sample=16,
        seed=0,
    )
    eng.prime("You are helping a test. Answer with a single token.\nAnswer: ")

    color, number = _pick_two().run(engine=eng)

    assert color in {"red", "blue", "green"}
    assert 1 <= number <= 3
