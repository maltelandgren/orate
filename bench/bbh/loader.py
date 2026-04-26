"""BBH logical-deduction dataset loader + grader.

Hugging Face hosts the ``lukaemon/bbh`` mirror of Suzgun et al.'s
BIG-Bench Hard. Each problem is two strings:

    {"input": "<paragraph + Options + (A)..(N)>", "target": "(X)"}

We expose three subtasks (``three_objects`` / ``five_objects`` /
``seven_objects``) and a small grading helper that pulls the last
``(X)`` letter out of a free-form model response. That matches what
the upstream BBH grader does — the official scorer just regex-finds
the last parenthesised letter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

# The three subtasks we run against. Counts are listed in the BBH paper
# and confirmed by ``len(load_dataset(...))``.
SUBTASKS: tuple[str, ...] = (
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
)

SUBTASK_OBJECT_COUNT: dict[str, int] = {
    "logical_deduction_three_objects": 3,
    "logical_deduction_five_objects": 5,
    "logical_deduction_seven_objects": 7,
}


@dataclass(frozen=True)
class BBHProblem:
    """One BBH logical-deduction problem.

    ``index`` is the row position in the upstream dataset — used as
    the stable identifier when matching baseline against orate runs.
    """

    subtask: str
    index: int
    question: str
    target: str  # always "(A)" .. "(G)"


def load_subtask(subtask: str, *, limit: int | None = None) -> list[BBHProblem]:
    """Load one BBH subtask via Hugging Face datasets.

    ``limit`` truncates to the first ``limit`` rows; useful for smoke
    tests and quick iteration. Default returns all 250 problems.
    """
    if subtask not in SUBTASKS:
        raise ValueError(f"unknown subtask {subtask!r}; pick one of {SUBTASKS}")
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("lukaemon/bbh", subtask, split="test")
    rows: list[BBHProblem] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append(
            BBHProblem(
                subtask=subtask,
                index=i,
                question=row["input"],
                target=row["target"].strip(),
            )
        )
    return rows


def iter_subtasks(*, limit: int | None = None) -> Iterator[BBHProblem]:
    """Yield problems across all three subtasks in canonical order."""
    for s in SUBTASKS:
        yield from load_subtask(s, limit=limit)


# ---- grading -----------------------------------------------------------

# Match a single bracketed letter A-G at the end of the response, being
# permissive about trailing punctuation/whitespace. The upstream BBH
# grader uses something equivalent: pull the last "(X)" the model emits.
_LETTER_RE = re.compile(r"\(([A-Ga-g])\)")


def extract_answer_letter(response: str) -> str | None:
    """Return the last "(X)" token in ``response`` as ``"(X)"``, or None.

    BBH targets are always uppercase like ``"(A)"``, so we normalise.
    Returns ``None`` if no parenthesised letter appears anywhere — the
    standard BBH grader treats that as a wrong answer.
    """
    matches = _LETTER_RE.findall(response)
    if not matches:
        return None
    return f"({matches[-1].upper()})"


def is_correct(response: str, target: str) -> bool:
    """True iff the last ``(X)`` in ``response`` matches ``target``.

    ``target`` arrives already in the ``"(X)"`` shape from the dataset.
    """
    pred = extract_answer_letter(response)
    return pred is not None and pred == target.upper()
