"""Deterministic English → (items, options) extraction for BBH problems.

Every BBH logical_deduction problem is built from a fixed template:

    "The following paragraphs each describe a set of N objects ... In a
    <scene>, there are N <kind>: <item1>, <item2>, ..., and <itemN>.
    <constraint>. <constraint>. ...
    Options:
    (A) <option1>
    (B) <option2>
    ..."

We extract the N items and the option list with a couple of regexes —
no LLM in this part. The premises themselves are *not* extracted here;
those go into the model's @-call chain (extracting them deterministically
would defeat the purpose of measuring whether orate's predicate-bound
authoring helps).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Match the items list. BBH scaffolds it three ways:
#   1. "On a branch, there are three birds: a blue jay, ..."
#   2. "In a golf tournament, there were three golfers: Amy, Eli, Eve."
#   3. "A fruit stand sells five fruits: apples, peaches, ..."
# We unify with a permissive verb alternation. Items themselves are
# comma-separated with an "and" before the last; articles may prefix.
_ITEMS_RE = re.compile(
    r"(?:there\s+(?:are|were)|sells|has|hold[s]?|displays|stocks|carries|offer[s]?)"
    r"\s+\w+\s+[a-z]+:\s*(.+?)\s*\.",
    re.IGNORECASE,
)

_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

_OPTIONS_RE = re.compile(r"\(([A-Ga-g])\)\s*(.+?)(?=\n|$)")


@dataclass(frozen=True)
class Extracted:
    """Result of parsing a BBH problem into (items, options) shape."""

    items: list[str]
    options: dict[str, str]  # "(A)" → option text


def _strip_article(s: str) -> str:
    return _ARTICLE_RE.sub("", s).strip()


def extract_problem(question: str) -> Extracted:
    """Pull (items, options) from a BBH ``input`` string.

    Returns an :class:`Extracted` even if extraction is partial — empty
    items / options are valid (the orate runner will simply not be able
    to validate facts, and the answer predicate falls through to the
    accept-anything branch).
    """
    items: list[str] = []
    m = _ITEMS_RE.search(question)
    if m:
        raw = m.group(1)
        # Split on commas; handle the trailing "and " before the last item.
        parts = re.split(r",\s*", raw)
        cleaned: list[str] = []
        for part in parts:
            # Strip leading "and " on the final part.
            part = re.sub(r"^and\s+", "", part, flags=re.IGNORECASE).strip()
            if not part:
                continue
            cleaned.append(_strip_article(part))
        items = [c for c in cleaned if c]

    # Options block: lines like "(A) The X is the leftmost"
    options: dict[str, str] = {}
    # Cut off the "Options:" portion so we don't accidentally pick up
    # a parenthesised letter from the body text.
    opt_part = question
    if "Options:" in question:
        opt_part = question.split("Options:", 1)[1]
    for letter, text in _OPTIONS_RE.findall(opt_part):
        options[f"({letter.upper()})"] = text.strip()

    return Extracted(items=items, options=options)
