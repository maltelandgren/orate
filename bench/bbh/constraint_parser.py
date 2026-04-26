"""Deterministic English → constraint extraction for BBH logical_deduction.

This is the *premise extraction* layer. It scans a BBH problem text and
produces a list of :class:`bench.bbh.ordering.Fact`. Used by the
"Path A" runner where the model only authors the @answer call — the
premises are hand-extracted, not model-extracted.

Templates we cover (sourced from inspecting BBH samples):

  - "X is to the {left,right} of Y"
  - "X is the leftmost / rightmost"
  - "X is the {oldest, newest, cheapest, most expensive}"
  - "X is the {second,third,...} from the {left,right}"
  - "X is the {second,third,...}-{cheapest, oldest, newest, most-expensive}"
  - "X finished {first, last, third, second-to-last, third-to-last}"
  - "X finished {above, below} Y"  → above ⇒ left_of, below ⇒ right_of
  - "X is {newer, older, cheaper, more expensive} than Y"
       → newer ⇒ right_of (newest=rightmost), older ⇒ left_of, etc.
  - "X is the {oldest, newest, cheapest, most expensive}"

Anything we miss is dropped silently. The Path A runner relies on the
extracted set being COMPLETE enough to make the answer derivable; we
verify that with the deterministic solver before invoking the model.
"""
from __future__ import annotations

import re

from .ordering import Fact, _norm

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7,
}

# We segment the body of a problem (the chunk after the items list and
# before "Options:") into sentences, then try each constraint pattern
# against each sentence.


def _segment_body(question: str) -> str:
    """Strip the BBH boilerplate, return just the constraints chunk."""
    s = question
    if "Options:" in s:
        s = s.split("Options:", 1)[0]
    # Drop the intro line ("The following paragraphs each describe ...")
    s = re.sub(r"^[^.]*paragraph\.\s*", "", s, flags=re.IGNORECASE)
    # Drop the items-list line — keep only the part *after* the colon's
    # period, which begins the constraints. We approximate by removing
    # everything up to the first ".".
    s = re.sub(r"^.*?(?:there\s+(?:are|were)|sells|has|hold[s]?|displays|stocks|carries|offer[s]?)\s+\w+\s+\w+:\s*[^.]+\.\s*", "", s, count=1, flags=re.IGNORECASE)
    return s.strip()


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"\.\s*", text) if s.strip()]


# --- pattern handlers --------------------------------------------------


def _try_left_right(sentence: str, n_items: int) -> Fact | None:
    """X is to the {left,right} of Y."""
    m = re.match(
        r"^(.+?)\s+is\s+(?:to\s+the\s+)?(left|right)\s+of\s+(.+)$",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    a, side, b = _norm(m.group(1)), m.group(2).lower(), _norm(m.group(3))
    if side == "left":
        return Fact("left_of", (a, b))
    return Fact("right_of", (a, b))


def _try_immediately_left_right(sentence: str, n_items: int) -> Fact | None:
    """X is immediately to the {left,right} of Y."""
    m = re.match(
        r"^(.+?)\s+is\s+immediately\s+(?:to\s+the\s+)?(left|right)\s+of\s+(.+)$",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    a, side, b = _norm(m.group(1)), m.group(2).lower(), _norm(m.group(3))
    pred = "immediately_left_of" if side == "left" else "immediately_right_of"
    return Fact(pred, (a, b))


def _try_extreme(sentence: str, n_items: int) -> Fact | None:
    """X is the {leftmost, rightmost, oldest, newest, cheapest, most expensive}.

    Unifies the various "extreme position" phrasings into position(item, 1)
    or position(item, n_items).
    """
    # plural-aware "is/are"
    m = re.match(
        r"^(.+?)\s+(?:is|are)\s+(?:the\s+)?(leftmost|rightmost|oldest|newest|cheapest|most\s+expensive)$",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    item = _norm(m.group(1))
    desc = re.sub(r"\s+", " ", m.group(2).lower())
    # Mapping: leftmost ↔ oldest ↔ cheapest = position 1.
    if desc in ("leftmost", "oldest", "cheapest"):
        return Fact("position", (item, 1))
    if desc in ("rightmost", "newest", "most expensive"):
        return Fact("position", (item, n_items))
    return None


def _try_ordinal_from_side(sentence: str, n_items: int) -> Fact | None:
    """X is the {second,third,...} from the {left,right}."""
    m = re.match(
        r"^(.+?)\s+(?:is|are)\s+(?:the\s+)?(\w+)\s+from\s+the\s+(left|right)$",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    item = _norm(m.group(1))
    word = m.group(2).lower()
    side = m.group(3).lower()
    k = _ORDINAL_WORDS.get(word)
    if k is None:
        return None
    if side == "right":
        k = n_items + 1 - k
    return Fact("position", (item, k))


def _try_compound_rank(sentence: str, n_items: int) -> Fact | None:
    """X is the {second,third,...}-{cheapest, oldest, newest, most expensive}.

    "the second-cheapest" → position 2 (cheapest=position 1, ascending).
    "the second-most expensive" → position N-1 (most expensive=position N).
    """
    m = re.match(
        r"^(.+?)\s+(?:is|are)\s+(?:the\s+)?(\w+)[\s-]+(most\s+expensive|cheapest|oldest|newest|youngest)$",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    item = _norm(m.group(1))
    rank_word = m.group(2).lower()
    scale = re.sub(r"\s+", " ", m.group(3).lower())
    k = _ORDINAL_WORDS.get(rank_word)
    if k is None:
        return None
    if scale in ("cheapest", "oldest"):
        return Fact("position", (item, k))
    if scale in ("newest", "youngest", "most expensive"):
        return Fact("position", (item, n_items + 1 - k))
    return None


def _try_finished_position(sentence: str, n_items: int) -> Fact | None:
    """X finished {first, third, last}; X finished second-to-last."""
    # "X finished last/first"
    m = re.match(r"^(.+?)\s+finished\s+(first|last)$", sentence, re.IGNORECASE)
    if m:
        item = _norm(m.group(1))
        if m.group(2).lower() == "first":
            return Fact("position", (item, 1))
        return Fact("position", (item, n_items))
    # "X finished second-to-last" / "third-to-last"
    m = re.match(r"^(.+?)\s+finished\s+(\w+)[\s-]+to[\s-]+last$", sentence, re.IGNORECASE)
    if m:
        item = _norm(m.group(1))
        word = m.group(2).lower()
        k = _ORDINAL_WORDS.get(word)
        if k is None:
            return None
        return Fact("position", (item, n_items + 1 - k))
    # "X finished {first/second/third/...}"
    m = re.match(r"^(.+?)\s+finished\s+(\w+)$", sentence, re.IGNORECASE)
    if m:
        item = _norm(m.group(1))
        word = m.group(2).lower()
        k = _ORDINAL_WORDS.get(word)
        if k is not None:
            return Fact("position", (item, k))
    return None


def _try_finished_above_below(sentence: str, n_items: int) -> Fact | None:
    """X finished {above, below} Y. above==better==left_of."""
    m = re.match(r"^(.+?)\s+finished\s+(above|below)\s+(.+)$", sentence, re.IGNORECASE)
    if not m:
        return None
    a, side, b = _norm(m.group(1)), m.group(2).lower(), _norm(m.group(3))
    if side == "above":
        return Fact("left_of", (a, b))
    return Fact("right_of", (a, b))


def _try_comparative(sentence: str, n_items: int) -> Fact | None:
    """X is {newer, older, cheaper, more expensive} than Y.

    Axis polarity:
      - 'older'  → left_of(x, y)  (older=leftmost)
      - 'newer'  → right_of(x, y) (newer=rightmost)
      - 'cheaper'→ left_of(x, y)  (cheaper=leftmost)
      - 'more expensive'→ right_of(x, y)
    """
    m = re.match(
        r"^(.+?)\s+is\s+(older|newer|cheaper|more\s+expensive)\s+than\s+(.+)$",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    a, comp, b = _norm(m.group(1)), re.sub(r"\s+", " ", m.group(2).lower()), _norm(m.group(3))
    if comp in ("older", "cheaper"):
        return Fact("left_of", (a, b))
    if comp in ("newer", "more expensive"):
        return Fact("right_of", (a, b))
    return None


def _try_less_more_expensive(sentence: str, n_items: int) -> Fact | None:
    """X is {less, more} expensive than Y."""
    m = re.match(
        r"^(.+?)\s+(?:is|are)\s+(less|more)\s+expensive\s+than\s+(.+)$",
        sentence,
        re.IGNORECASE,
    )
    if not m:
        return None
    a, comp, b = _norm(m.group(1)), m.group(2).lower(), _norm(m.group(3))
    return Fact("left_of" if comp == "less" else "right_of", (a, b))


_PATTERNS: tuple = (
    _try_immediately_left_right,
    _try_left_right,
    _try_compound_rank,
    _try_ordinal_from_side,
    _try_extreme,
    _try_finished_above_below,
    _try_finished_position,
    _try_less_more_expensive,
    _try_comparative,
)


def extract_premises(question: str, n_items: int) -> list[Fact]:
    """Run every constraint-matcher against every sentence in ``question``."""
    body = _segment_body(question)
    facts: list[Fact] = []
    for sent in _split_sentences(body):
        for handler in _PATTERNS:
            fact = handler(sent, n_items)
            if fact is not None:
                facts.append(fact)
                break
    return facts
