"""Ordering-puzzle solver + ``derivable`` predicate for BBH logical_deduction.

This is the domain-specific analogue of ``examples/legal_steps/checkers.py``'s
``derivable_under`` — instead of propositional inference rules, the
predicate here decides whether a candidate ordering fact is forced by
a set of prior constraints.

We model the puzzle as: ``N`` items occupy positions ``1..N`` along a
single axis (left/right, top/bottom, oldest/newest, cheapest/most-
expensive — all variants reduce to the same partial-order shape). A
*constraint* binds positions of items via primitive predicates:

    - ``left_of(a, b)``    — pos(a) < pos(b)
    - ``right_of(a, b)``   — pos(a) > pos(b)
    - ``leftmost(a)``      — pos(a) == 1
    - ``rightmost(a)``     — pos(a) == N
    - ``position(a, k)``   — pos(a) == k        (1-based, ``second_from_left`` etc.)
    - ``between(a, b, c)`` — pos(b) < pos(a) < pos(c) OR pos(c) < pos(a) < pos(b)
    - ``immediately_left_of(a, b)``  — pos(a) + 1 == pos(b)
    - ``immediately_right_of(a, b)`` — pos(a) == pos(b) + 1

Solver: enumerate permutations of items to positions; collect the
permutations consistent with all asserted constraints. A claim is
*derivable* iff every consistent permutation also satisfies the claim.
This is the same shape ``derivable_under`` uses (premises ⊨ conclusion)
but specialised to ordering.

Brute-force enumeration is fine: BBH caps at 7 items so the search
space is 7! = 5040 permutations — trivially fast.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass


# ---- fact AST ----------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """A primitive ordering fact about ``items`` (the universe).

    We use a sum-type via ``predicate`` plus a tuple of arguments.
    ``predicate`` is one of the names listed in :data:`PREDICATES`.
    ``args`` are item names (strings) except for ``position`` where the
    second arg is an int (1-based position).
    """

    predicate: str
    args: tuple

    def render(self) -> str:
        """Compact text form, e.g. ``"left_of(a, b)"`` — for traces."""
        return f"{self.predicate}({', '.join(str(a) for a in self.args)})"


PREDICATES: tuple[str, ...] = (
    "left_of",
    "right_of",
    "leftmost",
    "rightmost",
    "position",
    "between",
    "immediately_left_of",
    "immediately_right_of",
)


_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def _norm(name: str) -> str:
    """Canonical form for an item name. Lowercased, articles stripped, whitespace-collapsed.

    Articles ("a", "an", "the") are common in BBH item lists and easy
    for the model to slip in. Stripping them aligns "a falcon" with
    "falcon" so the predicate doesn't reject otherwise-valid facts on
    a cosmetic difference.
    """
    s = re.sub(r"\s+", " ", name).strip().lower()
    return _LEADING_ARTICLE_RE.sub("", s).strip()


def parse_fact(text: str) -> Fact | None:
    """Parse ``"predicate(arg1, arg2[, arg3])"`` into a :class:`Fact`.

    Item names are lowercased and whitespace-collapsed. Returns ``None``
    on shape mismatch — predicate semantics demand a clean failure.
    """
    s = text.strip()
    m = re.fullmatch(r"([a-z_]+)\s*\((.*)\)", s)
    if m is None:
        return None
    pred = m.group(1)
    if pred not in PREDICATES:
        return None
    raw_args = [a.strip() for a in m.group(2).split(",")]
    if pred == "position":
        if len(raw_args) != 2:
            return None
        try:
            k = int(raw_args[1])
        except ValueError:
            return None
        return Fact("position", (_norm(raw_args[0]), k))
    if pred == "between":
        if len(raw_args) != 3:
            return None
        return Fact("between", tuple(_norm(a) for a in raw_args))
    expected_arity = 1 if pred in ("leftmost", "rightmost") else 2
    if len(raw_args) != expected_arity:
        return None
    return Fact(pred, tuple(_norm(a) for a in raw_args))


# ---- evaluation against a permutation ----------------------------------


def _fact_holds(fact: Fact, positions: dict[str, int], n: int) -> bool:
    """Return True iff ``fact`` is true under ``positions`` (item → 1..n)."""
    p = fact.predicate
    a = fact.args
    if p == "left_of":
        return positions.get(a[0], -1) < positions.get(a[1], -1)
    if p == "right_of":
        return positions.get(a[0], -1) > positions.get(a[1], -1)
    if p == "leftmost":
        return positions.get(a[0], -1) == 1
    if p == "rightmost":
        return positions.get(a[0], -1) == n
    if p == "position":
        return positions.get(a[0], -1) == a[1]
    if p == "between":
        ax, bx, cx = (positions.get(x, -1) for x in a)
        return (bx < ax < cx) or (cx < ax < bx)
    if p == "immediately_left_of":
        return positions.get(a[0], -1) + 1 == positions.get(a[1], -1)
    if p == "immediately_right_of":
        return positions.get(a[0], -1) == positions.get(a[1], -1) + 1
    return False


def all_models(items: list[str], premises: list[Fact]) -> list[dict[str, int]]:
    """Return every assignment of ``items`` to positions ``1..len(items)`` that
    satisfies all ``premises`` (a list of :class:`Fact`).

    Brute force; bounded by ``len(items)!``. BBH never exceeds 7 items.
    """
    n = len(items)
    out: list[dict[str, int]] = []
    norm_items = [_norm(x) for x in items]
    for perm in itertools.permutations(range(1, n + 1)):
        pos = dict(zip(norm_items, perm))
        if all(_fact_holds(f, pos, n) for f in premises):
            out.append(pos)
    return out


def is_consistent(items: list[str], premises: list[Fact]) -> bool:
    """True if at least one permutation satisfies every premise."""
    return bool(all_models(items, premises))


def derivable(
    items: list[str],
    premises: list[Fact],
    claim: Fact,
) -> bool:
    """True iff ``claim`` is a logical consequence of ``premises``.

    Equivalent to: ``premises`` is consistent **and** every model of
    ``premises`` also satisfies ``claim``. That's the same shape
    ``derivable_under`` enforces for propositional logic — premises
    ⊨ conclusion, which is the bedrock of orate's predicate-bound
    primitives.
    """
    models = all_models(items, premises)
    if not models:
        return False  # premises themselves inconsistent → nothing derives
    n = len(items)
    return all(_fact_holds(claim, m, n) for m in models)


# ---- options grading ---------------------------------------------------

# Patterns for parsing BBH option text into a Fact. The official options
# are stylised: "The X is the leftmost", "The X finished last", etc.
# We allow ``is`` / ``are`` (plural fruits!) / ``finished`` / ``came``.
_BE = r"(?:is|are|finished|came)"
_OPT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # "The X is/are the leftmost"
    (re.compile(rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?leftmost$", re.I), "leftmost"),
    (re.compile(rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?rightmost$", re.I), "rightmost"),
    # "X finished last/first"
    (re.compile(rf"^(?:the\s+)?(.+?)\s+finished\s+last$", re.I), "rightmost"),
    (re.compile(rf"^(?:the\s+)?(.+?)\s+finished\s+first$", re.I), "leftmost"),
    # "The X is/are the oldest/newest" (age axis: oldest = leftmost)
    (re.compile(rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?oldest$", re.I), "leftmost"),
    (re.compile(rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?newest$", re.I), "rightmost"),
    # Cheapest = leftmost; most expensive = rightmost.
    (re.compile(rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?cheapest$", re.I), "leftmost"),
    (re.compile(rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?most\s+expensive$", re.I), "rightmost"),
)

# Ordinal positions: "is the second from the left", "finished third", etc.
# We map to position(item, k) where k is 1-based from the left.
_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7,
}


def parse_option(option_text: str, n_items: int) -> Fact | None:
    """Parse a BBH option line like ``"The white book is the leftmost"`` into a :class:`Fact`.

    Returns ``None`` if the option doesn't match a known shape — the
    caller treats unparseable options as non-derivable. The grader
    falls back to literal-letter matching in that case (orate runs
    aren't blocked by parser gaps; we only use this for predicate
    enforcement on @answer).
    """
    s = option_text.strip().rstrip(".")
    s = re.sub(r"^\s*", "", s)
    # Try simple leftmost/rightmost patterns first.
    for pat, pred in _OPT_PATTERNS:
        m = pat.match(s)
        if m:
            item = _norm(m.group(1))
            if pred == "leftmost":
                return Fact("position", (item, 1))
            if pred == "rightmost":
                return Fact("position", (item, n_items))

    # Compound rank: "X is the second-most expensive" → pos N-1.
    # "X is the third-cheapest" → pos 3. "X is the second-newest" → pos N-1.
    m = re.match(
        rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?(\w+)[\s-]+(?:most\s+expensive|expensive|cheapest|oldest|newest|youngest)$",
        s,
        re.I,
    )
    if m:
        item = _norm(m.group(1))
        rank_word = m.group(2).lower()
        # Re-extract the scale from the original string.
        scale_m = re.search(r"(most\s+expensive|expensive|cheapest|oldest|newest|youngest)$", s, re.I)
        if scale_m:
            scale = re.sub(r"\s+", " ", scale_m.group(1).lower())
            k = _ORDINAL_WORDS.get(rank_word)
            if k is not None and 1 <= k <= n_items:
                # Cheapest/oldest scale: rank 1 == position 1 (leftmost).
                if scale in ("cheapest", "oldest"):
                    return Fact("position", (item, k))
                # Expensive/newest scale: rank 1 == position N.
                if scale in ("expensive", "most expensive", "newest", "youngest"):
                    return Fact("position", (item, n_items + 1 - k))

    # "X finished second-to-last" / "third-to-last"
    m = re.match(
        rf"^(?:the\s+)?(.+?)\s+finished\s+(\w+)[\s-]+to[\s-]+last$",
        s,
        re.I,
    )
    if m:
        item = _norm(m.group(1))
        word = m.group(2).lower()
        k = _ORDINAL_WORDS.get(word)
        if k is not None and 1 <= k <= n_items:
            return Fact("position", (item, n_items + 1 - k))

    # Ordinal "second from the left" / "third from the right":
    m = re.match(
        rf"^(?:the\s+)?(.+?)\s+{_BE}\s+(?:the\s+)?"
        r"(\w+)(?:[\s-]+from\s+the\s+(left|right))?$",
        s,
        re.I,
    )
    if m:
        item = _norm(m.group(1))
        word = m.group(2).lower()
        side = (m.group(3) or "left").lower()
        # "third" alone after "finished"/"came" → position from the
        # left (i.e. ranking with first==best). For "from the right"
        # we mirror.
        k = _ORDINAL_WORDS.get(word)
        if k is not None and 1 <= k <= n_items:
            if side == "right":
                k = n_items + 1 - k
            return Fact("position", (item, k))
    return None
