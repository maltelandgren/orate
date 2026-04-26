"""orate ``@program`` primitives for BBH logical_deduction.

Three leaves the model can emit:

  ``@premise(predicate, args)`` — asserts a constraint from the puzzle
      text. Predicate verifies the asserted fact parses cleanly and is
      consistent with the global puzzle (i.e. at least one permutation
      satisfies premises-so-far).

  ``@deduce(predicate, args)`` — derives a new ordering fact. Predicate
      verifies the claim is **forced** by the union of premises +
      previous deductions. This is the analogue of ``derivable_under``
      from ``examples/legal_steps/checkers.py`` — instead of modus
      ponens / hypothetical syllogism, the predicate solves a permutation
      satisfaction problem (see :mod:`bench.bbh.ordering`).

  ``@answer(letter)`` — commits to one of (A)..(G). Ends the turn. The
      predicate verifies that the derived facts entail the option
      identified by ``letter`` (parsed from the problem's option list,
      which we passed in via the runner's ``Knowledge`` registry).

The runtime keeps a per-Session :class:`Knowledge` instance that mutates
as @-calls succeed. Each predicate consults it. This mirrors the
algebra/logic Act-4 pattern: the predicate sees the same chain the
model is building.

Note on body-grammar constraints: orate's ``derive_body_grammar_rules``
only accepts straight-line yield bodies (no ``if``/branching after the
yields). State mutation on success therefore happens **outside** the
@program — in the runner, when it observes a non-rejected
:class:`ProgramInvoked` event. See ``run_orate.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from orate import gen, program

from .ordering import (
    PREDICATES,
    Fact,
    derivable,
    is_consistent,
    parse_fact,
    parse_option,
)


@dataclass
class Knowledge:
    """Per-problem state shared by the @-call predicates.

    ``items`` and ``options`` come from upstream parsing (the runner
    extracts them from the problem text deterministically). ``premises``
    and ``deductions`` accumulate as the model emits @-calls — but only
    after the runner observes a clean :class:`ProgramInvoked` event.

    The ``Fact`` chain is mutable on purpose: each successful @-call
    extends it, and subsequent calls' predicates see the new context.
    """

    items: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)  # "(A)" → option text
    premises: list[Fact] = field(default_factory=list)
    deductions: list[Fact] = field(default_factory=list)

    def known_facts(self) -> list[Fact]:
        return list(self.premises) + list(self.deductions)


# Module-level mutable so the @program closures can see the live Knowledge
# (orate's @program decorator captures the function at decoration time;
# we want the predicate to query the *current* puzzle, set up just-in-time
# by the runner before each problem).
_active: Knowledge | None = None


def set_active(knowledge: Knowledge | None) -> None:
    """Install the :class:`Knowledge` instance the predicates will read.

    Called by the runner before each problem; cleared between problems
    so leakage is impossible.
    """
    global _active  # noqa: PLW0603
    _active = knowledge


def _knowledge() -> Knowledge:
    if _active is None:
        raise RuntimeError("no active Knowledge — call set_active() before run")
    return _active


@program
def premise():
    """One premise extracted verbatim from the puzzle text.

    Two yields (predicate, args). The predicate-on-args verifies the
    fact parses + is consistent with previously-asserted premises.

    The predicate-name list is inlined here as a string literal so
    orate's body-grammar derivation can extract the choice options.
    Keep in sync with ``PREDICATES`` in ``ordering.py``.
    """
    predicate = yield gen.choice(
        [
            "left_of",
            "right_of",
            "leftmost",
            "rightmost",
            "position",
            "between",
            "immediately_left_of",
            "immediately_right_of",
        ],
        description=(
            "ordering predicate: left_of, right_of, leftmost, rightmost, "
            "position, between, immediately_left_of, immediately_right_of"
        ),
    )
    args = yield gen.string(
        max_len=80,
        description=(
            "comma-separated arguments — item names (lowercase) or "
            "an integer position; e.g. 'falcon, blue jay' or 'raven, 3'"
        ),
        where=lambda s: _premise_valid(predicate, s),
    )
    return {"predicate": predicate, "args": args}


def _premise_valid(predicate: str, args_str: str) -> bool:
    """Predicate for :func:`premise`'s arg yield.

    Accepts iff the (predicate, args) pair parses to a :class:`Fact`,
    every name argument matches a known item (after normalization),
    AND that fact, added to the current premise list, leaves at least
    one permutation consistent.

    The known-item check matters because the brute-force solver maps
    unknown names to position -1, which lets pseudo-facts like
    ``right_of(falcon, blue jay extra)`` slip past consistency
    (3 > -1 is always true). Without this check, the model could
    "assert" a junk premise and the predicate wouldn't catch it.
    """
    fact = parse_fact(f"{predicate}({args_str})")
    if fact is None:
        return False
    k = _knowledge()
    if not _all_names_known(fact, k.items):
        return False
    return is_consistent(k.items, [*k.premises, fact])


def _all_names_known(fact: Fact, items: list[str]) -> bool:
    """True iff every item-name argument of ``fact`` is in ``items``.

    For ``position``, only the first arg is a name; the second is an
    integer. For everything else, all args are names.
    """
    from .ordering import _norm  # noqa: PLC0415

    item_set = {_norm(x) for x in items}
    if fact.predicate == "position":
        return fact.args[0] in item_set
    return all(a in item_set for a in fact.args)


@program
def deduce():
    """One ordering fact derived from prior premises + deductions.

    The predicate on ``args`` verifies the claim is **forced** by the
    accumulated facts (every model satisfies the claim). Same shape as
    ``inference_step``'s ``where=`` clause — the chain is only ever
    extended with logically valid steps.
    """
    predicate = yield gen.choice(
        [
            "left_of",
            "right_of",
            "leftmost",
            "rightmost",
            "position",
            "between",
            "immediately_left_of",
            "immediately_right_of",
        ],
        description="ordering predicate (same grammar as @premise)",
    )
    args = yield gen.string(
        max_len=80,
        description="comma-separated args; the runtime checks the fact is forced by prior facts",
        where=lambda s: _deduce_valid(predicate, s),
    )
    return {"predicate": predicate, "args": args}


def _deduce_valid(predicate: str, args_str: str) -> bool:
    fact = parse_fact(f"{predicate}({args_str})")
    if fact is None:
        return False
    k = _knowledge()
    if not _all_names_known(fact, k.items):
        return False
    if not k.premises:
        # Without any premises asserted yet, nothing follows. Force the
        # model to assert premises before it deduces.
        return False
    return derivable(k.items, k.known_facts(), fact)


@program(ends_turn=True)
def answer():
    """Commit to one of (A)..(G). Ends the turn.

    Predicate: the option's parsed Fact is derivable from premises +
    deductions. We accept liberally — if the option doesn't parse to a
    Fact we know about (the parser is small, English varies), the
    predicate falls back to "is there any model consistent with the
    option?" so the orate runner never gets *stuck* — instead the answer
    might be wrong and the grader catches it.
    """
    letter = yield gen.choice(
        ["A", "B", "C", "D", "E", "F", "G"],
        description="single uppercase letter; the option that follows from your derivations",
        where=_answer_valid,
    )
    return {"letter": letter}


def _answer_valid(letter: str) -> bool:
    k = _knowledge()
    paren = f"({letter})"
    text = k.options.get(paren)
    if not text:
        return False  # we never registered this letter as an option
    fact = parse_option(text, len(k.items))
    if fact is None:
        # We couldn't parse the option text as a Fact. Fall back: accept
        # the letter — orate runner doesn't have to block. The grader
        # downstream catches wrong answers.
        return True
    # Strict path: the option must be derivable from facts emitted so far.
    return derivable(k.items, k.known_facts(), fact)


def record_invocation(name: str, args: tuple | dict) -> None:
    """Apply a successful @-call's effect to the active :class:`Knowledge`.

    Called by the runner from within the ``advance()`` loop, *after* the
    Session has accepted the call (predicate already passed). Idempotent
    failures: if the args don't parse to a Fact (shouldn't happen post-
    predicate-success, but defensively), we drop silently.
    """
    if name not in ("premise", "deduce"):
        return
    if isinstance(args, dict):
        predicate = args.get("predicate")
        args_str = args.get("args")
    else:
        # Tuple form: (predicate_str, args_str) in yield order.
        if len(args) < 2:
            return
        predicate, args_str = args[0], args[1]
    if not isinstance(predicate, str) or not isinstance(args_str, str):
        return
    fact = parse_fact(f"{predicate}({args_str})")
    if fact is None:
        return
    k = _knowledge()
    if name == "premise":
        k.premises.append(fact)
    elif name == "deduce":
        k.deductions.append(fact)
