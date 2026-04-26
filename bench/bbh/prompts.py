"""BBH 3-shot CoT prompt for the logical_deduction subtasks.

The official Suzgun et al. BBH prompt for ``logical_deduction_three_objects``
ships at ``cot-prompts/logical_deduction_three_objects.txt`` in
https://github.com/suzgunmirac/BIG-Bench-Hard. We bake the same three
worked examples here verbatim because BBH uses the *same* exemplar set
across the 3 / 5 / 7 object subtasks (the hard variants test
generalization, not memorisation of more shots).

The prompt below reproduces the upstream text with the BIG-Bench
canary-GUID line stripped — we only need the preamble plus the three
``Q:``/``A:`` example pairs. The format teaches the model to:

  1. enumerate constraints with ``(N) ...`` numbering,
  2. visualise position with quoted ``"(left) ?  X ? Y ?  (right)"``
     fragments,
  3. emit a final ``So the answer is (X).`` line that we grep for.
"""
from __future__ import annotations

# The 3-shot exemplars are identical across BBH's 3/5/7-object splits.
# (Suzgun et al. didn't introduce harder shots for the harder variants.)
BBH_LOGICAL_DEDUCTION_PREAMBLE = (
    "A logical deduction task which requires deducing the order of a "
    "sequence of objects."
)


BBH_LOGICAL_DEDUCTION_EXEMPLARS = """\
Q: The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. In a golf tournament, there were three golfers: Amy, Eli, and Eve. Eve finished above Amy. Eli finished below Amy.
Options:
(A) Amy finished last
(B) Eli finished last
(C) Eve finished last
A: Let's think step by step.
(1) Eve finished above Amy: "(above) ? Eve ? Amy ? (below)".
(2) Eli finished below Amy: "(above) ? Amy ? Eli ? (below)".
(3) Combining (1) and (2) we get the following ordering: "(above) Eve Amy Eli (below)".
According to this ordering, the person who finished last (the one at the bottom of this list) is Eli.
Eli finished last. So the answer is (B).

Q: The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a shelf, there are three books: a white book, a green book, and an orange book. The green book is to the right of the white book. The orange book is the rightmost.
Options:
(A) The white book is the leftmost
(B) The green book is the leftmost
(C) The orange book is the leftmost
A: Let's think step by step.
(1) The green book is to the right of the white book: "(left) ? white ? green ? (right)".
(2) The orange book is the rightmost: "(left) ? white ? green orange (right)".
(3) Combining (1) and (2) we get the following ordering: "(left) white green orange (right)".
According to this ordering, the leftmost book is the white book.
The white book is the leftmost. So the answer is (A).

Q: The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a shelf, there are three books: a red book, a gray book, and a white book. The white book is to the left of the gray book. The red book is the second from the left.
Options:
(A) The red book is the leftmost
(B) The gray book is the leftmost
(C) The white book is the leftmost
A: Let's think step by step.
(1) The white book is to the left of the gray book: "(left) ? white ? gray ? (right)".
(2) The red book is the second from the left: "(left) ? white red gray ? (right)".
(3) Combining (1) and (2) we get the following ordering: "(left) white red gray (right)".
According to this ordering, the leftmost book is the white book.
The white book is the leftmost. So the answer is (C).
"""


def build_bbh_user_message(question: str) -> str:
    """Construct the user-message body for one BBH problem.

    Layout:
        <preamble>

        <three exemplar Q/A pairs>

        Q: <new question + options>
        A: Let's think step by step.

    The model's job is to fill in the reasoning + ``So the answer is (X).``
    line. We extract the last ``(X)`` token afterward.
    """
    return (
        f"{BBH_LOGICAL_DEDUCTION_PREAMBLE}\n\n"
        f"{BBH_LOGICAL_DEDUCTION_EXEMPLARS}\n"
        f"Q: {question}\n"
        f"A: Let's think step by step."
    )
