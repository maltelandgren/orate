"""Source-in-prompt rendering.

When a @program is executed against an LLM engine, we prepend the
program's own Python source to the prompt. The model then sees the
full schema — not as JSON-Schema fragments, but in the richer language
the program is actually written in. Gen `description=` hints become
inline comments above each yield, mirroring Anthropic's
tool-schema-as-text approach.

Entry point is `build_prompt`. The lower-level renderers
(`render_program_source`, `render_program_with_descriptions`) are
exported for callers that want the raw text.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Literal

_SOURCE_UNAVAILABLE = "# <source unavailable>"


def render_program_source(program_fn: Callable) -> str:
    """Return the @program's Python source as a string, lightly cleaned.

    - Unwraps the @program decorator (uses __wrapped__).
    - Preserves indentation and comments (after dedenting the outer block).
    - Handles the case where the function is a lambda or has no source
      (returns a minimal placeholder).
    """
    target = getattr(program_fn, "__wrapped__", program_fn)
    try:
        raw = inspect.getsource(target)
    except (OSError, TypeError):
        return _SOURCE_UNAVAILABLE
    return textwrap.dedent(raw).rstrip() + "\n"


def _extract_description(call: ast.Call) -> str | None:
    """Pull a string-literal `description=` kwarg out of an ast.Call, if present."""
    for kw in call.keywords:
        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
            val = kw.value.value
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _collect_yield_descriptions(tree: ast.AST) -> dict[int, str]:
    """Map yield-line-number -> description for every annotated yield in the tree.

    Line numbers are 1-based and refer to the dedented source the tree
    was parsed from.
    """
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Yield) or node.value is None:
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        desc = _extract_description(call)
        if desc is not None:
            # Use the yield node's own line; the enclosing statement is
            # typically on the same line for `x = yield gen.foo(...)`.
            out[node.lineno] = desc
    return out


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def render_program_with_descriptions(
    program_fn: Callable,
    *args,  # noqa: ANN002 — accepted for signature compat; unused in static path
    **kwargs,  # noqa: ANN003
) -> str:
    """Annotated version: each yield's description is shown as a leading comment.

    Implemented via static AST walk over the dedented source. We never
    execute the program body (that would require an engine). If anything
    in the parse / rewrite path fails, fall back to `render_program_source`.
    """
    source = render_program_source(program_fn)
    if source == _SOURCE_UNAVAILABLE:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    try:
        descriptions = _collect_yield_descriptions(tree)
    except Exception:
        return source
    if not descriptions:
        return source

    lines = source.splitlines()
    # Build output bottom-up so inserted lines don't shift later indices.
    out_lines = list(lines)
    for lineno in sorted(descriptions.keys(), reverse=True):
        idx = lineno - 1
        if not (0 <= idx < len(out_lines)):
            continue
        indent = _indent_of(out_lines[idx])
        comment = f"{indent}# {descriptions[lineno]}"
        out_lines.insert(idx, comment)
    return "\n".join(out_lines).rstrip() + "\n"


def build_prompt(
    program_fn: Callable,
    *,
    user_prompt: str = "",
    show_source: bool = True,
    source_mode: Literal["raw", "annotated"] = "annotated",
) -> str:
    """Build the final prompt string the engine should prime() with.

    Layout when `show_source=True`::

        <user_prompt>

        Your response will be parsed by the following program. Satisfy
        every yield's constraints. Respond with only the values in order.

        ```python
        <rendered program source>
        ```

        Begin:

    When `show_source=False`, just the user_prompt is returned.
    An empty `user_prompt` does not produce a leading blank line.
    """
    if not show_source:
        return user_prompt

    if source_mode == "raw":
        rendered = render_program_source(program_fn)
    else:
        rendered = render_program_with_descriptions(program_fn)

    body = (
        "Your response will be parsed by the following program. Satisfy\n"
        "every yield's constraints. Respond with only the values in order.\n"
        "\n"
        "```python\n"
        f"{rendered.rstrip()}\n"
        "```\n"
        "\n"
        "Begin:"
    )
    if user_prompt:
        return f"{user_prompt}\n\n{body}"
    return body
