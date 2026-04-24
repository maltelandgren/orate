"""Coverage for src/orate/prompt.py — source-in-prompt rendering."""

from __future__ import annotations

from orate import build_prompt, gen, program
from orate.prompt import (
    render_program_source,
    render_program_with_descriptions,
)

# --- fixtures: real @program bodies ----------------------------------------


@program
def _signup_with_descriptions():
    name = yield gen.string(max_len=40, description="full name of the user")
    age = yield gen.integer(13, 120, description="user's age in years; must be 13+")
    return {"name": name, "age": age}


@program
def _signup_no_descriptions():
    name = yield gen.string(max_len=40)
    age = yield gen.integer(13, 120)
    return {"name": name, "age": age}


# --- render_program_source -------------------------------------------------


def test_render_program_source_unwraps_decorator():
    text = render_program_source(_signup_with_descriptions)
    # The inner function name should be in the source; the decorator
    # leaves __wrapped__ pointing at the raw body.
    assert "def _signup_with_descriptions" in text
    assert "yield gen.string" in text
    assert "yield gen.integer" in text


def test_render_program_source_is_dedented():
    text = render_program_source(_signup_no_descriptions)
    # After dedent the `def` line starts at column 0.
    assert text.startswith("@program\ndef ") or text.startswith("def ")


def test_render_program_source_lambda_falls_back():
    # Lambdas — inspect.getsource tends to grab the whole line, which can
    # still succeed. What we care about is: no crash, returns a str.
    f = lambda: None  # noqa: E731
    out = render_program_source(f)
    assert isinstance(out, str) and out


def test_render_program_source_builtin_falls_back():
    # Builtins have no source; must not crash.
    out = render_program_source(len)
    assert isinstance(out, str) and out


# --- render_program_with_descriptions --------------------------------------


def test_annotated_inserts_comments_above_yields():
    text = render_program_with_descriptions(_signup_with_descriptions)
    lines = text.splitlines()
    # Each description must appear as a comment on the line immediately
    # before the corresponding yield.
    name_yield = next(i for i, line in enumerate(lines) if "yield gen.string" in line)
    age_yield = next(i for i, line in enumerate(lines) if "yield gen.integer" in line)
    assert "# full name of the user" in lines[name_yield - 1]
    assert "# user's age in years" in lines[age_yield - 1]


def test_annotated_preserves_indentation_on_comments():
    text = render_program_with_descriptions(_signup_with_descriptions)
    lines = text.splitlines()
    comment_line = next(line for line in lines if "full name of the user" in line)
    # 4-space indent to match the function body.
    assert comment_line.startswith("    # ")


def test_annotated_no_descriptions_unchanged():
    raw = render_program_source(_signup_no_descriptions)
    annotated = render_program_with_descriptions(_signup_no_descriptions)
    assert raw == annotated
    # And no stray `#` comments were introduced above yields.
    for line in annotated.splitlines():
        assert not line.lstrip().startswith("# ") or "yield" not in line


def test_annotated_falls_back_on_builtin():
    out = render_program_with_descriptions(len)
    assert isinstance(out, str) and out


# --- build_prompt ----------------------------------------------------------


def test_build_prompt_wraps_source_block():
    out = build_prompt(_signup_with_descriptions, user_prompt="Sign the user up.")
    assert out.startswith("Sign the user up.\n\n")
    assert "Your response will be parsed by the following program." in out
    assert "```python" in out
    assert "def _signup_with_descriptions" in out
    assert out.rstrip().endswith("Begin:")


def test_build_prompt_annotated_includes_description_comments():
    out = build_prompt(_signup_with_descriptions, user_prompt="hello")
    assert "# full name of the user" in out
    assert "# user's age in years" in out


def test_build_prompt_raw_mode_skips_description_comments():
    out = build_prompt(
        _signup_with_descriptions,
        user_prompt="hello",
        source_mode="raw",
    )
    # Raw source still contains `description=` as a kwarg, but not the
    # inserted leading comment line.
    assert "description=" in out
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("# full name of the user"):
            raise AssertionError("raw mode should not insert description comments")


def test_build_prompt_show_source_false_returns_user_prompt():
    out = build_prompt(
        _signup_with_descriptions,
        user_prompt="just this",
        show_source=False,
    )
    assert out == "just this"
    assert "program" not in out
    assert "```" not in out


def test_build_prompt_empty_user_prompt_has_no_leading_blank():
    out = build_prompt(_signup_with_descriptions, user_prompt="")
    assert not out.startswith("\n")
    assert out.startswith("Your response will be parsed")


def test_build_prompt_show_source_false_and_empty_user_prompt():
    out = build_prompt(_signup_with_descriptions, user_prompt="", show_source=False)
    assert out == ""


def test_build_prompt_handles_lambda_without_crash():
    f = lambda: None  # noqa: E731
    out = build_prompt(f, user_prompt="x")
    assert "x" in out
    assert "```python" in out


def test_build_prompt_handles_builtin_without_crash():
    out = build_prompt(len, user_prompt="x")
    # Falls back to the placeholder, still produces a valid prompt.
    assert "x" in out
    assert "```python" in out
    assert "<source unavailable>" in out
