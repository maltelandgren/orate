"""Derive a call-site GBNF body grammar from a @program function.

The session runner (Act-4 final form) samples under ONE outer grammar
that inlines every registered @program's body-grammar as a named
sub-rule. When the model emits ``@foo(args)`` the args are constrained
inline — no engine-level grammar switch.

This module builds the per-program body-grammar fragment by walking
the decorated function's source via ``ast``. We only inspect the yield
sequence; the return type is irrelevant at the call site.

Supported shape (exactly what ``orate.meta.PROGRAM_SOURCE_GRAMMAR``
admits): straight-line ``<var> = yield gen.<method>(<literals>)``
statements followed by a ``return``. Branches, loops, Flavor-B
(sub-program) yields, and ``yield from`` are rejected for v1 — the
whole point is a single composable fragment of regular-ish structure.

Each yield contributes one fragment to a comma-separated list. The
caller inlines the returned rules under the outer grammar by textual
concatenation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "ArgType",
    "BodyGrammarError",
    "derive_body_grammar",
    "derive_body_grammar_rules",
    "derive_call_arg_types",
    "scan_typed_args",
]


@dataclass(frozen=True)
class ArgType:
    """Type info for one positional arg of an @-call.

    The Session runner consults this when it parses ``@foo(args)``: each
    arg is decoded according to its kind so dispatch sees native Python
    values (int / bool / str) rather than the raw grammar fragment.

    ``kind`` is one of ``"choice" | "integer" | "string" | "boolean"``.
    The remaining fields carry kind-specific metadata used for
    diagnostics or future extensions; they're populated only when
    relevant.
    """

    kind: str
    options: tuple[str, ...] = ()  # for choice
    lo: int = 0  # for integer
    hi: int = 0  # for integer
    max_len: int = 0  # for string


class BodyGrammarError(ValueError):
    """Raised when a @program body cannot be lowered to a call-site grammar.

    The message identifies the offending construct (branches, loops,
    dynamic args, sub-program yields, …) so the caller can decide
    whether to re-author the program or skip it.
    """


# Flat-alternation threshold for integer ranges. Up to this many values,
# emit ``"lo" | "lo+1" | ... | "hi"``; beyond it, fall back to a digit
# DFA. The cutoff is deliberately modest — XGrammar copes with
# thousands of alternatives just fine but a DFA is cheaper to compile
# and easier to read in traces.
_INT_FLAT_MAX = 100


# ---- public entry points -------------------------------------------


def derive_body_grammar(program_fn: Callable) -> str:
    """Return a single concatenated GBNF fragment for ``program_fn``.

    The fragment contains the root rule ``<name>_body`` plus any helper
    rules referenced by it. Callers compose this with the outer grammar
    via textual concatenation.
    """
    rules = derive_body_grammar_rules(program_fn)
    return "\n".join(rules.values()) + "\n"


def derive_body_grammar_rules(program_fn: Callable) -> dict[str, str]:
    """Return ``{rule_name: rule_definition}`` for ``program_fn``.

    Root is always ``<name>_body``; helpers are prefixed with ``<name>_``
    so two programs with the same internal shape don't collide when
    their rules are merged into the outer grammar.

    Raises ``BodyGrammarError`` if ``program_fn`` is decorated as a
    composer (``@program(invocable=False)``); composers don't have a
    call-site grammar — they orchestrate leaves at runtime and are
    never embedded in another program's grammar.
    """
    _reject_composer(program_fn)
    name = _program_name(program_fn)
    fn_def = _parse_program_ast(program_fn)
    yields = _extract_yields(fn_def)

    builder = _RuleBuilder(program_name=name)
    fragments: list[str] = []
    for idx, call in enumerate(yields):
        fragments.append(builder.fragment_for_gen_call(call, idx))

    rules: dict[str, str] = {}
    # Helpers first (deterministic, but order doesn't matter semantically
    # because GBNF resolves by name); then the root.
    for rule_name, rule_body in builder.helper_rules.items():
        rules[rule_name] = f"{rule_name} ::= {rule_body}"

    root = f"{name}_body"
    if not fragments:
        rules[root] = f'{root} ::= ""'
    else:
        rules[root] = f"{root} ::= " + ' ", " '.join(fragments)
    return rules


def derive_call_arg_types(program_fn: Callable) -> list[ArgType]:
    """Return a list of ``ArgType`` (one per yielded gen call) for ``program_fn``.

    Used by the Session runner to coerce positional args at the call site
    back into native Python values: ``"42"`` → ``42`` for an integer,
    ``"true"`` → ``True`` for a boolean, ``"\\"hello\\""`` → ``"hello"``
    for a string/choice.

    Same accept-set as :func:`derive_body_grammar_rules`: only invocable
    leaf programs whose body is a straight-line sequence of
    ``var = yield gen.X(...)`` yields can be processed. Composers raise.
    """
    _reject_composer(program_fn)
    fn_def = _parse_program_ast(program_fn)
    yields = _extract_yields(fn_def)
    return [_arg_type_for_gen_call(call, idx) for idx, call in enumerate(yields)]


def scan_typed_args(text: str, arg_types: list[ArgType]) -> tuple:
    """Walk ``text`` left-to-right consuming one arg per ArgType.

    Driven by types: each kind has its own scanner that knows when
    its representation ends. ``", "`` separates args. The grammar
    emits exactly the shape we expect; no extra whitespace.

    The inverse of :func:`derive_body_grammar_rules` for the kinds
    we know about — given the text the matcher emitted and the
    ArgTypes of the program's yields, return the typed Python values
    those text spans denote.

    Returns a tuple of native Python values matching ``arg_types`` in
    order. Raises ``ValueError`` on shape mismatch with a position hint.
    """
    pos = 0
    n = len(text)
    values: list = []
    for idx, t in enumerate(arg_types):
        if idx > 0:
            if text[pos : pos + 2] != ", ":
                raise ValueError(
                    f"expected ', ' before arg #{idx} at pos {pos} in {text!r}"
                )
            pos += 2

        if t.kind == "integer":
            start = pos
            if pos < n and text[pos] == "-":
                pos += 1
            digit_start = pos
            while pos < n and text[pos].isdigit():
                pos += 1
            if pos == digit_start:
                raise ValueError(f"expected digits for arg #{idx} at pos {start}")
            values.append(int(text[start:pos]))
        elif t.kind == "boolean":
            if text.startswith("true", pos):
                values.append(True)
                pos += 4
            elif text.startswith("false", pos):
                values.append(False)
                pos += 5
            else:
                raise ValueError(f"expected true/false for arg #{idx} at pos {pos}")
        elif t.kind == "choice":
            # Choices are emitted bare per the body grammar
            # ((alt1 | alt2 | ...)). Match the longest option that
            # starts at pos.
            matched: str | None = None
            for opt in sorted(t.options, key=len, reverse=True):
                if text.startswith(opt, pos):
                    matched = opt
                    break
            if matched is None:
                raise ValueError(
                    f"arg #{idx}: expected one of {list(t.options)} at pos {pos}"
                )
            values.append(matched)
            pos += len(matched)
        elif t.kind == "string":
            if pos >= n or text[pos] != '"':
                raise ValueError(
                    f'expected \'"\' opening quote for arg #{idx} at pos {pos}'
                )
            pos += 1  # consume opening quote
            content_start = pos
            buf: list[str] = []
            while pos < n and text[pos] != '"':
                if text[pos] == "\\" and pos + 1 < n:
                    nxt = text[pos + 1]
                    buf.append('"' if nxt == '"' else "\\" if nxt == "\\" else nxt)
                    pos += 2
                else:
                    buf.append(text[pos])
                    pos += 1
            if pos >= n:
                raise ValueError(
                    f"unterminated string for arg #{idx} starting at pos {content_start}"
                )
            pos += 1  # consume closing quote
            values.append("".join(buf))
        else:
            raise ValueError(f"unknown arg kind {t.kind!r} at index {idx}")

    if pos != n:
        raise ValueError(f"trailing unparsed content at pos {pos} in {text!r}")
    return tuple(values)


def _reject_composer(program_fn: Callable) -> None:
    """Raise BodyGrammarError if ``program_fn`` is a composer.

    Looks at ``__orate_invocable__`` on the wrapper. Composers
    (``invocable=False``) do not have a call-site grammar; trying to
    derive one is a category error and almost always indicates the
    composer was passed somewhere expecting a leaf.
    """
    flag = getattr(program_fn, "__orate_invocable__", None)
    if flag is False:
        name = getattr(program_fn, "__name__", repr(program_fn))
        raise BodyGrammarError(
            f"{name!r} is a composer (@program(invocable=False)); "
            "composers don't have a call-site grammar. They orchestrate "
            "leaves and are run via .run(engine=...) directly. Pass a "
            "leaf @program (the default) to body-grammar derivation."
        )


def _arg_type_for_gen_call(call: ast.Call, index: int) -> ArgType:
    func = call.func
    assert isinstance(func, ast.Attribute)  # guaranteed by _check_straight_line_assign
    method = func.attr
    if method == "choice":
        if not call.args or not isinstance(call.args[0], ast.List):
            return ArgType(kind="choice")
        options: list[str] = []
        for elt in call.args[0].elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                options.append(elt.value)
        return ArgType(kind="choice", options=tuple(options))
    if method == "integer":
        if len(call.args) >= 2:
            try:
                lo = _as_int_literal(call.args[0], index, "integer lo", "<infer>")
                hi = _as_int_literal(call.args[1], index, "integer hi", "<infer>")
                return ArgType(kind="integer", lo=lo, hi=hi)
            except BodyGrammarError:
                pass
        return ArgType(kind="integer")
    if method == "string":
        max_len = 0
        for kw in call.keywords:
            if kw.arg == "max_len" and isinstance(kw.value, ast.Constant) and isinstance(
                kw.value.value, int
            ):
                max_len = kw.value.value
        return ArgType(kind="string", max_len=max_len)
    if method == "boolean":
        return ArgType(kind="boolean")
    raise BodyGrammarError(
        f"yield #{index} uses unsupported gen.{method}(...) for arg-type derivation"
    )


# ---- source inspection ---------------------------------------------


def _program_name(program_fn: Callable) -> str:
    name = getattr(program_fn, "__name__", None)
    if not name or name == "<lambda>":
        raise BodyGrammarError(f"cannot derive a body grammar for unnamed callable {program_fn!r}")
    if not name.isidentifier():
        raise BodyGrammarError(f"program name {name!r} is not a valid identifier")
    return name


def _parse_program_ast(program_fn: Callable) -> ast.FunctionDef:
    """Locate the generator function underlying an ``@program`` callable.

    The ``@program`` decorator wraps the original generator inside a
    ``wrapper`` closure and stashes the original on ``__wrapped__``. We
    prefer that; otherwise we fall back to ``program_fn`` itself.
    """
    target: Callable = getattr(program_fn, "__wrapped__", program_fn)

    # Dynamically compiled programs (from compile_program_source) carry
    # their source on the decorated function as `__orate_source__`. Prefer
    # that since inspect.getsource() fails on exec'd callables.
    src = getattr(program_fn, "__orate_source__", None) or getattr(target, "__orate_source__", None)
    if src is None:
        try:
            src = inspect.getsource(target)
        except (OSError, TypeError) as e:
            raise BodyGrammarError(
                f"could not get source for {program_fn!r}: {e} "
                "(for runtime-compiled programs, stash the source on "
                "`__orate_source__` before passing in)"
            ) from e
    src = textwrap.dedent(src)
    try:
        module = ast.parse(src)
    except SyntaxError as e:  # pragma: no cover - source is always valid Python
        raise BodyGrammarError(f"source for {program_fn!r} is not valid Python: {e}") from e
    fn_nodes = [n for n in module.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    if len(fn_nodes) != 1 or not isinstance(fn_nodes[0], ast.FunctionDef):
        raise BodyGrammarError(
            f"expected exactly one FunctionDef in source of {program_fn!r}; got {len(fn_nodes)}"
        )
    return fn_nodes[0]


def _extract_yields(fn: ast.FunctionDef) -> list[ast.Call]:
    """Walk the straight-line body and collect every ``gen.<method>(...)`` call.

    The body shape we accept is exactly the one admitted by
    ``PROGRAM_SOURCE_GRAMMAR``: a sequence of ``Assign(Name = Yield(Call))``
    followed by a single ``Return``. A leading docstring is permitted
    (and skipped) so hand-authored programs can document themselves.
    """
    if not fn.body:
        raise BodyGrammarError(f"function {fn.name!r} has an empty body")

    body = list(fn.body)
    # Skip a leading docstring if present.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        raise BodyGrammarError(f"function {fn.name!r}: body is empty after docstring")

    *body_stmts, last = body
    if not isinstance(last, ast.Return):
        raise BodyGrammarError(
            f"function {fn.name!r}: last statement must be `return`; got {type(last).__name__}"
        )

    # Reject control flow / nested defs first so the error message points
    # at e.g. "If" rather than the more confusing "yield in resolver"
    # secondary error you'd get if the yield happened to be inside the
    # branch body.
    _check_no_disallowed_nodes(fn)

    # Walk the body. Yield-assigns contribute to the call-site grammar;
    # plain Python statements after the yields run server-side as
    # ordinary code at predicate-verification time. The yields must come
    # first, in order, and define the call grammar — but a leaf can
    # follow them with arbitrary straight-line Python that produces the
    # resolver's return value (e.g. ``@roll`` rolls a d20 after the
    # (skill, dc) yields predicate-check; ``random.randint`` is a plain
    # Call statement, not a yield).
    yields: list[ast.Call] = []
    seen_non_yield = False
    for i, stmt in enumerate(body_stmts):
        if _is_yield_assign(stmt):
            if seen_non_yield:
                raise BodyGrammarError(
                    f"{fn.name}: yield-assign at statement #{i} comes after a "
                    "non-yield statement; all yields must precede the resolver code"
                )
            yields.append(_check_straight_line_assign(fn.name, stmt, i))
        else:
            seen_non_yield = True
            # Resolver code: control flow already rejected above; here we
            # just guard against yields buried inside a non-yield-assign
            # statement (would be weird but legal Python).
            _check_no_yields_in_resolver(fn.name, stmt, i)

    return yields


def _is_yield_assign(stmt: ast.stmt) -> bool:
    """True if ``stmt`` is ``<name> = yield <expr>`` (or ``yield from``)."""
    if not isinstance(stmt, ast.Assign):
        return False
    return isinstance(stmt.value, ast.Yield | ast.YieldFrom)


def _check_no_yields_in_resolver(fn_name: str, stmt: ast.stmt, index: int) -> None:
    """Resolver statements must not contain yields — those would mix
    grammar-bound emission with server-side computation in confusing
    ways. The yield-assigns above this point are the grammar contract;
    statements past them are pure Python.
    """
    for node in ast.walk(stmt):
        if isinstance(node, ast.Yield | ast.YieldFrom):
            raise BodyGrammarError(
                f"{fn_name}: resolver statement #{index} contains a yield; "
                "yields must precede the resolver code in a leaf @program"
            )


def _check_straight_line_assign(fn_name: str, stmt: ast.stmt, index: int) -> ast.Call:
    if not isinstance(stmt, ast.Assign):
        raise BodyGrammarError(
            f"{fn_name}: statement #{index} must be `<var> = yield gen.<method>(...)`; "
            f"got {type(stmt).__name__}"
        )
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        raise BodyGrammarError(f"{fn_name}: statement #{index} must assign to a single bare name")
    value = stmt.value
    if isinstance(value, ast.YieldFrom):
        raise BodyGrammarError(
            f"{fn_name}: `yield from` is not supported in a body-grammar-derivable program"
        )
    if not isinstance(value, ast.Yield):
        raise BodyGrammarError(
            f"{fn_name}: statement #{index} RHS must be a `yield` expression; "
            f"got {type(value).__name__}"
        )
    if value.value is None:
        raise BodyGrammarError(f"{fn_name}: statement #{index} yield has no value")
    call = value.value
    if not isinstance(call, ast.Call):
        raise BodyGrammarError(
            f"{fn_name}: statement #{index} must yield a call; got {type(call).__name__}"
        )
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "gen"
    ):
        # This is how Flavor-B / sub-program yields present — the RHS is
        # an arbitrary callable rather than ``gen.<method>``. We reject
        # them explicitly for v1.
        raise BodyGrammarError(
            f"{fn_name}: statement #{index} yields a non-gen call; "
            "sub-program (Flavor B) yields aren't supported yet"
        )
    return call


def _check_no_disallowed_nodes(fn: ast.FunctionDef) -> None:
    """Reject control-flow / nested function defs at the body level.

    We DON'T descend into kwarg values (so a ``where=lambda ...``
    survives) — only the top-level statement shape is constrained.
    Validation of model-authored sources is the separate, stricter
    pass in :func:`orate.meta.validate_program_source`.
    """
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(
                node,
                ast.If
                | ast.For
                | ast.AsyncFor
                | ast.While
                | ast.Try
                | ast.With
                | ast.AsyncWith
                | ast.Match
                | ast.FunctionDef
                | ast.AsyncFunctionDef
                | ast.ClassDef,
            ):
                raise BodyGrammarError(
                    f"{fn.name}: {type(node).__name__} is not supported in a "
                    "body-grammar-derivable program (straight-line yields only)"
                )
            if isinstance(node, ast.YieldFrom):
                raise BodyGrammarError(
                    f"{fn.name}: `yield from` is not supported in a "
                    "body-grammar-derivable program"
                )


# ---- per-yield fragment builder ------------------------------------


class _RuleBuilder:
    """Emits GBNF fragments for gen.<method>(...) calls.

    Helper rules are recorded in ``helper_rules`` (rule-name -> rhs) so
    the public API can return them alongside the root. Every rule name
    is namespaced by the program name to avoid cross-program collisions.
    """

    def __init__(self, program_name: str) -> None:
        self.program_name = program_name
        self.helper_rules: dict[str, str] = {}

    # -- dispatch ----------------------------------------------------

    def fragment_for_gen_call(self, call: ast.Call, index: int) -> str:
        func = call.func
        assert isinstance(func, ast.Attribute)  # guaranteed by _check_straight_line_assign
        method = func.attr
        if method == "choice":
            return self._fragment_choice(call, index)
        if method == "integer":
            return self._fragment_integer(call, index)
        if method == "string":
            return self._fragment_string(call, index)
        if method == "boolean":
            return self._fragment_boolean(call, index)
        raise BodyGrammarError(
            f"{self.program_name}: yield #{index} uses unsupported gen.{method}(...)"
        )

    # -- gen.choice --------------------------------------------------

    def _fragment_choice(self, call: ast.Call, index: int) -> str:
        # ``where`` and ``description`` are runtime-only — they shape
        # predicate enforcement and prompt rendering but don't change
        # the call-site grammar. Silently accept them; reject anything
        # else.
        for kw in call.keywords:
            if kw.arg not in ("where", "description"):
                raise BodyGrammarError(
                    f"{self.program_name}: yield #{index} gen.choice got unexpected "
                    f"kwarg {kw.arg!r} (only 'where' and 'description' are allowed)"
                )
        if len(call.args) != 1:
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.choice takes exactly one list literal"
            )
        lst = call.args[0]
        if not isinstance(lst, ast.List):
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.choice options must be a list literal "
                "(dynamic options can't be grammar-derived)"
            )
        options: list[str] = []
        for elt in lst.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                raise BodyGrammarError(
                    f"{self.program_name}: yield #{index} gen.choice option "
                    "must be a string literal"
                )
            options.append(elt.value)
        if not options:
            raise BodyGrammarError(f"{self.program_name}: yield #{index} gen.choice list is empty")
        # JSON-style quoting at the call site: "red" not red.
        alts = " | ".join(_gbnf_quote_string(o) for o in options)
        return f"({alts})"

    # -- gen.integer -------------------------------------------------

    def _fragment_integer(self, call: ast.Call, index: int) -> str:
        for kw in call.keywords:
            if kw.arg not in ("where", "description"):
                raise BodyGrammarError(
                    f"{self.program_name}: yield #{index} gen.integer got unexpected "
                    f"kwarg {kw.arg!r} (only 'where' and 'description' are allowed)"
                )
        if len(call.args) != 2:
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.integer needs two int literal bounds"
            )
        lo = _as_int_literal(call.args[0], index, "gen.integer lo", self.program_name)
        hi = _as_int_literal(call.args[1], index, "gen.integer hi", self.program_name)
        if lo > hi:
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.integer lo={lo} > hi={hi}"
            )
        if hi - lo + 1 <= _INT_FLAT_MAX:
            alts = " | ".join(_gbnf_quote_string(str(v)) for v in range(lo, hi + 1))
            return f"({alts})"
        rule_name = self._reserve(f"int_{index}")
        self.helper_rules[rule_name] = _digit_dfa_rhs(lo, hi)
        return rule_name

    # -- gen.string --------------------------------------------------

    def _fragment_string(self, call: ast.Call, index: int) -> str:
        if call.args:
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.string takes no positional arguments"
            )
        max_len: int | None = None
        pattern: str | None = None
        for kw in call.keywords:
            if kw.arg == "max_len":
                max_len = _as_int_literal(kw.value, index, "gen.string max_len", self.program_name)
            elif kw.arg == "pattern":
                if not (isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)):
                    raise BodyGrammarError(
                        f"{self.program_name}: yield #{index} gen.string pattern "
                        "must be a string literal"
                    )
                pattern = kw.value.value
            elif kw.arg in ("where", "description", "max_retries", "reject_message"):
                # runtime-only metadata; doesn't shape the grammar.
                continue
            else:
                raise BodyGrammarError(
                    f"{self.program_name}: yield #{index} gen.string got unexpected kwarg "
                    f"{kw.arg!r}"
                )
        if max_len is None:
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.string requires max_len=<int literal>"
            )
        if max_len < 1:
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.string max_len must be >= 1"
            )
        char_class = _pattern_to_char_class(pattern)
        rule_name = self._reserve(f"str_{index}")
        # "quote" chars quote
        body_parts = [char_class]
        for _ in range(max_len - 1):
            body_parts.append(f"{char_class}?")
        chars_body = " ".join(body_parts)
        self.helper_rules[rule_name] = f'"\\"" {chars_body} "\\""'
        return rule_name

    # -- gen.boolean -------------------------------------------------

    def _fragment_boolean(self, call: ast.Call, index: int) -> str:
        if call.args:
            raise BodyGrammarError(
                f"{self.program_name}: yield #{index} gen.boolean takes no positional arguments"
            )
        for kw in call.keywords:
            if kw.arg not in ("where", "description"):
                raise BodyGrammarError(
                    f"{self.program_name}: yield #{index} gen.boolean got unexpected "
                    f"kwarg {kw.arg!r}"
                )
        return '("true" | "false")'

    # -- helpers -----------------------------------------------------

    def _reserve(self, suffix: str) -> str:
        return f"{self.program_name}_{suffix}"


# ---- quoting & small helpers ---------------------------------------


def _gbnf_quote_string(s: str) -> str:
    """Quote a Python string as a GBNF double-quoted terminal."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _as_int_literal(node: ast.expr, index: int, ctx: str, prog: str) -> int:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    # Negative literals parse as UnaryOp(USub, Constant(int)); admit those.
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    raise BodyGrammarError(f"{prog}: yield #{index} {ctx} must be an integer literal")


def _pattern_to_char_class(pattern: str | None) -> str:
    """Reduce a regex fragment to a single GBNF char-class.

    Mirrors ``_string_grammar`` in the XGrammar engine: we only support
    bracketed classes (e.g. ``[a-z]``, ``[A-Z0-9_]``) — full regex is
    explicitly out of scope. No pattern means printable ASCII minus
    quote and backslash.
    """
    if pattern is None:
        return "[ !#-[\\]-~]"
    p = pattern.strip()
    # Strip a single trailing quantifier like `+` or `*` if present —
    # the unrolled repetition is supplied by max_len.
    if p.endswith(("+", "*")):
        p = p[:-1]
    if p.startswith("[") and p.endswith("]"):
        return p
    return f"[{p}]"


# ---- integer digit-DFA ---------------------------------------------


def _digit_dfa_rhs(lo: int, hi: int) -> str:
    """Produce a GBNF rhs that matches every integer in [lo, hi].

    Strategy: emit an alternation over each digit-width w in
    [len(str(lo_nn)), len(str(hi))], where each width contributes either
    an exact pattern (when lo and hi pin the range at that width) or a
    full 10^(w-1) .. 10^w - 1 block with a leading non-zero digit. The
    endpoints are handled by splitting into below-lo-same-width,
    middle-widths, and above-hi-same-width. Keep it simple: we only
    claim correctness via the DFA matching [lo, hi] exactly; no
    minimality.

    For the common "small range" case the caller uses a flat alternation
    instead; the DFA is reserved for larger ranges where enumerating
    every literal would blow up the grammar.
    """
    if lo < 0:
        # Negatives split into two sub-ranges: [lo, -1] (with a leading "-")
        # plus [0, hi] (positive side). Compose them.
        neg_rhs = _digit_dfa_rhs(1, -lo) if hi >= 0 else _digit_dfa_rhs(-hi, -lo)
        neg_branch = f'"-" ({neg_rhs})'
        if hi < 0:
            return neg_branch
        pos_rhs = _digit_dfa_rhs(0, hi)
        return f"{neg_branch} | ({pos_rhs})"

    # lo >= 0 from here on.
    lo_s, hi_s = str(lo), str(hi)
    n_lo, n_hi = len(lo_s), len(hi_s)

    if n_lo == n_hi:
        return _same_width_range(lo_s, hi_s)

    parts: list[str] = []
    # Lowest width: [lo_s, 10^n_lo - 1]
    upper_same = "9" * n_lo
    parts.append(_same_width_range(lo_s, upper_same))
    # Middle widths: full [10^(w-1), 10^w - 1] for w in (n_lo, n_hi).
    for w in range(n_lo + 1, n_hi):
        parts.append(_full_width_range(w))
    # Highest width: [10^(n_hi - 1), hi_s]
    lower_hi = "1" + "0" * (n_hi - 1)
    parts.append(_same_width_range(lower_hi, hi_s))
    return " | ".join(f"({p})" for p in parts)


def _full_width_range(w: int) -> str:
    """All w-digit integers with no leading zero: `[1-9] [0-9]{w-1}`."""
    tail = " ".join(["[0-9]"] * (w - 1))
    if not tail:
        return "[1-9]"
    return f"[1-9] {tail}"


def _same_width_range(lo_s: str, hi_s: str) -> str:
    """Match every integer in [lo_s, hi_s] where both have the same width.

    Exact match for single digit; otherwise split on the leading digit:
    ``lo_d`` (with its matching tail ``[lo_rest .. 9..9]``) | middle
    leading digits | ``hi_d`` (with tail ``[0..0 .. hi_rest]``). The
    recursion on the tail keeps the grammar size linear in the digit
    count.
    """
    assert len(lo_s) == len(hi_s), (lo_s, hi_s)
    w = len(lo_s)
    if w == 1:
        if lo_s == hi_s:
            return _gbnf_quote_string(lo_s)
        return f"[{lo_s}-{hi_s}]"

    lo_d, hi_d = lo_s[0], hi_s[0]
    lo_rest, hi_rest = lo_s[1:], hi_s[1:]
    nines = "9" * (w - 1)
    zeros = "0" * (w - 1)

    if lo_d == hi_d:
        inner = _same_width_range(lo_rest, hi_rest)
        return f'"{lo_d}" ({inner})'

    parts: list[str] = []
    # Leading digit == lo_d, tail in [lo_rest, 9..9]
    parts.append(f'"{lo_d}" ({_same_width_range(lo_rest, nines)})')
    # Leading digits strictly between lo_d and hi_d, tail anything
    if int(hi_d) - int(lo_d) > 1:
        mid_lo, mid_hi = chr(ord(lo_d) + 1), chr(ord(hi_d) - 1)
        tail = " ".join(["[0-9]"] * (w - 1))
        parts.append(f"[{mid_lo}-{mid_hi}] {tail}")
    # Leading digit == hi_d, tail in [0..0, hi_rest]
    parts.append(f'"{hi_d}" ({_same_width_range(zeros, hi_rest)})')
    return " | ".join(f"({p})" for p in parts)
