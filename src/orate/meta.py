"""Act-4 true-meta-programming: the LLM writes its own @program.

The loop:
  1. A grammar (``PROGRAM_SOURCE_GRAMMAR``) constrains token-level decoding
     so the model can only emit well-formed @program source.
  2. ``validate_program_source`` walks the AST and rejects anything the
     grammar failed to exclude (e.g. identifiers it didn't recognise,
     disallowed subscripts, unbound names in the return expression).
  3. ``compile_program_source`` execs the validated source under a
     locked-down globals dict and returns the ``@program``-decorated
     function. The caller invokes it to get a ProgramInvocation and
     runs that against the same engine.

Three tiers of correctness — mask, AST check, sandbox — so a single
oversight doesn't let arbitrary Python escape into the host process.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from orate import gen
from orate.program import ProgramInvocation, program

__all__ = [
    "MetaProgramInvalid",
    "PROGRAM_SOURCE_GRAMMAR",
    "MetaResult",
    "compile_program_source",
    "meta_solve",
    "synthesize_program",
    "validate_program_source",
]


# ---- Grammar -------------------------------------------------------------
#
# GBNF (as consumed by XGrammar via `Grammar.from_ebnf` /
# `GrammarCompiler.compile_grammar`). The rules below admit exactly the
# program shape specified in the feature brief:
#
#   @program\n
#   def <ident>():\n
#   (    <ident> = yield gen.<method>(<args>)\n)+
#       return <ident | dict>\n
#
# Hard caps enforced at the grammar level:
#   - integer literals: 1 to 6 digits
#   - string literal chars: [A-Za-z0-9 -] (no escapes, no backslashes)
#   - gen.<method>: one of choice / integer / string / boolean
#
# Caps enforced only in the validator (the grammar is unbounded for
# these to keep the rule count tractable):
#   - identifier length: 1 to 20
#   - string literal length: 1 to 20
#
# Both caps in the brief are preserved end-to-end because the validator
# runs on every accepted sample before we exec.
PROGRAM_SOURCE_GRAMMAR = r"""
root ::= decorator header stmt-list
decorator ::= "@program\n"
header ::= "def " ident "():\n"
stmt-list ::= assign-stmt stmt-list | return-stmt
assign-stmt ::= "    " ident " = yield " gen-call "\n"
return-stmt ::= "    return " return-expr "\n"
return-expr ::= dict-expr | ident
dict-expr ::= "{" dict-pair dict-pair-rest "}"
dict-pair-rest ::= ", " dict-pair dict-pair-rest | ""
dict-pair ::= str-lit ": " ident
gen-call ::= choice-call | int-call | str-call | bool-call
choice-call ::= "gen.choice([" str-lit str-lit-rest "])"
str-lit-rest ::= ", " str-lit str-lit-rest | ""
int-call ::= "gen.integer(" int-lit ", " int-lit ")"
str-call ::= "gen.string(max_len=" int-lit ")"
bool-call ::= "gen.boolean()"
ident ::= ident-start ident-rest
ident-start ::= [a-z_]
ident-rest ::= ident-char ident-rest | ""
ident-char ::= [a-z_0-9]
str-lit ::= "\"" str-chars "\""
str-chars ::= str-char str-chars-rest
str-chars-rest ::= str-char str-chars-rest | ""
str-char ::= [a-zA-Z0-9 \-]
int-lit ::= digit digit-rest
digit ::= [0-9]
digit-rest ::= digit digit-rest5 | ""
digit-rest5 ::= digit digit-rest4 | ""
digit-rest4 ::= digit digit-rest3 | ""
digit-rest3 ::= digit digit-rest2 | ""
digit-rest2 ::= digit | ""
"""


# ---- Errors --------------------------------------------------------------


class MetaProgramInvalid(ValueError):
    """Raised by ``compile_program_source`` when validation fails.

    The message is the joined list of validator errors; callers wanting
    structured access should call ``validate_program_source`` directly.
    """


# ---- Validator -----------------------------------------------------------


_ALLOWED_METHODS = {"choice", "integer", "string", "boolean"}
_IDENT_MAX_LEN = 20
_STR_LIT_MAX_LEN = 20
_INT_LIT_MAX_DIGITS = 6
_ALLOWED_STR_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")


def validate_program_source(source: str) -> list[str]:
    """Return a list of validation error strings. Empty list = valid.

    Two-phase: ``ast.parse`` for syntax, then a manual walk for shape.
    The walker collects all findings (doesn't short-circuit on first
    error) so callers get one shot per failure mode instead of repeated
    parse-fix-reparse cycles.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    errors: list[str] = []

    # Top-level: exactly one @program-decorated FunctionDef.
    if len(tree.body) != 1:
        errors.append(f"module must contain exactly one top-level statement; got {len(tree.body)}")
        return errors
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        errors.append(
            f"top-level statement must be a function definition; got {type(node).__name__}"
        )
        return errors

    fn: ast.FunctionDef = node
    _validate_decorators(fn, errors)
    _validate_signature(fn, errors)

    if len(fn.name) < 1 or len(fn.name) > _IDENT_MAX_LEN:
        errors.append(f"function name {fn.name!r} must be 1-{_IDENT_MAX_LEN} chars")
    if not _is_valid_ident(fn.name):
        errors.append(f"function name {fn.name!r} is not a valid identifier")

    bound: set[str] = set()
    body = fn.body
    if not body:
        errors.append("function body is empty")
    else:
        # All but the last must be Assign; the last must be Return.
        *assigns, last = body
        for i, stmt in enumerate(assigns):
            _validate_assign(stmt, i, bound, errors)
        _validate_return(last, bound, errors)

    # Deep walk for globally disallowed node shapes (imports, subscripts,
    # attribute access that's not gen.*, calls to anything but gen.*).
    _walk_disallowed(fn, bound, errors)

    return errors


def _is_valid_ident(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    if not all(c.isalnum() or c == "_" for c in name):
        return False
    # Grammar restricts the ident alphabet to [a-z_0-9]; the validator
    # mirrors that so we don't accidentally accept capitals.
    return name.lower() == name


def _validate_decorators(fn: ast.FunctionDef, errors: list[str]) -> None:
    if len(fn.decorator_list) != 1:
        errors.append(
            f"function must have exactly one @program decorator; got {len(fn.decorator_list)}"
        )
        return
    dec = fn.decorator_list[0]
    if not (isinstance(dec, ast.Name) and dec.id == "program"):
        errors.append("only the @program decorator is allowed (no parentheses, no aliases)")


def _validate_signature(fn: ast.FunctionDef, errors: list[str]) -> None:
    a = fn.args
    if (
        a.args
        or a.posonlyargs
        or a.kwonlyargs
        or a.vararg
        or a.kwarg
        or a.defaults
        or a.kw_defaults
    ):
        errors.append("function must have no arguments, *args, or **kwargs")


def _validate_assign(
    stmt: ast.stmt,
    index: int,
    bound: set[str],
    errors: list[str],
) -> None:
    if not isinstance(stmt, ast.Assign):
        errors.append(f"statement #{index} must be an assignment; got {type(stmt).__name__}")
        return
    if len(stmt.targets) != 1:
        errors.append(f"statement #{index}: only single-target assignment is allowed")
        return
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        errors.append(
            f"statement #{index}: assignment target must be a bare name; "
            f"got {type(target).__name__}"
        )
        return
    if not _is_valid_ident(target.id) or len(target.id) > _IDENT_MAX_LEN:
        errors.append(
            f"statement #{index}: target {target.id!r} is not a valid "
            f"1-{_IDENT_MAX_LEN} char identifier"
        )

    value = stmt.value
    if not isinstance(value, ast.Yield):
        errors.append(
            f"statement #{index}: assignment value must be a yield expression; "
            f"got {type(value).__name__}"
        )
        return
    if value.value is None:
        errors.append(f"statement #{index}: yield must have a value")
        return
    _validate_gen_call(value.value, index, errors)

    # Bind even if there were errors inside the call; that way later
    # statements referencing it don't double-fire "unbound name".
    if isinstance(target, ast.Name):
        bound.add(target.id)


def _validate_gen_call(node: ast.expr, index: int, errors: list[str]) -> None:
    if not isinstance(node, ast.Call):
        errors.append(f"statement #{index}: yield must wrap a call; got {type(node).__name__}")
        return
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "gen"
    ):
        errors.append(
            f"statement #{index}: yielded call must be gen.<method>(...); "
            "no other callables are allowed"
        )
        return
    method = func.attr
    if method not in _ALLOWED_METHODS:
        errors.append(
            f"statement #{index}: gen.{method} is not allowed; "
            f"pick one of {sorted(_ALLOWED_METHODS)}"
        )
        return

    # Per-method argument-shape checks.
    if method == "choice":
        _check_choice_args(node, index, errors)
    elif method == "integer":
        _check_integer_args(node, index, errors)
    elif method == "string":
        _check_string_args(node, index, errors)
    elif method == "boolean":
        _check_boolean_args(node, index, errors)


def _check_choice_args(call: ast.Call, index: int, errors: list[str]) -> None:
    if call.keywords:
        errors.append(f"statement #{index}: gen.choice takes no keyword arguments")
    if len(call.args) != 1:
        errors.append(
            f"statement #{index}: gen.choice takes exactly one positional argument (a list)"
        )
        return
    lst = call.args[0]
    if not isinstance(lst, ast.List):
        errors.append(
            f"statement #{index}: gen.choice argument must be a list literal; "
            f"got {type(lst).__name__}"
        )
        return
    if not lst.elts:
        errors.append(f"statement #{index}: gen.choice list must contain at least one option")
        return
    for elt in lst.elts:
        _check_str_lit(elt, index, errors, context="gen.choice option")


def _check_integer_args(call: ast.Call, index: int, errors: list[str]) -> None:
    if call.keywords:
        errors.append(f"statement #{index}: gen.integer takes no keyword arguments")
    if len(call.args) != 2:
        errors.append(f"statement #{index}: gen.integer takes exactly two positional int literals")
        return
    for arg in call.args:
        _check_int_lit(arg, index, errors, context="gen.integer bound")


def _check_string_args(call: ast.Call, index: int, errors: list[str]) -> None:
    if call.args:
        errors.append(f"statement #{index}: gen.string takes no positional arguments; use max_len=")
    if len(call.keywords) != 1 or call.keywords[0].arg != "max_len":
        errors.append(f"statement #{index}: gen.string requires exactly max_len=<int literal>")
        return
    _check_int_lit(call.keywords[0].value, index, errors, context="gen.string max_len")


def _check_boolean_args(call: ast.Call, index: int, errors: list[str]) -> None:
    if call.args or call.keywords:
        errors.append(f"statement #{index}: gen.boolean takes no arguments")


def _check_str_lit(
    node: ast.expr,
    index: int,
    errors: list[str],
    *,
    context: str,
) -> None:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        errors.append(
            f"statement #{index}: {context} must be a string literal; got {type(node).__name__}"
        )
        return
    s = node.value
    if len(s) < 1 or len(s) > _STR_LIT_MAX_LEN:
        errors.append(
            f"statement #{index}: {context} must be 1-{_STR_LIT_MAX_LEN} chars; got {len(s)}"
        )
    if any(c not in _ALLOWED_STR_CHARS for c in s):
        errors.append(
            f"statement #{index}: {context} {s!r} contains disallowed characters; "
            "allowed: ASCII letters, digits, space, hyphen"
        )


def _check_int_lit(
    node: ast.expr,
    index: int,
    errors: list[str],
    *,
    context: str,
) -> None:
    if not (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        errors.append(
            f"statement #{index}: {context} must be an integer literal; got {type(node).__name__}"
        )
        return
    value = node.value
    if value < 0:
        errors.append(f"statement #{index}: {context} must be non-negative; got {value}")
        return
    # Grammar permits 1..6 digits; validator mirrors that.
    digits = str(value)
    if len(digits) > _INT_LIT_MAX_DIGITS:
        errors.append(
            f"statement #{index}: {context} has {len(digits)} digits; max is {_INT_LIT_MAX_DIGITS}"
        )


def _validate_return(
    stmt: ast.stmt,
    bound: set[str],
    errors: list[str],
) -> None:
    if not isinstance(stmt, ast.Return):
        errors.append(f"last statement must be a return; got {type(stmt).__name__}")
        return
    expr = stmt.value
    if expr is None:
        errors.append("return must have a value")
        return
    if isinstance(expr, ast.Name):
        if expr.id not in bound:
            errors.append(f"return references unbound name {expr.id!r}")
        return
    if isinstance(expr, ast.Dict):
        if not expr.keys:
            errors.append("return dict must have at least one key")
        for key in expr.keys:
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                errors.append("return dict keys must be string literals")
            else:
                _check_str_lit(key, -1, errors, context="return dict key")
        for val in expr.values:
            if not isinstance(val, ast.Name):
                errors.append(f"return dict values must be bare names; got {type(val).__name__}")
            elif val.id not in bound:
                errors.append(f"return dict references unbound name {val.id!r}")
        return
    errors.append(f"return expression must be a name or a dict literal; got {type(expr).__name__}")


def _walk_disallowed(
    fn: ast.FunctionDef,
    bound: set[str],
    errors: list[str],
) -> None:
    """Catch escape-hatches that the shape-walker might miss.

    Belt-and-suspenders: the shape walker already ensures every
    statement matches an Assign(Name = Yield(Call(gen.method(...))))
    + Return template, but an attacker could still embed a disallowed
    node inside an allowed-looking slot (e.g. a Subscript used as a
    gen call argument). This walk is the final trap.
    """
    known: set[str] = set(bound) | {"gen", "program"}
    for node in ast.walk(fn):
        if isinstance(node, ast.Import | ast.ImportFrom):
            errors.append("import statements are not allowed")
        elif isinstance(node, ast.Subscript):
            errors.append("subscript access is not allowed")
        elif isinstance(node, ast.Attribute):
            # Only gen.<name> attribute access is permitted.
            if not (isinstance(node.value, ast.Name) and node.value.id == "gen"):
                errors.append("attribute access is only allowed on the `gen` module")
        elif isinstance(node, ast.Call):
            fn_node = node.func
            # Allowed call: gen.<method>
            if (
                isinstance(fn_node, ast.Attribute)
                and isinstance(fn_node.value, ast.Name)
                and fn_node.value.id == "gen"
            ):
                continue
            errors.append(f"calls are only allowed to gen.<method>; got {ast.dump(fn_node)}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known:
                errors.append(f"reference to unbound name {node.id!r}")
        elif isinstance(
            node,
            ast.Lambda
            | ast.AsyncFunctionDef
            | ast.FunctionDef
            | ast.ClassDef
            | ast.With
            | ast.AsyncWith
            | ast.For
            | ast.AsyncFor
            | ast.While
            | ast.If
            | ast.Try
            | ast.Global
            | ast.Nonlocal,
        ):
            # FunctionDef is allowed at top level only (the @program fn);
            # anything nested is disallowed.
            if node is fn:
                continue
            errors.append(f"{type(node).__name__} nodes are not allowed in a meta-program")


# ---- Compiler ------------------------------------------------------------


def compile_program_source(source: str) -> Callable[..., ProgramInvocation]:
    """Validate, sandbox-exec, and return the ``@program``-wrapped function.

    Raises ``MetaProgramInvalid`` if validation fails. The returned
    callable behaves exactly like a normal ``@program`` function: call
    it (no args) to get a ``ProgramInvocation``, then ``.run(engine=...)``.
    """
    errors = validate_program_source(source)
    if errors:
        raise MetaProgramInvalid("\n".join(errors))

    # The AST was already validated; safe to re-parse for the name.
    tree = ast.parse(source)
    fn_node = tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    fn_name = fn_node.name

    # Sandbox: hand-picked builtins. The validator already forbids any
    # call that isn't a gen.<method>, so in practice nothing in this
    # dict should ever be reached — but if a future bug in the validator
    # lets something through, this dict is the final stop before the
    # host-process attack surface.
    sandbox_builtins: dict[str, Any] = {
        # dict is needed for dict-literal construction.
        "dict": dict,
    }
    sandbox_globals: dict[str, Any] = {
        "__builtins__": sandbox_builtins,
        "program": program,
        "gen": gen,
    }
    sandbox_locals: dict[str, Any] = {}

    try:
        exec(compile(source, "<meta-program>", "exec"), sandbox_globals, sandbox_locals)
    except Exception as e:  # noqa: BLE001
        raise MetaProgramInvalid(f"sandbox exec failed: {e}") from e

    compiled = sandbox_locals.get(fn_name) or sandbox_globals.get(fn_name)
    if compiled is None or not callable(compiled):
        raise MetaProgramInvalid(f"compiled module does not expose callable {fn_name!r}")
    return compiled


# ---- Orchestrator: synthesize-then-invoke (the Act-4 self-referential loop)


from dataclasses import dataclass, field  # noqa: E402

SYNTHESIS_INSTRUCTIONS = """\
You write an orate @program that captures the task below.

Rules (ENFORCED by the grammar — the tokens you output are constrained):
- Start with `@program` on its own line, then `def <name>():` on the next.
- Use 4-space indentation for every body line.
- Each body line is either `<var> = yield gen.<method>(<args>)` or `return <expr>`.
- Allowed methods: gen.integer(lo, hi), gen.choice(["a", "b", ...]),
  gen.string(max_len=N), gen.boolean().
- Return either a single identifier OR a dict whose keys are string
  literals and whose values are identifiers bound earlier in the body.
- Identifiers are lowercase ASCII; string literals are double-quoted
  with no escapes.
- No imports, no control flow (no if/for/while), no arbitrary calls.

Task:
{task}

Write exactly the @program source. Nothing else.
"""


@dataclass
class MetaResult:
    """Outcome of a meta-programming run.

    ``source`` is the exact bytes the engine emitted; ``value`` is what
    the compiled @program returned when run. ``synthesis_attempts`` is
    how many times we had to re-sample before validation passed.
    """

    source: str
    value: Any
    synthesis_attempts: int
    trace: list[dict] = field(default_factory=list)


def synthesize_program(
    engine: Any,
    *,
    task: str,
    max_retries: int = 3,
    grammar: str = PROGRAM_SOURCE_GRAMMAR,
    instructions: str = SYNTHESIS_INSTRUCTIONS,
    max_tokens: int | None = 512,
) -> tuple[Callable[..., ProgramInvocation], str, list[dict]]:
    """Ask the engine to author a @program source, validate, compile, return.

    Three-tier correctness:
      1. grammar mask (enforced by the engine's ``sample_grammar``)
      2. AST validator (``validate_program_source``)
      3. sandbox exec (``compile_program_source``)

    On validator / compile failure, ``inject_context`` feeds the error
    back and the engine re-samples up to ``max_retries`` times.

    Returns ``(compiled_fn, source, trace)``. ``compiled_fn`` is a
    normal ``@program`` callable — invoke it (no args) to get a
    ``ProgramInvocation``, then ``.run(engine=...)``.
    """
    if not hasattr(engine, "sample_grammar"):
        raise TypeError(
            f"engine {type(engine).__name__} does not implement sample_grammar "
            "(needed for grammar-constrained source synthesis)"
        )

    engine.prime(instructions.format(task=task))

    trace: list[dict] = []
    last_errors: list[str] = []
    for attempt in range(max_retries + 1):
        source = engine.sample_grammar(grammar, max_tokens=max_tokens)
        errors = validate_program_source(source)
        entry: dict[str, Any] = {"attempt": attempt, "source": source, "errors": errors}
        if not errors:
            try:
                compiled = compile_program_source(source)
                entry["status"] = "accepted"
                trace.append(entry)
                return compiled, source, trace
            except MetaProgramInvalid as e:
                errors = [f"compile failed: {e}"]
                entry["errors"] = errors
        entry["status"] = "rejected"
        trace.append(entry)
        last_errors = errors
        if attempt < max_retries and hasattr(engine, "inject_context"):
            summary = "; ".join(errors[:3])
            engine.inject_context(
                f"(previous synthesis was rejected: {summary}. "
                f"Fix the issue and rewrite the @program.)"
            )

    raise MetaProgramInvalid(
        f"synthesis failed after {max_retries + 1} attempts. "
        f"last errors: {'; '.join(last_errors) or 'unknown'}"
    )


def meta_solve(
    engine: Any,
    *,
    task: str,
    max_retries: int = 3,
) -> MetaResult:
    """The full self-referential loop:

    1. engine writes a @program (grammar-constrained source synthesis)
    2. we validate + compile it
    3. the same engine runs the compiled program — now constrained by
       the schema it just authored

    The first phase fixes what the model plans to produce. The second
    phase produces it under that plan. Both phases use the same
    engine, so the model's second-phase argmax is shaped by the
    first-phase grammar it itself wrote.
    """
    compiled_fn, source, synth_trace = synthesize_program(
        engine, task=task, max_retries=max_retries
    )
    # Re-prime with the task + source so the model's phase-2 argmax has
    # full context. The same session's inject_context notes from phase-1
    # retries are preserved — a nice side-effect: the model sees why
    # earlier drafts were rejected when it fills its own schema.
    engine.prime(
        f"You previously authored this program:\n\n"
        f"```python\n{source}\n```\n\n"
        f"Task:\n{task}\n\n"
        f"Now produce the values. Respond with only the values "
        f"in the order the yields request them.\n"
    )
    invocation = compiled_fn()
    value = invocation.run(engine=engine)
    return MetaResult(
        source=source,
        value=value,
        synthesis_attempts=len(synth_trace),
        trace=synth_trace,
    )
