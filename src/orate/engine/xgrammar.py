"""Local engine: llama-cpp-python + XGrammar grammar-constrained decoding.

Implements the orate Engine protocol with a real model running under
Metal (or CPU fallback). Each sample_* call compiles a fresh GBNF
grammar from the current accept set, so ADR-0014 tightening-on-reject
is the natural mode of operation — if the runner calls us again with
a value in `excluded`, we recompile the grammar without it and the
mask guarantees the model cannot return it.

Sampling is deterministic (argmax over masked logits). Stochastic
sampling is out of scope for this pass; see `sampler` field.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


def _gbnf_quote(s: str) -> str:
    """Escape a string for inclusion as a GBNF terminal literal.

    GBNF string literals are double-quoted. Backslash and double-quote
    must be escaped; everything else is literal.
    """
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _alternation_grammar(literals: Sequence[str]) -> str:
    """Build `root ::= "a" | "b" | ...` from a sequence of string literals."""
    if not literals:
        raise ValueError("alternation grammar requires at least one literal")
    alts = " | ".join(_gbnf_quote(s) for s in literals)
    return f"root ::= {alts}"


def _int_grammar(min_val: int, max_val: int, excluded: set[int]) -> str:
    """Build a flat alternation over every allowed int in [min_val, max_val].

    Simple and correct. For small ranges (the common case in orate
    programs) this compiles in a millisecond. For huge ranges a
    digit-DFA grammar would be more efficient; that's a TODO.
    """
    allowed = [i for i in range(min_val, max_val + 1) if i not in excluded]
    if not allowed:
        raise ValueError(f"sample_int: all values in [{min_val},{max_val}] are excluded")
    return _alternation_grammar([str(i) for i in allowed])


def _string_grammar(max_len: int, pattern: str | None) -> str:
    """Build a GBNF for a bounded string.

    - No pattern: any printable ASCII (0x20-0x7E except backslash/quote
      get escaped in the char class).
    - Pattern: we accept a small regex subset — character classes like
      `[a-z]`, `[A-Z0-9_]`, and the full pattern is used as the per-char
      class. Repetition is controlled by max_len (repeat 1..max_len).
      Full regex is out of scope; the caller is expected to use simple
      char classes or leave pattern=None.
    """
    if max_len < 1:
        raise ValueError("sample_string: max_len must be >= 1")
    if pattern is None:
        # Printable ASCII minus " and \ (they'd need escape in the GBNF
        # class). Callers who want quotes in their strings should pass
        # a pattern.
        char_class = "[ !#-[\\]-~]"
    else:
        # Treat `pattern` as the per-char class. If it doesn't look
        # like a bracketed class, wrap it.
        p = pattern.strip()
        # If it looks like a bracketed class use it as-is; otherwise
        # wrap it. Full regex is explicitly out of scope.
        char_class = p if p.startswith("[") and p.endswith("]") else f"[{p}]"
    # root ::= char char? char? ... (max_len copies, each optional after first)
    # A simpler form: explicit length range via repetition. GBNF doesn't
    # have bounded repetition {m,n} directly, so we unroll.
    parts = [char_class]  # one required char
    for _ in range(max_len - 1):
        parts.append(f"{char_class}?")
    body = " ".join(parts)
    return f"root ::= {body}"


def _bool_grammar() -> str:
    """Two-way alternation: `true` | `false` (lowercase)."""
    return 'root ::= "true" | "false"'


@dataclass
class XGrammarEngine:
    """Grammar-constrained local engine.

    Lazy-loaded: `_llm` etc. stay None until the first sample_* or
    explicit .load() call. Construct cheaply in tests, only pay the
    model-load cost when we actually sample.
    """

    model_path: str
    tokenizer_name: str | None = None  # HF repo id; auto-detected if None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    seed: int = 0
    max_tokens_per_sample: int = 256
    sampler: Literal["argmax"] = "argmax"

    # Lazy init
    _llm: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _tokenizer_info: Any = field(default=None, init=False, repr=False)
    _compiler: Any = field(default=None, init=False, repr=False)
    _vocab_size: int = field(default=0, init=False, repr=False)
    _bitmask: Any = field(default=None, init=False, repr=False)
    _prompt_tokens: list[int] = field(default_factory=list, init=False, repr=False)
    _prime_text: str | None = field(default=None, init=False, repr=False)

    # Session state
    _context: list[str] = field(default_factory=list, init=False, repr=False)

    # ---- load / prime ------------------------------------------------

    def load(self) -> None:
        """Instantiate the Llama + tokenizer + XGrammar compiler.

        Idempotent. Safe to call explicitly for tests that want to pay
        the load cost outside a timed hot path.
        """
        if self._llm is not None:
            return

        import llama_cpp  # noqa: PLC0415
        import xgrammar  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        self._llm = llama_cpp.Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            seed=self.seed,
        )
        self._vocab_size = int(self._llm.n_vocab())

        tokenizer_name = self.tokenizer_name or self._auto_tokenizer_name()
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
            self._tokenizer,
            vocab_size=self._vocab_size,
        )
        self._compiler = xgrammar.GrammarCompiler(self._tokenizer_info)
        self._bitmask = xgrammar.allocate_token_bitmask(1, self._vocab_size)

    def _auto_tokenizer_name(self) -> str:
        """Guess the HF tokenizer repo id from the GGUF filename."""
        name = os.path.basename(self.model_path).lower()
        if "qwen2.5-0.5b" in name:
            return "Qwen/Qwen2.5-0.5B-Instruct"
        if "qwen2.5-1.5b" in name:
            return "Qwen/Qwen2.5-1.5B-Instruct"
        if "qwen2.5-3b" in name:
            return "Qwen/Qwen2.5-3B-Instruct"
        if "qwen2.5-7b" in name:
            return "Qwen/Qwen2.5-7B-Instruct"
        # Safe default for the hackathon box.
        return "Qwen/Qwen2.5-0.5B-Instruct"

    def prime(self, prompt: str) -> None:
        """Set a system/user prompt that prefixes every sample.

        Tokenized once; we don't re-tokenize on every sample. The
        current `_context` list (inject_context calls) is prepended
        *per sample*, not here — that gets re-tokenized each time it
        changes.
        """
        self.load()
        self._prime_text = prompt
        # add_bos=True for the first pass; Qwen uses no explicit BOS
        # but llama.cpp will do the right thing.
        self._prompt_tokens = list(
            self._llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        )

    # ---- context injection ------------------------------------------

    def inject_context(self, text: str) -> None:
        """Append a steering note that will prefix the next sample."""
        self._context.append(text)

    def _full_prefix_tokens(self) -> list[int]:
        """Tokens to feed before sampling: prompt + accumulated context notes."""
        if not self._context:
            return list(self._prompt_tokens)
        note_blob = "\n" + "\n".join(f"[note: {t}]" for t in self._context) + "\n"
        self.load()
        note_toks = list(
            self._llm.tokenize(note_blob.encode("utf-8"), add_bos=False, special=False)
        )
        return list(self._prompt_tokens) + note_toks

    # ---- public sample methods --------------------------------------

    def sample_choice(self, options: Sequence[str]) -> str:
        if not options:
            raise ValueError("sample_choice: options cannot be empty")
        opts = list(options)
        grammar = _alternation_grammar(opts)
        result = self._sample_with_grammar(grammar)
        if result not in opts:
            raise RuntimeError(f"sample_choice: grammar produced {result!r}, not in {opts!r}")
        return result

    def sample_int(
        self,
        min_val: int,
        max_val: int,
        *,
        excluded: set[int] | None = None,
    ) -> int:
        if min_val > max_val:
            raise ValueError(f"sample_int: min={min_val} > max={max_val}")
        excluded = excluded or set()
        grammar = _int_grammar(min_val, max_val, excluded)
        text = self._sample_with_grammar(grammar)
        try:
            value = int(text)
        except ValueError as e:
            raise RuntimeError(f"sample_int: grammar produced non-integer {text!r}") from e
        if value < min_val or value > max_val or value in excluded:
            raise RuntimeError(f"sample_int: value {value} violated constraints")
        return value

    def sample_string(
        self,
        *,
        max_len: int,
        pattern: str | None = None,
        excluded: set[str] | None = None,
    ) -> str:
        excluded = excluded or set()
        grammar = _string_grammar(max_len, pattern)
        # Grammar cannot express "string not in excluded" cleanly for
        # free strings; we retry with a fresh sample if we land on an
        # excluded value. Because sampling is argmax, a pure retry
        # would loop forever — we perturb via inject_context.
        text = ""
        for attempt in range(4):
            text = self._sample_with_grammar(grammar)
            if text not in excluded:
                return text
            # Nudge the argmax to change on the next pass.
            self.inject_context(f"avoid repeating {text!r}; pick something different")
            if attempt == 3:
                break
        raise RuntimeError(
            f"sample_string: could not escape excluded set after retries; last={text!r}"
        )

    def sample_bool(self) -> bool:
        grammar = _bool_grammar()
        text = self._sample_with_grammar(grammar)
        if text == "true":
            return True
        if text == "false":
            return False
        raise RuntimeError(f"sample_bool: grammar produced {text!r}")

    def sample_struct(self, fields: dict[str, Any]) -> dict:
        """Fallback: sequential per-field dispatch.

        A fused JSON-schema grammar is possible (XGrammar supports it),
        but isn't worth the complexity for this pass. TODO: lower the
        Struct into a single JSON grammar when all fields are
        primitives.
        """
        return {name: spec.dispatch(self) for name, spec in fields.items()}

    # ---- the grammar-constrained sampling loop ----------------------

    def _sample_with_grammar(self, grammar: str) -> str:
        """Run deterministic argmax decoding under a GBNF constraint.

        Returns the decoded text (the grammar-accepted prefix). The
        matcher terminates when the grammar is exhausted; we stop
        there. EOS tokens from the model are ignored before the
        grammar terminates.
        """
        self.load()
        import numpy as np  # noqa: PLC0415
        import xgrammar  # noqa: PLC0415

        compiled = self._compiler.compile_grammar(grammar)
        matcher = xgrammar.GrammarMatcher(compiled)

        # Seed the kv cache with the prompt + context.
        self._llm.reset()
        prefix = self._full_prefix_tokens()
        self._llm.eval(prefix)

        produced: list[int] = []
        for _ in range(self.max_tokens_per_sample):
            if matcher.is_terminated():
                break

            # Fast path: the grammar may dictate a deterministic
            # string from here (e.g. middle of a literal like "blue").
            # We use accept_string to advance the matcher; we still
            # need to feed the corresponding tokens through the LLM
            # to keep its kv cache in sync.
            jump = matcher.find_jump_forward_string()
            if jump:
                # Tokenize the forced string, but only take tokens
                # that the matcher accepts — find_jump_forward_string
                # is best-effort and may span a token boundary
                # awkwardly. The simplest correct move is to just
                # accept it and re-tokenize the next step; we also
                # feed it through the LLM as tokens for kv-cache
                # consistency.
                forced_tokens = list(
                    self._llm.tokenize(jump.encode("utf-8"), add_bos=False, special=False)
                )
                accepted_any = False
                for tid in forced_tokens:
                    if matcher.is_terminated():
                        break
                    if matcher.accept_token(tid):
                        self._llm.eval([tid])
                        produced.append(tid)
                        accepted_any = True
                    else:
                        break
                if accepted_any:
                    continue
                # fall through to masked sampling if nothing stuck

            # Mask logits and argmax.
            matcher.fill_next_token_bitmask(self._bitmask)
            logits_ptr = self._llm._ctx.get_logits_ith(-1)
            logits = np.ctypeslib.as_array(logits_ptr, shape=(self._vocab_size,))
            # Work on a torch view so we can reuse xgrammar's inplace
            # apply without writing our own mask arithmetic.
            import torch  # noqa: PLC0415

            logits_t = torch.from_numpy(logits).clone().unsqueeze(0)
            xgrammar.apply_token_bitmask_inplace(logits_t, self._bitmask)
            tid = int(logits_t[0].argmax().item())

            if not matcher.accept_token(tid):
                # This means argmax produced the stop token before the
                # grammar was satisfied — treat as termination.
                break

            self._llm.eval([tid])
            produced.append(tid)

        # Decode; stop-token bytes won't appear because they're masked
        # while matcher is live and we break when it terminates.
        text = self._llm.detokenize(produced).decode("utf-8", errors="replace")
        return text
