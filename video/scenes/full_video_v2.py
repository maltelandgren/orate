"""
Full hackathon submission — orate (V2).

Target ~2:30. Revised script with five "pages" and tight reveal/hide
pacing — every beat is iterative; no dump-and-hold. The visual hook
on each page is one anchor that stays put while transient text moves
through it.

  Page 1 — distribution shaping → developer-accessible → bridge        ~25s
  Page 2 — structured output (type) | the logic (program)              ~30s
  Page 3 — @algebra_step + free-text vs constrained contrast           ~30s
  Page 4 — D&D session: narration / roll / meta, then combat regrammar ~50s
           (the aria_turn cross-field where= is the headline visual)
  Page 5 — model authors its own primitive → thesis                    ~25s

Aesthetic: D1 paper. Terminal-dark insets only where literal source
code lives.

Run:
    cd video/scenes
    manim -ql --format mp4 --media_dir ../renders full_video_v2.py FullVideoV2
    manim -qm --format mp4 --media_dir ../renders full_video_v2.py FullVideoV2
    manim -qh --format mp4 --media_dir ../renders full_video_v2.py FullVideoV2
"""
from __future__ import annotations

import numpy as np
from manim import (
    BLACK,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    LaggedStart,
    Line,
    ORIGIN,
    RIGHT,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    Write,
    smooth,
)

import theme
from theme import Paper, Terminal
from llm import LLMProtagonist, LogitItem


# Thin-space (U+2009) used to add tracking to bolded display text.
# Hair-space (U+200A) tested too narrow at 1080p — barely visible. Thin
# space gives a clear breath between glyphs without breaking the read.
_THIN = "\u2009"


def _spaced(s: str, n: int = 1) -> str:
    """Add `n` thin spaces between every pair of characters of `s`.

    Used on bolded display text where manim's tight letter spacing
    cramps glyphs; one thin space gives a clear breath. We do NOT space
    across word boundaries — only between adjacent letters. Pass n=2
    for headline display text where tracking should be visibly loose.
    """
    sep = _THIN * n
    return sep.join(list(s))


# ============================================================================
# Helpers
# ============================================================================


def _chip(label: str,
          fg: str = Paper.ink,
          bg: str = Paper.card,
          stroke: str = Paper.grid,
          font_size: int = 16) -> VGroup:
    t = Text(label, font=theme.SANS_FALLBACK, font_size=font_size, color=fg)
    pad_x, pad_y = 0.32, 0.14
    pill = RoundedRectangle(
        width=t.width + pad_x * 2, height=t.height + pad_y * 2,
        corner_radius=0.22,
        fill_color=bg, fill_opacity=1.0,
        stroke_color=stroke, stroke_width=1.0,
    )
    pill.move_to(t.get_center())
    return VGroup(pill, t)


def _strike_line(target: VGroup, color: str = Paper.bad,
                 width: float = 2.2) -> Line:
    """Horizontal strike-through line across a chip / text mobject."""
    pad = 0.08
    return Line(
        target.get_left() + np.array([pad, 0, 0]),
        target.get_right() + np.array([-pad, 0, 0]),
        stroke_color=color, stroke_width=width,
    )


def _code_text(line: str, color: str, font_size: int = 16) -> Text:
    return Text(line, font=theme.MONO_FALLBACK,
                font_size=font_size, color=color)


def _code_block(lines: list[tuple[str, str]],
                anchor: np.ndarray,
                line_height: float = 0.36,
                font_size: int = 16,
                indent_em: float = 0.12) -> VGroup:
    """Render a list of (text, color) lines as a monospace block with
    indentation preserved by horizontal offset (per leading space)."""
    grp = VGroup()
    for i, (t, c) in enumerate(lines):
        # Strip leading whitespace; compute the indent in mobject units.
        n_leading = len(t) - len(t.lstrip(" "))
        stripped = t.lstrip(" ")
        ln = _code_text(stripped, c, font_size=font_size)
        # Each leading space ≈ font-size dependent — we use a unit derived
        # from the font size so larger blocks stay proportional.
        per_space = indent_em * (font_size / 14.0)
        ln.align_to(np.array([anchor[0], 0, 0]), LEFT)
        ln.shift(RIGHT * (n_leading * per_space))
        ln.shift(UP * (anchor[1] - i * line_height))
        grp.add(ln)
    return grp


# ---- Syntax-highlighted code lines --------------------------------------
#
# A "rich line" is a list of (text, color) tokens. We compose each token
# as a separate Text mobject and concatenate horizontally with a known
# per-space offset (mono so it's predictable). Trailing tokens are placed
# next-to the previous token with no buffer; a literal " " in a token
# becomes its own glyph. Indent is handled via the leading-space prefix
# in the first token.
#
# Distinct tokens take their own color, so we get:
#   - decorator (Paper.accent)
#   - keyword   (Paper.accent_soft / Terminal.amber)
#   - name      (Terminal.amber / Paper.ink)
#   - string    (Paper.good)
#   - number    (Terminal.blue)
#   - punct     (Paper.ink_soft)
#   - default   (Paper.ink)


# Cached monospace per-character width by font_size (monospace fonts have
# uniform glyph advance width).
_MONO_CHAR_WIDTH_CACHE: dict[int, float] = {}


def _mono_char_width(font_size: int) -> float:
    if font_size in _MONO_CHAR_WIDTH_CACHE:
        return _MONO_CHAR_WIDTH_CACHE[font_size]
    # Render a long ruler in mono, divide by length.
    ruler = Text("M" * 40, font=theme.MONO_FALLBACK,
                 font_size=font_size, color=Paper.ink)
    w = (ruler.get_right()[0] - ruler.get_left()[0]) / 40.0
    _MONO_CHAR_WIDTH_CACHE[font_size] = w
    return w


def _rich_line(tokens: list[tuple[str, str]],
               font_size: int = 14,
               indent_units: float = 0.0) -> VGroup:
    """Build a horizontal VGroup from (text, color) tokens.

    Strategy: monospace fonts have a constant per-character advance width.
    We compute that once per font_size and place each colored token at
    the cumulative character offset from the line's origin. This avoids
    the manim quirk where leading/trailing whitespace inside a Text gets
    collapsed (which broke an earlier Text-concatenation approach).
    """
    full_text = "".join(t for t, _ in tokens)
    if not full_text:
        return VGroup()

    char_w = _mono_char_width(font_size)
    grp = VGroup()
    char_offset = 0
    # Pick a baseline mobject (rendered first, non-empty) to use as the
    # vertical reference — its center-y becomes the row's baseline.
    baseline_y = None
    for t, c in tokens:
        if not t:
            continue
        # Whitespace-only tokens contribute width but no glyphs we need
        # to render; skip rendering but still advance the cursor.
        if t.strip() == "":
            char_offset += len(t)
            continue
        # Render token (keep internal spaces as non-breaking so manim
        # doesn't collapse them).
        rendered = t.replace(" ", "\u00a0")
        m = Text(rendered, font=theme.MONO_FALLBACK,
                 font_size=font_size, color=c)
        if baseline_y is None:
            baseline_y = m.get_center()[1]
        # Shift this token to its target x; lock baseline.
        target_x = char_offset * char_w
        dx = target_x - m.get_left()[0]
        dy = baseline_y - m.get_center()[1]
        m.shift(np.array([dx, dy, 0]))
        grp.add(m)
        char_offset += len(t)
    if indent_units:
        grp.shift(RIGHT * indent_units)
    return grp


def _rich_block(rich_lines: list[list[tuple[str, str]]],
                anchor: np.ndarray,
                line_height: float = 0.32,
                font_size: int = 14,
                indent_em: float = 0.12) -> VGroup:
    """A code block where each row is a token list (text, color).

    Indent is computed from leading whitespace on the FIRST token of each
    row. After the first token is built, we shift the entire row so its
    first token's left edge is at `anchor[0] + n_leading * per_space_unit`.
    """
    grp = VGroup()
    per_space_unit = indent_em * (font_size / 14.0)
    for i, tokens in enumerate(rich_lines):
        first_t = tokens[0][0] if tokens else ""
        n_leading = len(first_t) - len(first_t.lstrip(" "))
        if n_leading:
            new_tokens = [(first_t.lstrip(" "), tokens[0][1])] + list(tokens[1:])
        else:
            new_tokens = tokens
        line = _rich_line(new_tokens, font_size=font_size)
        if len(line) == 0:
            continue
        first = line[0]
        # x: align first child to anchor[0] + indent
        dx = (anchor[0] + n_leading * per_space_unit) - first.get_left()[0]
        # y: align center of first child to row y
        target_y = anchor[1] - i * line_height
        dy = target_y - first.get_center()[1]
        line.shift(np.array([dx, dy, 0]))
        grp.add(line)
    return grp


def _terminal_card(width: float, height: float,
                   tag: str = "  orate · session",
                   tag_above: bool = True) -> tuple[VGroup, RoundedRectangle, Text]:
    shadow = VGroup(*[
        RoundedRectangle(
            width=width + pad, height=height + pad,
            corner_radius=0.20 + pad / 2,
            fill_color=BLACK, fill_opacity=op, stroke_opacity=0,
        ).shift([0, dy, 0])
        for pad, op, dy in [(0.18, 0.025, -0.24), (0.10, 0.04, -0.16),
                            (0.04, 0.05, -0.08)]
    ])
    body = RoundedRectangle(
        width=width, height=height, corner_radius=0.20,
        fill_color=Terminal.bg, fill_opacity=1.0,
        stroke_color=Paper.ink_soft, stroke_width=1.2,
    )
    tag_txt = Text(tag, font=theme.MONO_FALLBACK,
                   font_size=13, color=Terminal.ink_soft)
    if tag_above:
        tag_txt.next_to(body, UP, buff=0.08, aligned_edge=LEFT)
    else:
        tag_txt.next_to(body, DOWN, buff=0.08, aligned_edge=LEFT)
    return shadow, body, tag_txt


def _paper_card(width: float, height: float) -> VGroup:
    """A subtle paper-tone drawer with a thin border and faint shadow."""
    shadow = VGroup(*[
        RoundedRectangle(
            width=width + pad, height=height + pad,
            corner_radius=0.16 + pad / 2,
            fill_color=BLACK, fill_opacity=op, stroke_opacity=0,
        ).shift([0, dy, 0])
        for pad, op, dy in [(0.12, 0.018, -0.18), (0.05, 0.03, -0.10)]
    ])
    body = RoundedRectangle(
        width=width, height=height, corner_radius=0.16,
        fill_color=Paper.card, fill_opacity=1.0,
        stroke_color=Paper.grid, stroke_width=1.2,
    )
    return VGroup(shadow, body)


# ============================================================================
# The video
# ============================================================================


class FullVideoV2(Scene):
    def construct(self):
        self.camera.background_color = Paper.bg
        self.add(theme.paper_grid(opacity=0.32))

        self._page1_distribution_shaping()
        self._page2_structured_output_and_logic()
        self._page3_algebra_predicate_and_contrast()
        self._page4_dnd_session_and_combat_regrammar()
        self._page5_meta_authorship_and_close()

    # ======================================================================
    # PAGE 1 — distribution shaping → developer-accessible → bridge
    # ======================================================================

    def _page1_distribution_shaping(self):
        # Beat 1.A — headline.
        head = Text(
            "Three years in, we're pretty good at shaping the",
            font="Georgia", slant="ITALIC", font_size=28, color=Paper.ink,
        )
        head2 = Text(
            "output distribution of LLMs.",
            font="Georgia", slant="ITALIC", font_size=28, color=Paper.ink,
        )
        head_grp = VGroup(head, head2).arrange(DOWN, buff=0.18)
        head_grp.move_to(UP * 1.6)
        self.play(FadeIn(head, shift=UP * 0.1, run_time=0.55))
        self.play(FadeIn(head2, shift=UP * 0.08, run_time=0.5))
        self.wait(1.8)

        # Beat 1.B — two columns: training-time vs inference-time.
        # The two columns sit closer together than before — feedback was
        # the original layout left a wide negative-space gap that read as
        # disconnection rather than contrast.
        tt_label = Text("training time", font=theme.SANS_FALLBACK,
                        font_size=14, color=Paper.ink_soft)
        it_label = Text("inference time", font=theme.SANS_FALLBACK,
                        font_size=14, color=Paper.ink_soft)
        tt_label.move_to(np.array([-3.7, 0.5, 0]))
        it_label.move_to(np.array([-0.6, 0.5, 0]))

        ft_chip = _chip("fine-tuning")
        ft_chip.next_to(tt_label, DOWN, buff=0.25)
        # Center fine-tuning under its label
        ft_chip.align_to(tt_label, LEFT)

        inf_chips = VGroup(
            _chip("prompts"),
            _chip("structured output"),
            _chip("tool calls"),
        ).arrange(RIGHT, buff=0.3)
        inf_chips.next_to(it_label, DOWN, buff=0.25)
        inf_chips.align_to(it_label, LEFT)

        self.play(
            FadeIn(tt_label, run_time=0.3),
            FadeIn(it_label, run_time=0.3),
        )
        self.play(
            FadeIn(ft_chip, shift=UP * 0.05, run_time=0.4),
            LaggedStart(*[FadeIn(c, shift=UP * 0.05) for c in inf_chips],
                        lag_ratio=0.22, run_time=1.0),
        )
        self.wait(1.4)

        # Beat 1.C — strike out training-time. Inference-time → "Developer accessible".
        # Bolder strikethrough — feedback called the original line too thin.
        strike_ft = _strike_line(ft_chip, color=Paper.bad, width=4.5)
        self.play(FadeIn(strike_ft, run_time=0.45))
        self.play(
            ft_chip.animate.set_opacity(0.35),
            strike_ft.animate.set_opacity(0.6),
            tt_label.animate.set_opacity(0.35),
            run_time=0.45,
        )
        self.wait(0.4)

        # Relabel "inference time" → "developer accessible"
        new_label = Text("developer accessible", font=theme.SANS_FALLBACK,
                         font_size=14, color=Paper.accent)
        new_label.move_to(it_label.get_center()).align_to(it_label, LEFT)
        self.play(
            FadeOut(it_label, shift=UP * 0.05, run_time=0.3),
            FadeIn(new_label, shift=UP * 0.05, run_time=0.35),
        )
        self.wait(1.2)

        # Beat 1.D — fade the headline + struck-out column. Center remaining chips.
        self.play(
            FadeOut(head_grp, run_time=0.35),
            FadeOut(VGroup(tt_label, ft_chip, strike_ft), run_time=0.35),
        )
        # Move remaining chips to upper-third
        new_chips = VGroup(*[c for c in inf_chips])
        target_chips = VGroup(*[c.copy() for c in new_chips]).arrange(
            RIGHT, buff=0.35,
        )
        target_chips.move_to(UP * 2.0)
        # Map inf_chips one-to-one to target positions
        anims = []
        for src, dst in zip(new_chips, target_chips):
            anims.append(src.animate.move_to(dst.get_center()))
        anims.append(new_label.animate.move_to(UP * 2.7))
        self.play(*anims, run_time=0.7)
        self.wait(0.5)

        # Beat 1.E — one-line per chip, expand below. Use a single anchor.
        # We'll cycle: prompts → structured output → tool calls.
        # For tool calls, just pivot it back into "structured output" since
        # the script focuses there — but we still highlight all three briefly
        # to keep the user's flow.
        prompts_chip, so_chip, tools_chip = new_chips
        # Reusable single line under the row
        anchor_y = 0.9

        def _highlight(chip: VGroup) -> list:
            return [
                chip.animate.set_stroke(Paper.accent, width=1.8),
            ]

        def _unhighlight(chip: VGroup) -> list:
            return [
                chip.animate.set_stroke(Paper.grid, width=1.0),
            ]

        # prompts: very flexible, intuitive, but fiddly. Hard to write programs around.
        prompts_line1 = Text(
            "very flexible · intuitive · fiddly",
            font="Georgia", slant="ITALIC", font_size=20,
            color=Paper.ink_soft,
        )
        prompts_line1.move_to(np.array([0, anchor_y, 0]))
        self.play(*_highlight(prompts_chip),
                  FadeIn(prompts_line1, shift=UP * 0.08, run_time=0.4))
        self.wait(1.4)

        # "Hard to write programs around." attaches to PROMPTS only.
        # Bolded display text uses tracked letter-spacing (hair spaces) —
        # cramped glyphs were visible at 1080p without it.
        prompts_punch = Text(
            _spaced("Hard to write programs around."),
            font="Georgia", weight="BOLD", font_size=22, color=Paper.ink,
        )
        prompts_punch.move_to(np.array([0, anchor_y - 0.55, 0]))
        self.play(FadeIn(prompts_punch, shift=UP * 0.08, run_time=0.4))
        self.wait(2.0)

        # structured output: bridge — same line transforms.
        # "Lets us write programs around" highlights BOTH structured output
        # AND tool calls (both are program-friendly inference primitives).
        self.play(*_unhighlight(prompts_chip),
                  *_highlight(so_chip),
                  *_highlight(tools_chip),
                  FadeOut(prompts_line1, shift=UP * 0.05, run_time=0.3))

        # The bridge: "Hard to write programs around" → "Lets us write programs around".
        # Use cross-fade rather than Transform — Transform morphs glyph-by-glyph
        # and produces a glitchy intermediate frame.
        prompts_punch_target = Text(
            _spaced("Lets us write programs around."),
            font="Georgia", weight="BOLD", font_size=22, color=Paper.accent,
        )
        prompts_punch_target.move_to(prompts_punch.get_center())
        self.play(
            FadeOut(prompts_punch, shift=UP * 0.1, run_time=0.4),
            FadeIn(prompts_punch_target, shift=UP * 0.1, run_time=0.45),
        )
        # Carry the new text on going forward so cleanup picks it up.
        prompts_punch = prompts_punch_target
        self.wait(2.0)

        # follow-on caption:
        so_caption = Text(
            "constrained grammar over decoding → well-formed output",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink_soft,
        )
        so_caption.move_to(np.array([0, anchor_y - 1.15, 0]))
        self.play(FadeIn(so_caption, shift=UP * 0.06, run_time=0.4))
        self.wait(2.2)

        # Beat 1.F — dim everything else; carry "structured output" forward.
        self.play(
            FadeOut(VGroup(prompts_chip, tools_chip, prompts_punch,
                           so_caption, new_label), run_time=0.4),
        )
        # so_chip stays — page 2 will dock it top-left as a header.
        self._so_chip = so_chip  # carry across beats

    # ======================================================================
    # PAGE 2 — structured output (type) | the logic (program)
    # ======================================================================

    def _page2_structured_output_and_logic(self):
        so_chip = self._so_chip

        # Beat 2.A — dock structured output → top-left header. Drawer drops.
        target = _chip("structured output", font_size=16)
        target.move_to(np.array([-3.8, 3.55, 0]))
        target.set_stroke(Paper.accent, width=1.8)
        self.play(Transform(so_chip, target, run_time=0.6))

        left_drawer = _paper_card(width=6.4, height=4.4)
        left_drawer.move_to(np.array([-3.55, 0.2, 0]))
        # Drawer drops in from beneath the header
        left_drawer.shift(UP * 0.6)
        self.play(
            left_drawer.animate.shift(DOWN * 0.6),
            FadeIn(left_drawer, run_time=0.45),
            run_time=0.45,
        )

        # Beat 2.B — LLM fills a schema inside the drawer with annotations.
        # Syntax highlighting:
        #   keys (e.g. "name") in Paper.ink
        #   string values in Paper.good
        #   numbers in Terminal.blue
        #   booleans in Terminal.amber
        #   punctuation muted (Paper.ink_soft)
        schema_anchor = np.array([-6.55, 2.1, 0])
        P = Paper.ink_soft  # punctuation
        schema_rich = [
            [("{", P)],
            [('  ', Paper.ink),
             ('"name"', Paper.ink),
             (':', P),
             ('     ', Paper.ink),
             ('"Aria"', Paper.good),
             (',', P)],
            [('  ', Paper.ink),
             ('"level"', Paper.ink),
             (':', P),
             ('    ', Paper.ink),
             ('3', Terminal.blue),
             (',', P)],
            [('  ', Paper.ink),
             ('"class"', Paper.ink),
             (':', P),
             ('    ', Paper.ink),
             ('"bard"', Paper.good),
             (',', P)],
            [('  ', Paper.ink),
             ('"alive"', Paper.ink),
             (':', P),
             ('    ', Paper.ink),
             ('true', Terminal.amber),
             (',', P)],
            [("}", P)],
        ]
        schema = _rich_block(schema_rich, anchor=schema_anchor,
                             line_height=0.40, font_size=16)
        # We'll reveal the lines progressively + caption each field.

        self.play(FadeIn(schema[0], run_time=0.18))  # opening brace

        captions = []

        def reveal_field(idx: int, type_label: str, type_color: str,
                         tail: str = "") -> None:
            """Reveal schema line idx, point a typed caption to its right.

            The type_label (String / Integer / Enum / Boolean) renders
            BOLD in `type_color`; the tail (e.g. " [bard, cleric, rogue]")
            renders in muted text. Pacing is ~30% slower than the
            original feedback round.
            """
            self.play(FadeIn(schema[idx], shift=LEFT * 0.1, run_time=0.4))
            arrow = Text("← must be a", font=theme.MONO_FALLBACK,
                         font_size=12, color=Paper.ink_soft)
            type_t = Text(_spaced(type_label, n=1),
                          font=theme.MONO_FALLBACK, weight="BOLD",
                          font_size=13, color=type_color)
            cap_grp = VGroup(arrow, type_t).arrange(RIGHT, buff=0.18)
            if tail:
                tail_t = Text(tail, font=theme.MONO_FALLBACK,
                              font_size=12, color=Paper.ink_soft)
                cap_grp = VGroup(arrow, type_t, tail_t).arrange(
                    RIGHT, buff=0.18,
                )
            cap_grp.next_to(schema[idx], RIGHT, buff=0.35)
            self.play(FadeIn(cap_grp, shift=LEFT * 0.05, run_time=0.35))
            captions.append(cap_grp)
            # ~30% longer hold than the prior 0.25s pacing.
            self.wait(0.55)

        reveal_field(1, "String", Paper.accent)
        reveal_field(2, "Integer", Terminal.blue)
        reveal_field(3, "Enum", Paper.accent_soft,
                     tail="  [bard, cleric, rogue]")
        reveal_field(4, "Boolean", Terminal.amber)
        self.play(FadeIn(schema[5], run_time=0.2))  # closing brace
        self.wait(1.0)

        # Beat 2.C — the full-quote bridge. We render every clause as
        # its own mobject so we can keep "the logic" while fading the
        # rest, and then dock "the logic" up to the upper-right header
        # position (mirroring "structured output" upper-left).
        self.play(FadeOut(VGroup(*captions), run_time=0.35))

        bridge_pre = Text(
            "With structured output we've put typing",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink_soft,
        )
        bridge_pre2 = Text(
            "directly into the decoding process.",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink_soft,
        )
        # Question line: "Where's the logic in that?"
        # Build as separate mobjects so "the logic" can survive the fade.
        q_left = Text("Where's", font="Georgia", slant="ITALIC",
                      font_size=22, color=Paper.ink)
        q_logic = Text(_spaced("the logic"),
                       font="Georgia", weight="BOLD", font_size=22,
                       color=Paper.accent)
        q_right = Text("in that?", font="Georgia", slant="ITALIC",
                       font_size=22, color=Paper.ink)
        question_row = VGroup(q_left, q_logic, q_right).arrange(
            RIGHT, buff=0.22,
        )
        bridge_grp = VGroup(bridge_pre, bridge_pre2, question_row).arrange(
            DOWN, buff=0.14,
        )
        bridge_grp.to_edge(DOWN, buff=0.55)

        self.play(FadeIn(bridge_pre, shift=UP * 0.06, run_time=0.55))
        self.play(FadeIn(bridge_pre2, shift=UP * 0.06, run_time=0.55))
        self.wait(0.5)
        self.play(FadeIn(question_row, shift=UP * 0.08, run_time=0.55))
        self.wait(2.0)

        # Beat 2.D — fade everything except "the logic"; dock it to the
        # upper-right header position. Mirrors "structured output" top-left.
        logic_dock_target = _chip("the logic", font_size=16,
                                  fg=Paper.accent, stroke=Paper.accent)
        logic_dock_target.move_to(np.array([3.8, 3.55, 0]))
        # We DO NOT include `q_logic` in the fade-out — it's the survivor.
        self.play(
            FadeOut(VGroup(bridge_pre, bridge_pre2, q_left, q_right),
                    run_time=0.45),
        )
        # Animate the survivor up to its dock position. We rebuild as a
        # chip there to mirror so_chip's geometry; the text travels via
        # a Transform on the survivor's center.
        self.play(
            q_logic.animate.move_to(logic_dock_target.get_center())
                            .scale(0.8),
            run_time=0.7,
        )
        # Place the chip behind the now-docked text and fade it in.
        self.play(FadeIn(logic_dock_target, run_time=0.35))
        # The text on top of the chip should be the chip's own label —
        # remove the survivor and let the chip text take over.
        self.remove(q_logic)
        logic_chip = logic_dock_target

        right_drawer = _paper_card(width=6.4, height=4.4)
        right_drawer.move_to(np.array([3.55, 0.2, 0]))
        right_drawer.shift(UP * 0.6)
        self.play(
            right_drawer.animate.shift(DOWN * 0.6),
            FadeIn(right_drawer, run_time=0.45),
            run_time=0.45,
        )

        # Beat 2.E — draw the dm_turn @program in the right drawer.
        # SYNTAX HIGHLIGHTED:
        #   @program (Paper.accent)
        #   def keyword (Paper.accent_soft) — function name (Terminal.amber)
        #   yield / if / return (Paper.accent_soft)
        #   gen.X attribute (Terminal.amber)
        #   strings (Paper.good), numbers (Terminal.blue)
        right_anchor = np.array([0.7, 2.1, 0])
        ACC = Paper.accent
        KW  = Paper.accent_soft   # keywords
        FN  = Terminal.amber      # function / attribute names
        ID  = Paper.ink           # identifiers + neutral
        DIM = Paper.ink_soft      # dimmed
        NUM = Terminal.blue
        prog_rich = [
            [("@program", ACC)],
            [("def ", KW), ("dm_turn", FN), ("(", DIM),
             ("scene", ID), ("):", DIM)],
            [("    narration  = ", ID), ("yield ", KW),
             ("gen.string", FN), ("(...)", DIM)],
            [("    needs_roll = ", ID), ("yield ", KW),
             ("gen.boolean", FN), ("()", DIM)],
            [("    if ", KW), ("needs_roll", ID), (":", DIM)],
            [("        dc     = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("5", NUM), (", ", DIM), ("25", NUM), (")", DIM)],
            [("        result = ", ID), ("yield ", KW),
             ("gen.tool", FN), ("(", DIM)],
            [("                     ", ID), ("roll_d20", FN),
             (", dc=dc)", DIM)],
            [("    npc_line   = ", ID), ("yield ", KW),
             ("gen.string", FN), ("(...)", DIM)],
            [("    return ", KW), ("{...}", DIM)],
        ]
        prog = _rich_block(prog_rich, anchor=right_anchor,
                           line_height=0.32, font_size=13)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in prog],
            lag_ratio=0.13, run_time=2.4,
        ))
        self.wait(2.4)

        # Punchline caption
        punch = Text(
            "Types, tool calls, control flow — same yield stream.",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink,
        )
        punch.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(punch, shift=UP * 0.06, run_time=0.5))
        self.wait(3.4)

        # Clean page (carry nothing forward except mental anchor).
        self.play(FadeOut(VGroup(so_chip, logic_chip, left_drawer,
                                 right_drawer, schema, prog, punch),
                          run_time=0.45))

    # ======================================================================
    # PAGE 3 — @algebra_step + free-text vs constrained contrast
    # ======================================================================

    def _page3_algebra_predicate_and_contrast(self):
        # Beat 3.A — headline that sets up "well-defined problems".
        intro = Text(
            "Where the problem is well-defined, the program forces",
            font="Georgia", slant="ITALIC", font_size=22, color=Paper.ink,
        )
        intro2 = Text(
            "the model to reason inside its rules.",
            font="Georgia", slant="ITALIC", font_size=22, color=Paper.ink,
        )
        intro_grp = VGroup(intro, intro2).arrange(DOWN, buff=0.16)
        intro_grp.to_edge(UP, buff=0.7)
        self.play(FadeIn(intro, shift=UP * 0.1, run_time=0.4))
        self.play(FadeIn(intro2, shift=UP * 0.08, run_time=0.4))
        self.wait(0.6)

        # Beat 3.B — algebra_step source on the left, syntax-highlighted.
        anchor_l = np.array([-6.7, 0.9, 0])
        ACC = Paper.accent
        KW  = Paper.accent_soft
        FN  = Terminal.amber
        ID  = Paper.ink
        DIM = Paper.ink_soft
        STR = Paper.good
        algebra_rich = [
            [("@program", ACC)],
            [("def ", KW), ("algebra_step", FN), ("():", DIM)],
            [("    before = ", ID), ("yield ", KW),
             ("gen.string", FN), ("(...)", DIM)],
            [("    rule   = ", ID), ("yield ", KW),
             ("gen.choice", FN), ("([", DIM)],
            [("        'simplify'", STR), (", ", DIM),
             ("'isolate_var'", STR), (",", DIM)],
            [("        'evaluate'", STR), (", ", DIM),
             ("'combine_like'", STR), ("])", DIM)],
            [("    after  = ", ID), ("yield ", KW),
             ("gen.string", FN), ("(", DIM)],
            [("        where=", ACC), ("lambda ", KW), ("s", ID), (":", DIM)],
            [("            equivalent_under", FN), ("(", DIM)],
            [("                rule, before, s", ID), ("))", DIM)],
            [("    return ", KW), ("{...}", DIM)],
        ]
        algebra = _rich_block(algebra_rich, anchor=anchor_l,
                              line_height=0.34, font_size=14)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in algebra],
            lag_ratio=0.10, run_time=2.0,
        ))
        self.wait(0.7)

        # Highlight the where= line specifically
        where_box = RoundedRectangle(
            width=5.4, height=1.18, corner_radius=0.10,
            fill_color=Paper.accent, fill_opacity=0.10,
            stroke_color=Paper.accent, stroke_width=1.4,
        )
        # Centered around lines [7..9] (where= block)
        center_y = (algebra[7].get_center()[1] + algebra[9].get_center()[1]) / 2
        center_x = anchor_l[0] + 2.7
        where_box.move_to(np.array([center_x, center_y, 0]))
        # Caption goes to the RIGHT of the box, mid-height — avoids overlapping
        # the `return {...}` line beneath the highlighted block.
        where_caption = Text("logic constraint,",
                             font="Georgia", slant="ITALIC",
                             font_size=14, color=Paper.accent)
        where_caption2 = Text("in Python.",
                              font="Georgia", slant="ITALIC",
                              font_size=14, color=Paper.accent)
        where_caption_grp = VGroup(where_caption, where_caption2).arrange(
            DOWN, aligned_edge=LEFT, buff=0.06,
        )
        where_caption_grp.next_to(where_box, RIGHT, buff=0.25)
        self.play(FadeIn(where_box, run_time=0.4),
                  FadeIn(where_caption_grp, run_time=0.4))
        self.wait(1.8)

        # Beat 3.C — same problem, two runs (right side).
        anchor_r_top = 1.5
        prob = Text("Solve:   3x + 5 = 14",
                    font=theme.MONO_FALLBACK, font_size=18,
                    color=Paper.ink_soft)
        prob.move_to(np.array([3.7, anchor_r_top, 0]))
        self.play(FadeIn(prob, run_time=0.3))

        # Two stacked outputs
        free_label = Text("free text", font=theme.SANS_FALLBACK,
                          font_size=12, color=Paper.ink_soft)
        free_label.move_to(np.array([1.5, anchor_r_top - 0.6, 0]))
        free_lines = [
            ("3x = 14 - 5", Paper.ink),
            ("3x = 9", Paper.ink),
            ("x = 9 / 3 = 4", Paper.bad),
        ]
        free_block = VGroup()
        for i, (t, c) in enumerate(free_lines):
            ln = _code_text(t, c, font_size=14)
            ln.move_to(np.array([3.6, anchor_r_top - 1.05 - 0.36 * i, 0]))
            ln.align_to(np.array([1.5, 0, 0]), LEFT)
            free_block.add(ln)
        # red strike-through on the wrong answer
        free_x_mark = Text("  ✗", font=theme.MONO_FALLBACK,
                           font_size=15, color=Paper.bad)
        free_x_mark.next_to(free_block[2], RIGHT, buff=0.15)

        self.play(FadeIn(free_label, run_time=0.3),
                  LaggedStart(*[FadeIn(l, shift=LEFT * 0.06) for l in free_block],
                              lag_ratio=0.3, run_time=1.2))
        self.play(FadeIn(free_x_mark, run_time=0.25))
        self.wait(1.2)

        # Constrained run beneath. Each call sits on a SINGLE line —
        # earlier renders broke at the 2nd argument and read as ugly.
        # We tighten by dropping internal whitespace, shrinking font 1pt,
        # and rendering each token rich-coloured (call name in ink, args
        # in muted, OK-mark in green).
        cons_label = Text("under @algebra_step", font=theme.SANS_FALLBACK,
                          font_size=12, color=Paper.accent)
        cons_label.move_to(np.array([1.5, anchor_r_top - 2.5, 0]))
        # Build lines as token VGroups so call-name / arg / mark each
        # take their own colour without relying on t2c heuristics.
        cons_specs = [
            [("@algebra_step", Paper.ink),
             ('("3x+5=14", simplify, "3x=9")', Paper.ink_soft),
             ("  ✓", Paper.good)],
            [("@algebra_step", Paper.ink),
             ('("3x=9", isolate_var, "x=3")', Paper.ink_soft),
             ("  ✓", Paper.good)],
            [("@done", Paper.accent),
             ('("x = 3")', Paper.ink_soft),
             ("  ✓", Paper.good)],
        ]
        cons_block = VGroup()
        line_anchor_x = 1.5
        for i, tokens in enumerate(cons_specs):
            ln = _rich_line(tokens, font_size=12)
            ln.align_to(np.array([line_anchor_x, 0, 0]), LEFT)
            ln.move_to(np.array([
                ln.get_center()[0],
                anchor_r_top - 2.92 - 0.34 * i,
                0,
            ]))
            ln.align_to(np.array([line_anchor_x, 0, 0]), LEFT)
            cons_block.add(ln)
        self.play(FadeIn(cons_label, run_time=0.3))
        self.play(LaggedStart(
            *[FadeIn(l, shift=LEFT * 0.06) for l in cons_block],
            lag_ratio=0.22, run_time=1.4,
        ))
        self.wait(1.8)

        # Beat 3.D — benchmark line + same weights, different gate.
        bench = Text(
            "free-text 4/7   ·   constrained 6/7   ·   11 illegal-step rejections",
            font=theme.MONO_FALLBACK, font_size=14, color=Paper.ink_soft,
        )
        bench.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(bench, shift=UP * 0.08, run_time=0.5))
        self.wait(2.0)
        gate = Text(
            "Same weights. Different gate.",
            font="Georgia", slant="ITALIC", font_size=18, color=Paper.accent,
        )
        gate.next_to(bench, DOWN, buff=0.18)
        self.play(FadeIn(gate, shift=UP * 0.05, run_time=0.45))
        self.wait(3.4)

        # Clean page.
        self.play(FadeOut(VGroup(intro_grp, algebra, where_box,
                                  where_caption_grp,
                                  prob, free_label, free_block, free_x_mark,
                                  cons_label, cons_block, bench, gate),
                          run_time=0.45))

    # ======================================================================
    # PAGE 4 — D&D session: narration / roll / meta, then combat regrammar
    # ======================================================================

    def _page4_dnd_session_and_combat_regrammar(self):
        # ====================================================================
        # Beat 4.A — page transition. The header that EARNS the reveal:
        #   "We can nest and compose programs together."
        # then morphs to the punchier "One KV. Many grammars." for the body.
        # ====================================================================
        compose_header = Text(
            _spaced("We can nest and compose programs together."),
            font="Georgia", weight="BOLD", font_size=26, color=Paper.ink,
        )
        compose_header.to_edge(UP, buff=0.65)
        self.play(FadeIn(compose_header, shift=UP * 0.08, run_time=0.5))
        self.wait(2.4)

        title = Text(
            _spaced("One KV. Many grammars."),
            font="Georgia", slant="ITALIC", weight="BOLD",
            font_size=22, color=Paper.ink_soft,
        )
        title.to_edge(UP, buff=0.4)
        # Cross-fade to the title — keeps a header anchor at the top.
        self.play(
            FadeOut(compose_header, shift=UP * 0.1, run_time=0.45),
            FadeIn(title, shift=UP * 0.06, run_time=0.5),
        )
        self.wait(0.4)

        # ====================================================================
        # Tab indicator (the navigational anchor). Two levels:
        #   OUTER:  Many grammars  [ *narrative* | combat ]
        #   SUBBAR:                [ *narration* | roll | meta ]
        # The active outer tab is bolded; the active subbar leaf bolds
        # in real time as emissions fire.
        # ====================================================================
        TAB_TOP_Y = 2.55      # outer tab vertical center
        SUB_Y    = 1.95       # subbar vertical center
        TAB_X    = 0.0        # whole indicator centered horizontally

        outer_label = Text("Many grammars", font=theme.SANS_FALLBACK,
                           font_size=14, color=Paper.ink_soft)
        outer_label.move_to(np.array([-3.4, TAB_TOP_Y, 0]))

        def _tab(name: str, active: bool, font_size: int = 14) -> VGroup:
            """A flat tab — bold when active, muted otherwise.

            We render the active variant with hair-space tracking so
            bolded text doesn't read cramped. Inactive tabs use the
            muted ink. The active tab also gets a thin underline.
            """
            color = Paper.accent if active else Paper.ink_soft
            label = (_spaced(name) if active else name)
            t = Text(
                label, font=theme.SANS_FALLBACK,
                weight=("BOLD" if active else "NORMAL"),
                font_size=font_size, color=color,
            )
            grp = VGroup(t)
            if active:
                ul = Line(
                    t.get_corner(np.array([-1, -1, 0])) + np.array([0, -0.06, 0]),
                    t.get_corner(np.array([1, -1, 0])) + np.array([0, -0.06, 0]),
                    stroke_color=Paper.accent, stroke_width=2.2,
                )
                grp.add(ul)
            return grp

        def _bracketed_tabs(names: list[str], active_idx: int,
                             font_size: int = 14) -> VGroup:
            """Render `[ a | b | c ]` with the `active_idx`-th name bolded."""
            lb = Text("[", font=theme.SANS_FALLBACK, font_size=font_size,
                     color=Paper.ink_soft)
            rb = Text("]", font=theme.SANS_FALLBACK, font_size=font_size,
                     color=Paper.ink_soft)
            sep_specs = []
            tab_mobs = []
            for i, n in enumerate(names):
                tab_mobs.append(_tab(n, active=(i == active_idx),
                                      font_size=font_size))
            row = VGroup(lb)
            for i, t in enumerate(tab_mobs):
                row.add(t)
                if i < len(tab_mobs) - 1:
                    sep = Text("|", font=theme.SANS_FALLBACK,
                               font_size=font_size, color=Paper.mute)
                    row.add(sep)
                    sep_specs.append(sep)
            row.add(rb)
            row.arrange(RIGHT, buff=0.18)
            return row

        # Outer tabs: [narrative | combat]; "narrative" is initially active.
        outer_tabs = _bracketed_tabs(["narrative", "combat"], active_idx=0)
        outer_tabs.next_to(outer_label, RIGHT, buff=0.3)

        # Subbar: starts as [narration | roll | meta], "narration" hot
        sub_names = ["narration", "roll", "meta"]
        subbar = _bracketed_tabs(sub_names, active_idx=0, font_size=12)
        # Drop subbar slightly to right-of-center under the outer tabs.
        subbar.move_to(np.array([
            outer_tabs.get_center()[0],
            SUB_Y, 0,
        ]))

        self.play(
            FadeIn(outer_label, run_time=0.35),
            FadeIn(outer_tabs, run_time=0.45),
        )
        self.play(FadeIn(subbar, shift=UP * 0.06, run_time=0.4))
        self.wait(0.4)

        # Helper: rebuild the subbar with a different active leaf, fading
        # the old subbar out and the new one in atomically. Cheap and
        # bulletproof — beats animating individual letters.
        subbar_state = {"current": subbar, "names": list(sub_names),
                        "active_idx": 0, "font_size": 12}

        def set_subbar(names: list[str] | None = None,
                        active_idx: int = 0,
                        font_size: int | None = None,
                        run_time: float = 0.35) -> VGroup:
            current = subbar_state["current"]
            new_names = names if names is not None else subbar_state["names"]
            fs = font_size if font_size is not None else subbar_state["font_size"]
            new_bar = _bracketed_tabs(new_names, active_idx=active_idx,
                                       font_size=fs)
            new_bar.move_to(np.array([
                outer_tabs.get_center()[0],
                SUB_Y, 0,
            ]))
            self.play(
                FadeOut(current, run_time=run_time * 0.5),
                FadeIn(new_bar, run_time=run_time),
            )
            subbar_state["current"] = new_bar
            subbar_state["names"] = new_names
            subbar_state["active_idx"] = active_idx
            subbar_state["font_size"] = fs
            return new_bar

        def set_outer(active_idx: int, run_time: float = 0.45) -> VGroup:
            new_outer = _bracketed_tabs(
                ["narrative", "combat"], active_idx=active_idx,
            )
            new_outer.next_to(outer_label, RIGHT, buff=0.3)
            self.play(
                FadeOut(outer_tabs[:], run_time=run_time * 0.5),
                FadeIn(new_outer, run_time=run_time),
            )
            return new_outer

        # ====================================================================
        # Beat 4.B — trace area. Anchored on the LEFT half of the frame
        # below the tab indicator — single text column, doesn't compete.
        # ====================================================================
        trace_x = -6.7
        trace_top_y = 1.2
        line_h = 0.42

        idx = [0]
        trace_items = VGroup()

        # Mono per-char width at the trace font size — used to honour
        # leading whitespace in emit() text (manim Text strips it).
        emit_char_w = _mono_char_width(14)

        def emit(text: str, color: str = Paper.ink, font_size: int = 14,
                 indent: float = 0.0, hold: float = 0.0,
                 run_time: float = 0.32) -> Text:
            # If `text` has leading spaces (visual indent), preserve them
            # by adding their pixel-width to `indent`. Otherwise manim
            # strips them and the continuation line slams left.
            n_leading = len(text) - len(text.lstrip(" "))
            stripped = text.lstrip(" ")
            extra_indent = n_leading * emit_char_w
            t = _code_text(stripped, color, font_size=font_size)
            t.align_to(np.array([trace_x + indent + extra_indent, 0, 0]), LEFT)
            t.shift(UP * (trace_top_y - idx[0] * line_h))
            self.play(FadeIn(t, shift=LEFT * 0.08, run_time=run_time))
            trace_items.add(t)
            idx[0] += 1
            if hold:
                self.wait(hold)
            return t

        # Narration emission — subbar shows "narration" hot.
        # (subbar already starts on narration, so no rebuild needed.)
        emit('@narrate("You try to convince the hooded figure',
             color=Paper.ink, hold=0.0)
        emit('         this is all a misunderstanding…")',
             color=Paper.ink, hold=1.6)

        # Roll emission — subbar bolds "roll".
        set_subbar(active_idx=1, run_time=0.3)
        emit('@roll("persuasion", dc=14)', color=Paper.ink)
        # Show the round-trip: client returns the resolved tool result.
        roll_arrow = Text("            ↓  client resolves",
                          font=theme.MONO_FALLBACK,
                          font_size=11, color=Paper.ink_soft)
        roll_arrow.align_to(np.array([trace_x + 0.4, 0, 0]), LEFT)
        roll_arrow.shift(UP * (trace_top_y - idx[0] * line_h))
        self.play(FadeIn(roll_arrow, run_time=0.35))
        idx[0] += 1
        trace_items.add(roll_arrow)

        roll_resolved = _code_text(
            "→ {d20: 1, success: false}",
            Paper.bad, font_size=13,
        )
        roll_resolved.align_to(np.array([trace_x + 0.4, 0, 0]), LEFT)
        roll_resolved.shift(UP * (trace_top_y - idx[0] * line_h))
        self.play(FadeIn(roll_resolved, shift=LEFT * 0.08, run_time=0.35))
        idx[0] += 1
        trace_items.add(roll_resolved)
        self.wait(1.2)

        # Meta emission — subbar bolds "meta".
        set_subbar(active_idx=2, run_time=0.3)
        emit('@meta("Haha — a 1. Sorry, won\'t cut it.")',
             color=Paper.accent_soft, hold=1.4)

        # Brief caption — anchored at bottom-right so it doesn't compete
        # with the trace stream above.
        meta_caption = Text(
            "@meta and @narrate are both string-typed —",
            font="Georgia", slant="ITALIC", font_size=14,
            color=Paper.ink_soft,
        )
        meta_caption2 = Text(
            "no XML tags, no post-parse. Just two tools.",
            font="Georgia", slant="ITALIC", font_size=14,
            color=Paper.ink_soft,
        )
        meta_grp = VGroup(meta_caption, meta_caption2).arrange(
            DOWN, aligned_edge=LEFT, buff=0.1,
        )
        meta_grp.to_edge(DOWN, buff=0.55)
        meta_grp.shift(RIGHT * 0.6)
        self.play(FadeIn(meta_grp, run_time=0.45))
        self.wait(2.4)
        self.play(FadeOut(meta_grp, run_time=0.35))

        # Back to narration — subbar bolds "narration".
        set_subbar(active_idx=0, run_time=0.3)
        emit('@narrate("\'My fist is going to make you',
             color=Paper.ink)
        emit('         miss understanding, punk.\'")',
             color=Paper.ink, hold=1.4)

        # @enter_combat — the regrammar moment.
        emit('@enter_combat(aria, borin, hooded_figure)',
             color=Paper.accent, hold=0.6)

        # ====================================================================
        # Beat 4.C — grammar reshapes for combat.
        # OUTER tab swaps narrative→combat; SUBBAR reshapes from
        #   [narration, roll, meta] → [aria_turn, borin_turn, hooded_figure_turn]
        # ====================================================================
        outer_tabs = set_outer(active_idx=1, run_time=0.5)
        set_subbar(
            names=["aria_turn", "borin_turn", "hooded_figure_turn"],
            active_idx=0, run_time=0.55,
        )
        switch_caption = Text(
            "↑  grammar swap on the same KV",
            font="Georgia", slant="ITALIC", font_size=14, color=Paper.accent,
        )
        switch_caption.next_to(subbar_state["current"], DOWN, buff=0.18)
        self.play(FadeIn(switch_caption, shift=UP * 0.06, run_time=0.45))
        self.wait(1.6)
        self.play(FadeOut(switch_caption, run_time=0.3))

        # A pair of in-combat emissions — subbar bolds the active NPC.
        # aria_turn (active_idx=0) is already hot.
        emit('@aria_turn(action="longsword",', color=Paper.ink)
        emit('           bonus_action="healing_word")',
             color=Paper.ink, hold=1.0)

        # hooded_figure's turn — subbar bolds index 2.
        set_subbar(active_idx=2, run_time=0.3)
        emit('@hooded_figure_turn(action="dagger",', color=Paper.ink)
        emit('                    target="aria")',
             color=Paper.ink, hold=1.4)

        # ====================================================================
        # Beat 4.D — Aria's turn. Pull aria_turn out to the LEFT, fold
        # the program definition out next to it. The other elements dim
        # so the source code is unambiguously the visual focus.
        # ====================================================================
        # Bring aria_turn back into focus on the subbar.
        set_subbar(
            names=["aria_turn", "borin_turn", "hooded_figure_turn"],
            active_idx=0, run_time=0.3,
        )

        # Clear the trace area; we re-anchor with the program reveal.
        self.play(FadeOut(trace_items, run_time=0.4))
        idx[0] = 0
        trace_items = VGroup()

        # Dim other elements so the program is the visual focus.
        self.play(
            outer_label.animate.set_opacity(0.35),
            outer_tabs.animate.set_opacity(0.35),
            title.animate.set_opacity(0.4),
            run_time=0.4,
        )

        # Animate aria_turn out to the LEFT of frame, scale up a touch.
        # The subbar leaf at active_idx=0 is the second child of subbar
        # (after lb). We pull a fresh, larger label out — easier than
        # animating the bracketed group.
        aria_focus = Text(
            _spaced("aria_turn"),
            font=theme.SANS_FALLBACK, weight="BOLD",
            font_size=22, color=Paper.accent,
        )
        # Start at the active subbar leaf position (best-effort) and
        # animate to a left-side anchor where the program will fold open.
        active_leaf = subbar_state["current"][2]  # [ , aria_turn, |, ...]
        aria_focus.move_to(active_leaf.get_center())
        self.add(aria_focus)
        self.play(
            subbar_state["current"].animate.set_opacity(0.3),
            aria_focus.animate.move_to(np.array([-5.0, 1.95, 0]))
                              .scale(1.05),
            run_time=0.7,
        )

        # Fold the program definition out — appears beneath the focus
        # label and slightly to its right, occupying the left half of
        # frame width-wise. Syntax-highlighted.
        anchor_aria = np.array([-7.0, 1.4, 0])
        ACC = Paper.accent
        KW  = Paper.accent_soft
        FN  = Terminal.amber
        ID  = Paper.ink
        DIM = Paper.ink_soft
        STR = Paper.good
        aria_rich = [
            [("@program", ACC)],
            [("def ", KW), ("aria_turn", FN), ("():", DIM)],
            [("    move = ", ID), ("yield ", KW),
             ("gen.struct", FN), ("(", DIM)],
            [("        action=", ID), ("gen.choice", FN), ("([", DIM)],
            [("            'longsword'", STR), (", ", DIM),
             ("'fireball'", STR), (",", DIM)],
            [("            'vicious_mockery'", STR), (", ", DIM),
             ("'hold'", STR), ("]),", DIM)],
            [("        bonus_action=", ID), ("gen.choice", FN), ("([", DIM)],
            [("            'dagger'", STR), (", ", DIM),
             ("'healing_word'", STR), (",", DIM)],
            [("            'thorn_whip'", STR), (", ", DIM),
             ("'hold'", STR), ("]),", DIM)],
            [("        where=", ACC), ("lambda ", KW),
             ("d", ID), (": ", DIM), ("not (", ACC)],
            [("            d['action'] in NON_CANTRIPS", ACC)],
            [("            and d['bonus_action'] in SPELLS", ACC)],
            [("        ),", ACC)],
            [("    )", DIM)],
            [("    return ", KW), ("move", ID)],
        ]
        aria_code = _rich_block(aria_rich, anchor=anchor_aria,
                                line_height=0.30, font_size=13)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in aria_code],
            lag_ratio=0.07, run_time=2.0,
        ))
        self.wait(0.9)

        # Highlight the where= block — this is THE moment.
        # Height is fitted to the lines [9..12] block (4 lines x 0.30
        # line_height) with a slim padding so we don't bite into the
        # bonus_action line above or `return move` below.
        where_box = RoundedRectangle(
            width=6.6, height=1.32, corner_radius=0.10,
            fill_color=Paper.accent, fill_opacity=0.12,
            stroke_color=Paper.accent, stroke_width=1.6,
        )
        center_y = (aria_code[9].get_center()[1]
                    + aria_code[12].get_center()[1]) / 2
        where_box.move_to(np.array([anchor_aria[0] + 3.2, center_y, 0]))
        # Caption goes to the RIGHT of the where_box (avoids overlapping
        # `return move` which is the line immediately below the box).
        where_label_grp = VGroup(
            Text("logic constraint,", font="Georgia", slant="ITALIC",
                 font_size=15, color=Paper.accent),
            Text("in Python.", font="Georgia", slant="ITALIC",
                 font_size=15, color=Paper.accent),
            Text("across fields.", font="Georgia", slant="ITALIC",
                 font_size=15, color=Paper.accent),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        where_label_grp.next_to(where_box, RIGHT, buff=0.3)
        self.play(FadeIn(where_box, run_time=0.5),
                  FadeIn(where_label_grp, run_time=0.5))
        self.wait(2.0)

        # Punchline pair on the right side — JSON Schema vs Python.
        self.play(FadeOut(where_label_grp, run_time=0.35))
        impossible = Text(
            _spaced("JSON Schema cannot"),
            font="Georgia", weight="BOLD", font_size=20, color=Paper.bad,
        )
        impossible2 = Text(
            _spaced("express this constraint."),
            font="Georgia", weight="BOLD", font_size=20, color=Paper.bad,
        )
        impossible_grp = VGroup(impossible, impossible2).arrange(
            DOWN, buff=0.08,
        )
        impossible_grp.move_to(np.array([4.4, 0.5, 0]))
        self.play(FadeIn(impossible_grp, shift=UP * 0.08, run_time=0.55))
        self.wait(1.8)
        ours = Text(
            _spaced("Predicates are Python."),
            font="Georgia", slant="ITALIC", weight="BOLD",
            font_size=22, color=Paper.accent,
        )
        ours.next_to(impossible_grp, DOWN, buff=0.4)
        self.play(FadeIn(ours, shift=UP * 0.06, run_time=0.5))
        self.wait(2.8)

        # Mention "fields can reference Python program state, too" — fast.
        bonus = Text(
            "Fields can reference any Python program state.",
            font="Georgia", slant="ITALIC", font_size=14,
            color=Paper.ink_soft,
        )
        bonus.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(bonus, shift=UP * 0.06, run_time=0.5))
        self.wait(2.8)

        # Clean.
        self.play(FadeOut(VGroup(
            title, aria_code, where_box, impossible_grp, ours, bonus,
            outer_label, outer_tabs, subbar_state["current"], aria_focus,
        ), run_time=0.5))

    # ======================================================================
    # PAGE 5 — model authors its own primitive → thesis
    # ======================================================================

    def _page5_meta_authorship_and_close(self):
        # Beat 5.A — bridge.
        bridge = Text(
            "This is already pretty nice. But we kept thinking.",
            font="Georgia", slant="ITALIC", font_size=22, color=Paper.ink,
        )
        bridge.move_to(UP * 0.5)
        self.play(FadeIn(bridge, shift=UP * 0.08, run_time=0.5))
        self.wait(2.4)

        question = Text(
            "What if the model defined its own schemas",
            font="Georgia", font_size=24, color=Paper.ink,
        )
        question2 = Text(
            "as structure on its own future generation?",
            font="Georgia", slant="ITALIC", font_size=24, color=Paper.accent,
        )
        question_grp = VGroup(question, question2).arrange(DOWN, buff=0.16)
        question_grp.move_to(DOWN * 0.4)
        self.play(FadeIn(question, shift=UP * 0.08, run_time=0.5))
        self.play(FadeIn(question2, shift=UP * 0.08, run_time=0.55))
        self.wait(3.4)
        self.play(FadeOut(VGroup(bridge, question_grp), run_time=0.45))

        # Beat 5.B — problem on screen, then meta-call, then source materialises.
        prob = Text("Solve:   x² − 5x + 6 = 0",
                    font=theme.MONO_FALLBACK, font_size=20,
                    color=Paper.ink_soft)
        prob.to_edge(UP, buff=0.5)
        self.play(FadeIn(prob, run_time=0.35))

        # The model emits @make_new_program first.
        emit_call = Text(
            '@make_new_program("quadratic_solver",',
            font=theme.MONO_FALLBACK, font_size=15, color=Paper.accent,
        )
        emit_call2 = Text(
            '                  "find roots of a quadratic")',
            font=theme.MONO_FALLBACK, font_size=15, color=Paper.accent,
        )
        emit_grp = VGroup(emit_call, emit_call2).arrange(
            DOWN, aligned_edge=LEFT, buff=0.08,
        )
        emit_grp.move_to(np.array([0, 2.0, 0]))
        self.play(FadeIn(emit_grp, shift=UP * 0.08, run_time=0.45))
        self.wait(0.6)

        switch_label = Text(
            "[grammar switch → PROGRAM_SOURCE_GRAMMAR]",
            font=theme.MONO_FALLBACK, font_size=12, color=Paper.ink_soft,
        )
        switch_label.next_to(emit_grp, DOWN, buff=0.18)
        self.play(FadeIn(switch_label, run_time=0.35))
        self.wait(0.5)

        # Source materialises — syntax-highlighted to match the rest.
        anchor_src = np.array([-3.4, 0.7, 0])
        ACC = Paper.accent
        KW  = Paper.accent_soft
        FN  = Terminal.amber
        ID  = Paper.ink
        DIM = Paper.ink_soft
        NUM = Terminal.blue
        STR = Paper.good
        source_rich = [
            [("@program", ACC)],
            [("def ", KW), ("quadratic_solver", FN), ("():", DIM)],
            [("    a     = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("-9", NUM), (", ", DIM), ("9", NUM), (")", DIM)],
            [("    b     = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("-9", NUM), (", ", DIM), ("9", NUM), (")", DIM)],
            [("    c     = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("-9", NUM), (", ", DIM), ("9", NUM), (")", DIM)],
            [("    root1 = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("-9", NUM), (", ", DIM), ("9", NUM), (")", DIM)],
            [("    root2 = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("-9", NUM), (", ", DIM), ("9", NUM), (")", DIM)],
            [("    return ", KW),
             ("{", DIM), ("'a'", STR), (": a, ", ID),
             ("'b'", STR), (": b, ", ID), ("'c'", STR), (": c,", ID)],
            [("            ", ID),
             ("'roots'", STR), (": [root1, root2]", ID), ("}", DIM)],
        ]
        source = _rich_block(source_rich, anchor=anchor_src,
                             line_height=0.30, font_size=13)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in source],
            lag_ratio=0.14, run_time=2.8,
        ))
        self.wait(2.0)

        # Compile callout — to the right of the source.
        compile_note = Text("[validated · compiled · registered]",
                            font=theme.MONO_FALLBACK, font_size=13,
                            color=Paper.good)
        compile_note.next_to(source, RIGHT, buff=0.6)
        compile_note.align_to(source, UP)
        self.play(FadeIn(compile_note, shift=LEFT * 0.06, run_time=0.4))
        self.wait(0.9)

        # The model uses what it just authored — also right of source.
        usage = Text(
            "@quadratic_solver(1, -5, 6, 2, 3)",
            font=theme.MONO_FALLBACK, font_size=14, color=Paper.accent,
        )
        usage.next_to(compile_note, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(usage, shift=UP * 0.08, run_time=0.4))

        done_t = Text("@done(\"x = 2 or x = 3\")  ✓",
                      font=theme.MONO_FALLBACK, font_size=14,
                      color=Paper.good)
        done_t.next_to(usage, DOWN, buff=0.16, aligned_edge=LEFT)
        self.play(FadeIn(done_t, shift=UP * 0.05, run_time=0.45))
        self.wait(3.4)

        # Capability footnote. Predicate-bound bodies (via `where=`) are
        # shipped as of commit 6473880 — PROGRAM_SOURCE_GRAMMAR admits
        # `where=<lib_predicate>(<bound_args>)` clauses, and the host
        # library at src/orate/meta_predicates.py exposes is_prime,
        # digit_sum_eq, lt, gt, equivalent_under, factors_to (13 unit
        # tests green). The earlier "on the roadmap" caveat is obsolete.
        footnote = Text(
            "shipped: predicate-bound bodies via where=",
            font="Georgia", slant="ITALIC", font_size=12,
            color=Paper.mute,
        )
        footnote.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(footnote, run_time=0.45))
        self.wait(3.0)

        # Clean for thesis card.
        self.play(FadeOut(VGroup(prob, emit_grp, switch_label, source,
                                  compile_note, usage, done_t, footnote),
                          run_time=0.45))

        # Beat 5.C — thesis card. Letter-tracked headline; mute setup
        # lines stay at default tracking.
        thesis = VGroup(
            Text("Structured output constrained the shape.",
                 font="Georgia", font_size=24, color=Paper.ink_soft),
            Text("Tool calling constrained the side effect.",
                 font="Georgia", font_size=24, color=Paper.ink_soft),
            Text(_spaced("orate lets the model enforce", n=1),
                 font="Georgia", weight="BOLD",
                 font_size=30, color=Paper.ink),
            Text(_spaced("the legality of its own thought.", n=1),
                 font="Georgia", slant="ITALIC", weight="BOLD",
                 font_size=30, color=Paper.accent),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        thesis.move_to(ORIGIN)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=UP * 0.12) for ln in thesis],
            lag_ratio=0.55, run_time=5.2,
        ))
        self.wait(5.5)

        gh = Text("github.com/maltelandgren/orate",
                  font=theme.MONO_FALLBACK, font_size=14,
                  color=Paper.ink_soft)
        gh.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(gh, run_time=0.55))
        self.wait(5.0)
