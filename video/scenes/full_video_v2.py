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
        # Training-time chips
        tt_label = Text("training time", font=theme.SANS_FALLBACK,
                        font_size=14, color=Paper.ink_soft)
        it_label = Text("inference time", font=theme.SANS_FALLBACK,
                        font_size=14, color=Paper.ink_soft)
        tt_label.move_to(np.array([-5.2, 0.5, 0]))
        it_label.move_to(np.array([0.5, 0.5, 0]))

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
        strike_ft = _strike_line(ft_chip, color=Paper.bad, width=2.4)
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

        prompts_punch = Text(
            "Hard to write programs around.",
            font="Georgia", font_size=22, color=Paper.ink,
        )
        prompts_punch.move_to(np.array([0, anchor_y - 0.55, 0]))
        self.play(FadeIn(prompts_punch, shift=UP * 0.08, run_time=0.4))
        self.wait(2.0)

        # structured output: bridge — same line transforms.
        self.play(*_unhighlight(prompts_chip),
                  *_highlight(so_chip),
                  FadeOut(prompts_line1, shift=UP * 0.05, run_time=0.3))

        # The bridge: "Hard to write programs around" → "Lets us write programs around".
        # Use cross-fade rather than Transform — Transform morphs glyph-by-glyph
        # and produces a glitchy intermediate frame.
        prompts_punch_target = Text(
            "Lets us write programs around.",
            font="Georgia", font_size=22, color=Paper.accent,
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
        schema_anchor = np.array([-6.55, 2.1, 0])
        schema_lines = [
            ('{', Paper.ink),
            ('  "name":     "Aria",', Paper.ink),
            ('  "level":    3,', Paper.ink),
            ('  "class":    "bard",', Paper.ink),
            ('  "alive":    true,', Paper.ink),
            ('}', Paper.ink),
        ]
        schema = _code_block(schema_lines, anchor=schema_anchor,
                             line_height=0.40, font_size=16)
        # We'll reveal the lines progressively + caption each field.

        self.play(FadeIn(schema[0], run_time=0.18))  # opening brace

        captions = []

        def reveal_field(idx: int, caption: str, color: str = Paper.accent_soft):
            """Reveal schema line idx, point a caption to its right."""
            self.play(FadeIn(schema[idx], shift=LEFT * 0.1, run_time=0.3))
            cap = Text(caption, font=theme.MONO_FALLBACK,
                       font_size=12, color=color)
            cap.next_to(schema[idx], RIGHT, buff=0.35)
            self.play(FadeIn(cap, shift=LEFT * 0.05, run_time=0.25))
            captions.append(cap)
            self.wait(0.25)

        reveal_field(1, "← string")
        reveal_field(2, "← integer")
        reveal_field(3, "← enum [bard, cleric, rogue]")
        reveal_field(4, "← boolean")
        self.play(FadeIn(schema[5], run_time=0.2))  # closing brace
        self.wait(1.0)

        # Beat 2.C — caption: where's the logic in that?
        bridge_q = Text(
            "Type lives in the decoder.   Where's the logic?",
            font="Georgia", slant="ITALIC", font_size=20,
            color=Paper.ink_soft,
        )
        bridge_q.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(bridge_q, shift=UP * 0.08, run_time=0.5))
        self.wait(2.4)

        # Fade captions out (but keep schema)
        self.play(FadeOut(VGroup(*captions), run_time=0.35))

        # Beat 2.D — "the logic" header drops top-right; mirrored drawer.
        logic_chip = _chip("the logic", font_size=16,
                           fg=Paper.accent, stroke=Paper.accent)
        logic_chip.move_to(np.array([3.8, 3.55, 0]))
        # Bridge_q transforms into the logic chip on the right.
        # Snap bridge_q first; then materialise the chip.
        self.play(FadeOut(bridge_q, shift=UP * 0.1, run_time=0.3))
        self.play(FadeIn(logic_chip, shift=DOWN * 0.15, run_time=0.4))

        right_drawer = _paper_card(width=6.4, height=4.4)
        right_drawer.move_to(np.array([3.55, 0.2, 0]))
        right_drawer.shift(UP * 0.6)
        self.play(
            right_drawer.animate.shift(DOWN * 0.6),
            FadeIn(right_drawer, run_time=0.45),
            run_time=0.45,
        )

        # Beat 2.E — draw the dm_turn @program in the right drawer.
        right_anchor = np.array([0.7, 2.1, 0])
        prog_lines = [
            ("@program", Paper.accent),
            ("def dm_turn(scene):", Paper.ink),
            ("    narration  = yield gen.string(...)", Paper.ink),
            ("    needs_roll = yield gen.boolean()", Paper.ink),
            ("    if needs_roll:", Paper.ink_soft),
            ("        dc     = yield gen.integer(5, 25)", Paper.ink),
            ("        result = yield gen.tool(", Paper.ink),
            ("                     roll_d20, dc=dc)", Paper.ink),
            ("    npc_line   = yield gen.string(...)", Paper.ink),
            ("    return {...}", Paper.ink),
        ]
        prog = _code_block(prog_lines, anchor=right_anchor,
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

        # Beat 3.B — algebra_step source on the left.
        anchor_l = np.array([-6.7, 0.9, 0])
        algebra_lines = [
            ("@program", Paper.accent),
            ("def algebra_step():", Paper.ink),
            ("    before = yield gen.string(...)", Paper.ink),
            ("    rule   = yield gen.choice([", Paper.ink),
            ("        'simplify', 'isolate_var',", Paper.ink_soft),
            ("        'evaluate', 'combine_like'])", Paper.ink_soft),
            ("    after  = yield gen.string(", Paper.ink),
            ("        where=lambda s:", Paper.accent),
            ("            equivalent_under(", Paper.accent),
            ("                rule, before, s))", Paper.accent),
            ("    return {...}", Paper.ink),
        ]
        algebra = _code_block(algebra_lines, anchor=anchor_l,
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

        # Constrained run beneath
        cons_label = Text("under @algebra_step", font=theme.SANS_FALLBACK,
                          font_size=12, color=Paper.accent)
        cons_label.move_to(np.array([1.5, anchor_r_top - 2.5, 0]))
        cons_lines = [
            ('@algebra_step("3x+5=14",', Paper.ink),
            ('  simplify, "3x=9")  ✓', Paper.good),
            ('@algebra_step("3x=9",', Paper.ink),
            ('  isolate_var, "x=3")  ✓', Paper.good),
            ('@done("x = 3")  ✓', Paper.accent),
        ]
        cons_block = VGroup()
        for i, (t, c) in enumerate(cons_lines):
            ln = _code_text(t, c, font_size=13)
            ln.move_to(np.array([3.6, anchor_r_top - 2.92 - 0.34 * i, 0]))
            ln.align_to(np.array([1.5, 0, 0]), LEFT)
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
        # Beat 4.A — page header + grammar indicator (the visual workhorse).
        title = Text(
            "One KV. Many grammars.",
            font="Georgia", slant="ITALIC", font_size=26, color=Paper.ink,
        )
        title.to_edge(UP, buff=0.45)
        self.play(FadeIn(title, shift=UP * 0.08, run_time=0.4))

        # Grammar indicator panel (right edge). It lists active leaf-tools.
        # Horizontal pill row inside a slim frame.
        gi_frame = RoundedRectangle(
            width=6.6, height=0.7, corner_radius=0.16,
            fill_color=Paper.card, fill_opacity=1.0,
            stroke_color=Paper.grid, stroke_width=1.0,
        )
        gi_frame.move_to(np.array([2.4, 2.55, 0]))
        gi_label = Text("active grammar", font=theme.SANS_FALLBACK,
                        font_size=11, color=Paper.ink_soft)
        gi_label.next_to(gi_frame, UP, buff=0.06, aligned_edge=LEFT)

        narrative_chips = VGroup(
            _chip("@narrate", font_size=11),
            _chip("@roll", font_size=11),
            _chip("@meta", font_size=11),
            _chip("@enter_combat", font_size=11),
        ).arrange(RIGHT, buff=0.16)
        narrative_chips.move_to(gi_frame.get_center())

        self.play(
            FadeIn(gi_frame, run_time=0.3),
            FadeIn(gi_label, run_time=0.3),
            FadeIn(narrative_chips, run_time=0.4),
        )
        self.wait(0.4)

        # Beat 4.B — trace area on the left.
        trace_x = -6.7
        trace_top_y = 1.6
        line_h = 0.42

        idx = [0]
        trace_items = VGroup()

        def emit(text: str, color: str = Paper.ink, font_size: int = 14,
                 indent: float = 0.0, hold: float = 0.0,
                 run_time: float = 0.32) -> Text:
            t = _code_text(text, color, font_size=font_size)
            t.align_to(np.array([trace_x + indent, 0, 0]), LEFT)
            t.shift(UP * (trace_top_y - idx[0] * line_h))
            self.play(FadeIn(t, shift=LEFT * 0.08, run_time=run_time))
            trace_items.add(t)
            idx[0] += 1
            if hold:
                self.wait(hold)
            return t

        def highlight_grammar(chip_idx: int):
            """Visually pop one grammar chip to indicate it's the active emission."""
            chip = narrative_chips[chip_idx]
            return [chip.animate.set_stroke(Paper.accent, width=1.8)]

        def unhighlight_grammar(chip_idx: int):
            chip = narrative_chips[chip_idx]
            return [chip.animate.set_stroke(Paper.grid, width=1.0)]

        # Narration emission
        self.play(*highlight_grammar(0), run_time=0.3)
        emit('@narrate("You try to convince the hooded figure',
             color=Paper.ink, hold=0.0)
        emit('         this is all a misunderstanding…")',
             color=Paper.ink, hold=2.0)
        self.play(*unhighlight_grammar(0), run_time=0.3)

        # Roll emission
        self.play(*highlight_grammar(1), run_time=0.25)
        emit('@roll("persuasion", dc=14)',
             color=Paper.ink)
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
        self.wait(1.4)
        self.play(*unhighlight_grammar(1), run_time=0.25)

        # Meta emission — out-of-character commentary
        self.play(*highlight_grammar(2), run_time=0.3)
        emit('@meta("Haha — a 1. Sorry, won\'t cut it.")',
             color=Paper.accent_soft, hold=1.6)
        self.play(*unhighlight_grammar(2), run_time=0.3)

        # Quick caption: same shape, different purpose.
        meta_caption = Text(
            "@meta and @narrate are both string-typed —",
            font="Georgia", slant="ITALIC", font_size=15,
            color=Paper.ink_soft,
        )
        meta_caption2 = Text(
            "no XML tags, no post-parse. Just two tools.",
            font="Georgia", slant="ITALIC", font_size=15,
            color=Paper.ink_soft,
        )
        meta_caption.move_to(np.array([3.0, -1.6, 0]))
        meta_caption2.next_to(meta_caption, DOWN, buff=0.12)
        self.play(FadeIn(meta_caption, run_time=0.45),
                  FadeIn(meta_caption2, run_time=0.45))
        self.wait(3.0)
        self.play(FadeOut(meta_caption, run_time=0.35),
                  FadeOut(meta_caption2, run_time=0.35))

        # Back to narrate
        self.play(*highlight_grammar(0), run_time=0.3)
        emit('@narrate("\'My fist is going to make you',
             color=Paper.ink)
        emit('         miss understanding, punk.\'")',
             color=Paper.ink, hold=1.8)
        self.play(*unhighlight_grammar(0), run_time=0.3)

        # @enter_combat
        self.play(*highlight_grammar(3), run_time=0.3)
        emit('@enter_combat(aria, borin, hooded_figure)',
             color=Paper.accent, hold=1.2)

        # ----- Beat 4.C — grammar reshapes for combat. -----
        # Build the new chips, position over the same frame; transform.
        combat_chips = VGroup(
            _chip("@aria_turn", font_size=11),
            _chip("@borin_turn", font_size=11),
            _chip("@hooded_figure_turn", font_size=11),
            _chip("@exit_combat", font_size=11),
        ).arrange(RIGHT, buff=0.16)
        combat_chips.move_to(gi_frame.get_center())

        # The transition itself is the content.
        self.play(
            FadeOut(narrative_chips, shift=UP * 0.2, run_time=0.5),
            FadeIn(combat_chips, shift=DOWN * 0.2, run_time=0.6),
        )
        switch_caption = Text(
            "↑  grammar swap on the same KV",
            font="Georgia", slant="ITALIC", font_size=14, color=Paper.accent,
        )
        switch_caption.next_to(gi_frame, DOWN, buff=0.18)
        self.play(FadeIn(switch_caption, shift=UP * 0.06, run_time=0.45))
        self.wait(2.0)
        self.play(FadeOut(switch_caption, run_time=0.3))

        # A pair of in-combat emissions so the regrammar feels real.
        self.play(*[combat_chips[0].animate.set_stroke(Paper.accent, width=1.8)],
                  run_time=0.3)
        emit('@aria_turn(action="longsword",',
             color=Paper.ink)
        emit('           bonus_action="healing_word")',
             color=Paper.ink, hold=1.4)
        self.play(*[combat_chips[0].animate.set_stroke(Paper.grid, width=1.0)],
                  run_time=0.25)

        self.play(*[combat_chips[2].animate.set_stroke(Paper.accent, width=1.8)],
                  run_time=0.3)
        emit('@hooded_figure_turn(action="dagger",',
             color=Paper.ink)
        emit('                    target="aria")',
             color=Paper.ink, hold=1.6)
        self.play(*[combat_chips[2].animate.set_stroke(Paper.grid, width=1.0)],
                  run_time=0.25)

        # ----- Beat 4.D — Aria's turn with the cross-field where=. -----
        # Clear the trace area to make room for the load-bearing source code.
        self.play(FadeOut(trace_items, run_time=0.35))
        idx[0] = 0
        trace_items = VGroup()

        # Restart the trace area with a small heading
        aria_head = Text(
            "Aria's @program — action + bonus_action",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink,
        )
        aria_head.move_to(np.array([-3.6, 1.85, 0]))
        self.play(FadeIn(aria_head, shift=UP * 0.08, run_time=0.35))

        anchor_aria = np.array([-7.0, 1.4, 0])
        aria_lines = [
            ("@program", Paper.accent),
            ("def aria_turn():", Paper.ink),
            ("    move = yield gen.struct(", Paper.ink),
            ("        action=gen.choice([", Paper.ink),
            ("            'longsword', 'fireball',", Paper.ink_soft),
            ("            'vicious_mockery', 'hold']),", Paper.ink_soft),
            ("        bonus_action=gen.choice([", Paper.ink),
            ("            'dagger', 'healing_word',", Paper.ink_soft),
            ("            'thorn_whip', 'hold']),", Paper.ink_soft),
            ("        where=lambda d: not (", Paper.accent),
            ("            d['action'] in NON_CANTRIPS", Paper.accent),
            ("            and d['bonus_action'] in SPELLS", Paper.accent),
            ("        ),", Paper.accent),
            ("    )", Paper.ink),
            ("    return move", Paper.ink),
        ]
        aria_code = _code_block(aria_lines, anchor=anchor_aria,
                                line_height=0.30, font_size=13)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in aria_code],
            lag_ratio=0.08, run_time=2.2,
        ))
        self.wait(1.2)

        # Highlight the where= block — this is THE moment.
        where_box = RoundedRectangle(
            width=6.4, height=1.6, corner_radius=0.10,
            fill_color=Paper.accent, fill_opacity=0.12,
            stroke_color=Paper.accent, stroke_width=1.6,
        )
        center_y = (aria_code[9].get_center()[1]
                    + aria_code[12].get_center()[1]) / 2
        where_box.move_to(np.array([anchor_aria[0] + 3.1, center_y, 0]))
        # Caption goes to the RIGHT of the where_box (avoids overlapping
        # `return move` which is the line immediately below the box).
        where_label = Text(
            "logic constraint,",
            font="Georgia", slant="ITALIC",
            font_size=15, color=Paper.accent,
        )
        where_label2 = Text(
            "in Python.",
            font="Georgia", slant="ITALIC",
            font_size=15, color=Paper.accent,
        )
        where_label3 = Text(
            "across fields.",
            font="Georgia", slant="ITALIC",
            font_size=15, color=Paper.accent,
        )
        where_label_grp = VGroup(where_label, where_label2,
                                 where_label3).arrange(
            DOWN, aligned_edge=LEFT, buff=0.06,
        )
        where_label_grp.next_to(where_box, RIGHT, buff=0.3)
        self.play(FadeIn(where_box, run_time=0.5),
                  FadeIn(where_label_grp, run_time=0.5))
        self.wait(2.2)

        # Fade the inline where_label, swap to the punchline pair.
        self.play(FadeOut(where_label_grp, run_time=0.35))

        # Right-side caption: JSON Schema cannot say this.
        impossible = Text(
            "JSON Schema cannot",
            font="Georgia", font_size=20, color=Paper.bad,
        )
        impossible2 = Text(
            "express this constraint.",
            font="Georgia", font_size=20, color=Paper.bad,
        )
        impossible_grp = VGroup(impossible, impossible2).arrange(
            DOWN, buff=0.08,
        )
        impossible_grp.move_to(np.array([4.4, 0.5, 0]))
        self.play(FadeIn(impossible_grp, shift=UP * 0.08, run_time=0.55))
        self.wait(2.0)
        # And the Python equivalent statement
        ours = Text(
            "Predicates are Python.",
            font="Georgia", slant="ITALIC", font_size=22,
            color=Paper.accent,
        )
        ours.next_to(impossible_grp, DOWN, buff=0.4)
        self.play(FadeIn(ours, shift=UP * 0.06, run_time=0.5))
        self.wait(3.2)

        # Mention "fields can reference Python program state, too" — fast.
        bonus = Text(
            "Fields can reference any Python program state.",
            font="Georgia", slant="ITALIC", font_size=14,
            color=Paper.ink_soft,
        )
        bonus.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(bonus, shift=UP * 0.06, run_time=0.5))
        self.wait(3.6)

        # Clean.
        self.play(FadeOut(VGroup(title, aria_head, aria_code, where_box,
                                  impossible_grp, ours,
                                  bonus, gi_frame, gi_label,
                                  combat_chips), run_time=0.5))

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

        # Source materialises.
        anchor_src = np.array([-3.4, 0.7, 0])
        source_lines = [
            ("@program", Paper.accent),
            ("def quadratic_solver():", Paper.ink),
            ("    a     = yield gen.integer(-9, 9)", Paper.ink),
            ("    b     = yield gen.integer(-9, 9)", Paper.ink),
            ("    c     = yield gen.integer(-9, 9)", Paper.ink),
            ("    root1 = yield gen.integer(-9, 9)", Paper.ink),
            ("    root2 = yield gen.integer(-9, 9)", Paper.ink),
            ("    return {'a': a, 'b': b, 'c': c,", Paper.ink),
            ("            'roots': [root1, root2]}", Paper.ink),
        ]
        source = _code_block(source_lines, anchor=anchor_src,
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

        # Honest-scope footnote (small, brief)
        footnote = Text(
            "(today: typed schema. predicate-bound bodies on the roadmap.)",
            font="Georgia", slant="ITALIC", font_size=11,
            color=Paper.mute,
        )
        footnote.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(footnote, run_time=0.45))
        self.wait(3.0)

        # Clean for thesis card.
        self.play(FadeOut(VGroup(prob, emit_grp, switch_label, source,
                                  compile_note, usage, done_t, footnote),
                          run_time=0.45))

        # Beat 5.C — thesis card.
        thesis = VGroup(
            Text("Structured output constrained the shape.",
                 font="Georgia", font_size=24, color=Paper.ink_soft),
            Text("Tool calling constrained the side effect.",
                 font="Georgia", font_size=24, color=Paper.ink_soft),
            Text("orate lets the model enforce",
                 font="Georgia", font_size=30, color=Paper.ink),
            Text("the legality of its own thought.",
                 font="Georgia", slant="ITALIC", font_size=30,
                 color=Paper.accent),
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
