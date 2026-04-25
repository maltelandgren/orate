"""
Full 3:00 hackathon submission — orate.

Pacing target (D1 paper aesthetic per user direction):

  Act 1 — protagonist-first cold open                 0:00–0:25
  Act 2 — the @program primitive                      0:25–0:55
  Act 3 — D&D session: mode-switch + composed NPCs    0:55–1:55
  Act 4 — legal steps:
            Beat 1 (algebra contrast)                 1:55–2:25
            Beat 2 (logic)                            2:25–2:40
            Beat 3 (meta finisher)                    2:40–2:55
  Close — thesis card                                 2:55–3:00

The aesthetic is D1 paper throughout, with terminal-dark insets only
where literal source code is the visual. The technical-accuracy
correction lands in Act 1: structured output / tool calling
constrain *type*, not *logic*.

Run:
    cd video/scenes
    manim -qm --format mp4 --media_dir ../renders full_video.py FullVideo
    manim -qh --format mp4 --media_dir ../renders full_video.py FullVideo
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


def _chip(label: str) -> VGroup:
    t = Text(label, font=theme.SANS_FALLBACK, font_size=16, color=Paper.ink)
    pad_x, pad_y = 0.3, 0.12
    pill = RoundedRectangle(
        width=t.width + pad_x * 2, height=t.height + pad_y * 2,
        corner_radius=0.22,
        fill_color=Paper.card, fill_opacity=1.0,
        stroke_color=Paper.grid, stroke_width=1.0,
    )
    pill.move_to(t.get_center())
    return VGroup(pill, t)


def _terminal_inset(width: float = 13.0, height: float = 7.0) -> tuple[VGroup, RoundedRectangle, Text]:
    """Returns (shadow, body, tag). Caller adds in order."""
    shadow = VGroup(*[
        RoundedRectangle(
            width=width + pad, height=height + pad,
            corner_radius=0.22 + pad / 2,
            fill_color=BLACK, fill_opacity=op, stroke_opacity=0,
        ).shift([0, dy, 0])
        for pad, op, dy in [(0.24, 0.02, -0.34), (0.15, 0.035, -0.22), (0.07, 0.05, -0.13)]
    ])
    body = RoundedRectangle(
        width=width, height=height, corner_radius=0.22,
        fill_color=Terminal.bg, fill_opacity=1.0,
        stroke_color=Paper.ink_soft, stroke_width=1.4,
    )
    tag = Text("  orate · session", font=theme.MONO_FALLBACK,
               font_size=15, color=Terminal.ink_soft)
    tag.next_to(body, UP, buff=0.1, aligned_edge=LEFT)
    return shadow, body, tag


def _code_lines(scene_obj: Scene, lines: list[tuple[str, str]],
                anchor_x: float = -5.3, top_y: float = 1.6,
                line_height: float = 0.4, font_size: int = 17) -> VGroup:
    code = VGroup()
    for i, (text, color) in enumerate(lines):
        ln = Text(text, font=theme.MONO_FALLBACK, font_size=font_size, color=color)
        ln.align_to(np.array([anchor_x, 0, 0]), LEFT)
        ln.shift(DOWN * (i * line_height) + UP * top_y)
        code.add(ln)
    return code


# ============================================================================
# The video
# ============================================================================


class FullVideo(Scene):
    def construct(self):
        self.camera.background_color = Paper.bg
        self.add(theme.paper_grid(opacity=0.35))

        self._act1_protagonist_open()      # 0:00 – 0:25
        self._act2_the_primitive()          # 0:25 – 0:55
        self._act3_dnd_session()            # 0:55 – 1:55
        self._act4_beat1_algebra_contrast()  # 1:55 – 2:25
        self._act4_beat2_logic()             # 2:25 – 2:40
        self._act4_beat3_meta_finisher()     # 2:40 – 2:55
        self._close_thesis()                 # 2:55 – 3:00

    # ----------------------------------------------------------------------
    # ACT 1 — protagonist-first cold open (0:00 – 0:25)
    # ----------------------------------------------------------------------

    def _act1_protagonist_open(self):
        # Beat 1.1: the LLM appears, emits, opens its logits, picks one. (0–8s)
        llm = LLMProtagonist(palette="paper", label="language model",
                             width=6.0, height=2.2)
        llm.move_to(ORIGIN)
        self.play(FadeIn(llm, shift=UP * 0.25, run_time=0.6))

        sub1 = Text(
            "Every token is a sample from a distribution.",
            font="Georgia", slant="ITALIC", font_size=22,
            color=Paper.ink_soft,
        )
        sub1.to_edge(DOWN, buff=0.9)
        self.play(FadeIn(sub1, shift=UP * 0.1, run_time=0.5))
        self.wait(1.2)

        llm.stream_tokens(self, ["the ", "cat ", "sat ", "on ", "the "], speed=0.18)
        llm.open_logits(
            self,
            [
                LogitItem("mat", 0.42),
                LogitItem("rug", 0.24),
                LogitItem("porch", 0.16),
                LogitItem("moon", 0.10),
                LogitItem("???", 0.08),
            ],
            column_width=2.2, gap=0.4,
        )
        self.wait(1.5)
        llm.choose_logit(self, 0)
        self.wait(0.8)
        llm.close_logits(self)
        self.wait(1.0)
        self.play(FadeOut(sub1), FadeOut(llm), run_time=0.4)

        # Beat 1.2: chips of what we've shaped distributions with. (8–14s)
        sub2 = Text(
            "Three years in, we're good at shaping distributions.",
            font="Georgia", slant="ITALIC", font_size=24, color=Paper.ink,
        )
        sub2.shift(UP * 2.4)
        self.play(FadeIn(sub2, shift=UP * 0.1, run_time=0.5))
        self.wait(1.0)

        chips = VGroup(*[
            _chip(t) for t in ["prompts", "fine-tuning", "structured output", "tool calls"]
        ]).arrange(RIGHT, buff=0.45)
        chips.move_to(UP * 1.0)
        self.play(LaggedStart(
            *[FadeIn(c, shift=UP * 0.1) for c in chips],
            lag_ratio=0.3, run_time=2.0,
        ))
        self.wait(2.5)

        # Beat 1.3: the technical accuracy correction. Type vs logic. (14–22s)
        sub3 = Text(
            "Type, not logic.",
            font="Georgia", slant="ITALIC", font_size=28, color=Paper.ink,
        )
        sub3.shift(DOWN * 0.4)
        self.play(FadeIn(sub3, shift=UP * 0.1, run_time=0.5))
        self.wait(2.5)

        sub4 = Text(
            "Logic has always lived outside the decoder.",
            font="Georgia", slant="ITALIC", font_size=22, color=Paper.ink_soft,
        )
        sub4.next_to(sub3, DOWN, buff=0.35)
        self.play(FadeIn(sub4, shift=UP * 0.05, run_time=0.5))
        self.wait(4.0)

        # Beat 1.4: the hook. (22–25s)
        self.play(FadeOut(VGroup(sub2, chips, sub3, sub4), run_time=0.4))

        hook = Text(
            "orate puts logic inside the decoder.",
            font="Georgia", font_size=32, color=Paper.accent,
        )
        hook.move_to(ORIGIN)
        self.play(FadeIn(hook, shift=UP * 0.1, run_time=0.6))
        self.wait(4.5)
        self.play(FadeOut(hook, run_time=0.4))

    # ----------------------------------------------------------------------
    # ACT 2 — the @program primitive (0:25 – 0:55)
    # ----------------------------------------------------------------------

    def _act2_the_primitive(self):
        title = Text(
            "One primitive. yield gen.X.",
            font="Georgia", slant="ITALIC", font_size=28, color=Paper.ink,
        )
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=UP * 0.1, run_time=0.5))

        # Terminal inset for the source code
        shadow, body, tag = _terminal_inset(width=12.5, height=5.5)
        shadow.shift(DOWN * 0.4)
        body.shift(DOWN * 0.4)
        tag.next_to(body, UP, buff=0.1, aligned_edge=LEFT)

        self.play(
            FadeIn(shadow, run_time=0.4),
            FadeIn(body, scale=0.94, run_time=0.4),
            FadeIn(tag, run_time=0.3),
        )

        code_lines: list[tuple[str, str]] = [
            ("@program", Terminal.accent),
            ("def dm_turn(scene):", Terminal.ink),
            ("    narration   = yield gen.string(...)", Terminal.ink),
            ("    needs_roll  = yield gen.boolean()", Terminal.ink),
            ("    if needs_roll:", Terminal.amber),
            ("        dc      = yield gen.integer(5, 25)", Terminal.ink),
            ("        result  = yield gen.tool(roll_d20, dc=dc)", Terminal.ink),
            ("    npc_line    = yield gen.string(...)", Terminal.ink),
            ("    return {...}", Terminal.ink),
        ]
        code = _code_lines(self, code_lines, anchor_x=-5.3, top_y=1.4, line_height=0.42)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.1) for ln in code],
            lag_ratio=0.10, run_time=2.4,
        ))
        self.wait(2.0)

        sub = Text(
            "Types, tool calls, control flow — same yield stream.",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink_soft,
        )
        sub.next_to(body, DOWN, buff=0.2)
        self.play(FadeIn(sub, shift=UP * 0.05, run_time=0.5))
        self.wait(6.0)

        # The where= aside — the logic constraint that doesn't fit JSON Schema
        self.play(FadeOut(code, run_time=0.4))

        where_lines: list[tuple[str, str]] = [
            ("# the predicate is the gate", Terminal.ink_soft),
            ("@program", Terminal.accent),
            ("def algebra_step():", Terminal.ink),
            ("    rule  = yield gen.choice(", Terminal.ink),
            ("        [\"isolate_var\", \"simplify\", ...])", Terminal.amber),
            ("    after = yield gen.string(", Terminal.ink),
            ("        where=lambda s:", Terminal.amber),
            ("            equivalent_under(rule, before, s))", Terminal.amber),
            ("    return after", Terminal.ink),
        ]
        where_code = _code_lines(self, where_lines, anchor_x=-5.3, top_y=1.4,
                                  line_height=0.42)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.1) for ln in where_code],
            lag_ratio=0.08, run_time=2.0,
        ))
        self.wait(1.5)

        # Highlight the where= predicate
        where_box = RoundedRectangle(
            width=8.5, height=1.4, corner_radius=0.1,
            fill_color=Terminal.accent, fill_opacity=0.12,
            stroke_color=Terminal.accent, stroke_width=1.4,
        )
        where_box.move_to(where_code[6].get_center() + DOWN * 0.2)
        where_label = Text("logic constraint, in Python",
                            font=theme.SANS_FALLBACK,
                            font_size=14, color=Terminal.accent)
        where_label.next_to(where_box, DOWN, buff=0.1, aligned_edge=LEFT)
        self.play(FadeIn(where_box, run_time=0.3),
                  FadeIn(where_label, run_time=0.3))
        self.wait(11.0)

        self.play(FadeOut(VGroup(title, shadow, body, tag, sub,
                                   where_code, where_box, where_label),
                          run_time=0.5))

    # ----------------------------------------------------------------------
    # ACT 3 — D&D session: mode-switch + composed NPCs (0:55 – 1:55)
    # ----------------------------------------------------------------------

    def _act3_dnd_session(self):
        title = Text(
            "One KV. Many grammars.",
            font="Georgia", slant="ITALIC", font_size=28, color=Paper.ink,
        )
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=UP * 0.1, run_time=0.5))

        # Active grammar pill (right-side panel)
        grammar_label = Text("active grammar:", font=theme.SANS_FALLBACK,
                             font_size=14, color=Paper.ink_soft)
        grammar_label.move_to(np.array([4.5, 2.7, 0]))
        grammar_text = Text(
            "narrative",
            font=theme.MONO_FALLBACK, font_size=18, color=Paper.accent,
        )
        grammar_text.next_to(grammar_label, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(FadeIn(grammar_label, run_time=0.3),
                  FadeIn(grammar_text, run_time=0.3))
        self.wait(1.0)

        # Trace area on the left — collect everything into one VGroup
        # so cleanup at the end of Act 3 is a single FadeOut.
        trace_x = -5.0
        trace_top_y = 2.2
        trace_items = VGroup()
        idx = [0]

        def emit_at(line: str, color: str = Paper.ink) -> Text:
            t = Text(line, font=theme.MONO_FALLBACK, font_size=18, color=color)
            t.align_to(np.array([trace_x, 0, 0]), LEFT)
            t.shift(UP * (trace_top_y - idx[0] * 0.5))
            self.play(FadeIn(t, shift=LEFT * 0.1, run_time=0.32))
            idx[0] += 1
            trace_items.add(t)
            return t

        # Beat 3.1: roll
        narr_placeholder = Text(
            '[narration: "The tavern is dim and smells of woodsmoke."]',
            font=theme.MONO_FALLBACK, font_size=14, color=Paper.ink_soft,
        )
        narr_placeholder.align_to(np.array([trace_x, 0, 0]), LEFT)
        narr_placeholder.shift(UP * (trace_top_y - idx[0] * 0.5))
        self.play(FadeIn(narr_placeholder, run_time=0.35))
        idx[0] += 1
        trace_items.add(narr_placeholder)

        emit_at("@roll(perception, 13)", color=Paper.ink)
        roll_result = Text("→ {d20: 17, success: true}",
                           font=theme.MONO_FALLBACK, font_size=15,
                           color=Paper.good)
        roll_result.align_to(np.array([trace_x + 0.4, 0, 0]), LEFT)
        roll_result.shift(UP * (trace_top_y - idx[0] * 0.5))
        self.play(FadeIn(roll_result, shift=LEFT * 0.1, run_time=0.4))
        idx[0] += 1
        trace_items.add(roll_result)
        self.wait(4.0)

        # Beat 3.2: enter combat — mode switch animation
        emit_at("@enter_combat(hooded_figure)", color=Paper.ink)
        self.wait(0.4)
        # Animate grammar text swapping
        new_grammar = Text(
            "combat",
            font=theme.MONO_FALLBACK, font_size=18, color=Paper.accent,
        )
        new_grammar.move_to(grammar_text.get_center())
        self.play(FadeOut(grammar_text, shift=UP * 0.1, run_time=0.5),
                  FadeIn(new_grammar, shift=UP * 0.1, run_time=0.6))
        grammar_text = new_grammar
        self.wait(1.5)

        # Hold so viewer reads the roll result before the mode switch lands
        self.wait(2.5)

        # Cut-away: flash the three character programs side-by-side
        chars = VGroup()
        char_specs = [
            ("@program\ndef aria_attack():\n  action = yield gen.choice([\n    \"longsword\",\n    \"vicious_mockery\",\n    \"hold\"])\n  target = yield gen.choice(...)\n  damage = yield gen.integer(0, 6)\n  return {...}", "aria"),
            ("@program\ndef hooded_figure_attack():\n  action = yield gen.choice([\n    \"dagger\",\n    \"shadow_step\",\n    \"retreat\"])\n  target = yield gen.choice(...)\n  damage = yield gen.integer(0, 4)\n  return {...}", "hooded"),
            ("@program\ndef borin_attack():\n  action = yield gen.choice([\n    \"warhammer\",\n    \"shield_bash\",\n    \"intimidate\"])\n  target = yield gen.choice(...)\n  damage = yield gen.integer(0, 8)\n  return {...}", "borin"),
        ]
        x_offsets = [-4.8, 0.0, 4.8]
        for src, _ in char_specs:
            txt = Text(src, font=theme.MONO_FALLBACK, font_size=11,
                       color=Paper.ink, line_spacing=0.7)
            chars.add(txt)
        for i, txt in enumerate(chars):
            txt.move_to(np.array([x_offsets[i], -1.6, 0]))

        self.play(LaggedStart(
            *[FadeIn(c, shift=UP * 0.15) for c in chars],
            lag_ratio=0.18, run_time=1.4,
        ))

        # Caption: this is all it takes
        cap = Text("each character's grammar IS its stat sheet",
                   font="Georgia", slant="ITALIC", font_size=18,
                   color=Paper.ink_soft)
        cap.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap, shift=UP * 0.08, run_time=0.45))
        self.wait(11.0)

        # Trace continues with NPC turns
        self.play(FadeOut(chars, run_time=0.4),
                  FadeOut(cap, run_time=0.3))

        emit_at("@hooded_figure_attack(dagger, aria, 3)", color=Paper.ink)
        self.wait(2.0)
        emit_at("@aria_attack(longsword, hooded_figure, 5)", color=Paper.ink)
        self.wait(2.0)
        emit_at("@borin_attack(warhammer, hooded_figure, 6)", color=Paper.ink)
        self.wait(2.5)

        # Exit combat — mode switch back
        emit_at("@exit_combat(victory)", color=Paper.ink)
        new_grammar_2 = Text("narrative",
                             font=theme.MONO_FALLBACK, font_size=18,
                             color=Paper.accent)
        new_grammar_2.move_to(grammar_text.get_center())
        self.play(FadeOut(grammar_text, shift=DOWN * 0.1, run_time=0.25),
                  FadeIn(new_grammar_2, shift=DOWN * 0.1, run_time=0.3))
        grammar_text = new_grammar_2
        self.wait(1.0)

        # Punchline
        kv_note = Text(
            "One inference. One mode switch. Three composed-per-character grammars.",
            font="Georgia", slant="ITALIC", font_size=18, color=Paper.ink,
        )
        kv_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(kv_note, shift=UP * 0.1, run_time=0.5))
        self.wait(11.0)

        # Clean up Act 3 — single FadeOut over everything we added
        self.play(FadeOut(VGroup(title, grammar_label, grammar_text,
                                   trace_items, kv_note),
                          run_time=0.5))

    # ----------------------------------------------------------------------
    # ACT 4 BEAT 1 — algebra contrast (1:55 – 2:25)
    # ----------------------------------------------------------------------

    def _act4_beat1_algebra_contrast(self):
        # Setup: same model, same problem, two runs
        hdr = Text("Same model. Same prompt. Two runs.",
                   font="Georgia", slant="ITALIC",
                   font_size=26, color=Paper.ink)
        hdr.to_edge(UP, buff=0.6)
        self.play(FadeIn(hdr, shift=UP * 0.1, run_time=0.45))

        prob = Text("Solve for x:    3x + 5 = 14",
                    font=theme.MONO_FALLBACK, font_size=20,
                    color=Paper.ink_soft)
        prob.next_to(hdr, DOWN, buff=0.4)
        self.play(FadeIn(prob, run_time=0.35))

        a = LLMProtagonist(palette="paper", label="free text",
                           width=5.4, height=2.4)
        b = LLMProtagonist(palette="paper", label="under @algebra_step",
                           width=5.4, height=2.4)
        a.move_to(np.array([-3.5, -0.6, 0]))
        b.move_to(np.array([3.5, -0.6, 0]))
        self.play(FadeIn(a, run_time=0.35), FadeIn(b, run_time=0.35))

        self.wait(1.5)

        # Free text path — gets it wrong
        a.stream_tokens(self, ["3x ", "= ", "14 ", "- ", "5"], speed=0.18)
        a.newline()
        a.stream_tokens(self, ["3x ", "= ", "9"], speed=0.16)
        a.newline()
        a.stream_tokens(self, ["x ", "= ", "9 ", "/ ", "3 ", "= ", "4"],
                        speed=0.16, color=Paper.bad)

        # Constrained path — gets it right
        b.stream_tokens(self, ["@algebra_step("], speed=0.16)
        b.newline()
        b.stream_tokens(self, ["  3x+5=14, simplify, 3x=9)"],
                        speed=0.14, color=Paper.good)
        b.newline()
        b.stream_tokens(self, ["@algebra_step("], speed=0.16)
        b.newline()
        b.stream_tokens(self, ["  3x=9, isolate_var, x=3)"],
                        speed=0.14, color=Paper.good)
        b.newline()
        b.stream_tokens(self, ["@done(x = 3) ✓"],
                        speed=0.14, color=Paper.good)

        self.wait(4.0)

        contrast = Text(
            "free-text 4/7      ·      constrained 6/7      ·      11 illegal-step rejections",
            font=theme.MONO_FALLBACK, font_size=15, color=Paper.ink_soft,
        )
        contrast.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(contrast, shift=UP * 0.1, run_time=0.5))
        self.wait(8.0)

        sub = Text(
            "Same weights. Different gate.",
            font="Georgia", slant="ITALIC", font_size=22, color=Paper.accent,
        )
        sub.next_to(contrast, DOWN, buff=0.2)
        self.play(FadeIn(sub, shift=UP * 0.05, run_time=0.4))
        self.wait(7.5)

        self.play(FadeOut(VGroup(hdr, prob, a, b, contrast, sub), run_time=0.5))

    # ----------------------------------------------------------------------
    # ACT 4 BEAT 2 — logic (2:25 – 2:40)
    # ----------------------------------------------------------------------

    def _act4_beat2_logic(self):
        hdr = Text(
            "Same gate. Different domain.",
            font="Georgia", slant="ITALIC", font_size=26, color=Paper.ink,
        )
        hdr.to_edge(UP, buff=0.6)
        self.play(FadeIn(hdr, shift=UP * 0.1, run_time=0.4))

        prob_lines = [
            "Given:    A → B    B → C    A",
            "Prove:    C",
        ]
        prob = VGroup(*[
            Text(t, font=theme.MONO_FALLBACK, font_size=20, color=Paper.ink_soft)
            for t in prob_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        prob.next_to(hdr, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(t) for t in prob], lag_ratio=0.2, run_time=0.7))

        # The composer pattern: 5 lines that loop on the leaves
        composer_lines: list[tuple[str, str]] = [
            ("@program(invocable=False)", Paper.accent),
            ("def derive():", Paper.ink),
            ("    while True:", Paper.ink),
            ("        step = yield gen.alternative(", Paper.ink),
            ("            [inference_step, qed])", Paper.ink_soft),
            ("        if step.name == 'qed':", Paper.ink),
            ("            return step.value", Paper.ink),
        ]
        composer = _code_lines(self, composer_lines, anchor_x=-5.5, top_y=0.5,
                                line_height=0.36, font_size=14)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.1) for ln in composer],
            lag_ratio=0.1, run_time=0.9,
        ))

        # The trace
        trace_lines = [
            ('@inference_step("A→B; A", modus_ponens, "B")', Paper.good),
            ('@inference_step("B→C; B", modus_ponens, "C")', Paper.good),
            ('@qed("C")  ✓', Paper.accent),
        ]
        trace_x = 0.5
        trace_top_y = 0.5
        trace_grp = VGroup()
        for i, (text, col) in enumerate(trace_lines):
            t = Text(text, font=theme.MONO_FALLBACK, font_size=14, color=col)
            t.align_to(np.array([trace_x, 0, 0]), LEFT)
            t.shift(UP * (trace_top_y - i * 0.45))
            trace_grp.add(t)
        self.play(LaggedStart(*[FadeIn(t, shift=LEFT * 0.1) for t in trace_grp],
                              lag_ratio=0.4, run_time=2.4))

        sub = Text(
            "Wherever 'legal step' is a Python predicate.",
            font="Georgia", slant="ITALIC", font_size=20,
            color=Paper.ink_soft,
        )
        sub.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(sub, shift=UP * 0.05, run_time=0.4))
        self.wait(9.0)

        self.play(FadeOut(VGroup(hdr, prob, composer, trace_grp, sub),
                          run_time=0.5))

    # ----------------------------------------------------------------------
    # ACT 4 BEAT 3 — meta finisher (2:40 – 2:55)
    # ----------------------------------------------------------------------

    def _act4_beat3_meta_finisher(self):
        hdr = Text(
            "The model writes its own primitive.",
            font="Georgia", slant="ITALIC", font_size=28, color=Paper.ink,
        )
        hdr.to_edge(UP, buff=0.5)
        self.play(FadeIn(hdr, shift=UP * 0.1, run_time=0.5))

        prob = Text("Solve:   x² − 5x + 6 = 0",
                    font=theme.MONO_FALLBACK, font_size=20,
                    color=Paper.ink_soft)
        prob.next_to(hdr, DOWN, buff=0.35)
        self.play(FadeIn(prob, run_time=0.3))

        # Hard cut: model emits @make_new_program
        emit1 = Text(
            '@make_new_program("quadratic_solver", ...)',
            font=theme.MONO_FALLBACK, font_size=18, color=Paper.accent,
        )
        emit1.move_to(UP * 1.0)
        self.play(FadeIn(emit1, shift=UP * 0.1, run_time=0.4))

        switch_label = Text(
            "[grammar switch → PROGRAM_SOURCE_GRAMMAR]",
            font=theme.MONO_FALLBACK, font_size=13, color=Paper.ink_soft,
        )
        switch_label.next_to(emit1, DOWN, buff=0.18)
        self.play(FadeIn(switch_label, run_time=0.35))
        self.wait(1.5)

        # Source materializes
        source_lines: list[tuple[str, str]] = [
            ("@program", Paper.accent),
            ("def quadratic_solver():", Paper.ink),
            ("    a     = yield gen.integer(-9, 9)", Paper.ink),
            ("    b     = yield gen.integer(-9, 9)", Paper.ink),
            ("    c     = yield gen.integer(-9, 9)", Paper.ink),
            ("    root1 = yield gen.integer(-9, 9)", Paper.ink),
            ("    root2 = yield gen.integer(-9, 9)", Paper.ink),
            ("    return {\"a\": a, \"b\": b, \"c\": c,", Paper.ink),
            ("            \"root1\": root1, \"root2\": root2}", Paper.ink),
        ]
        source = _code_lines(self, source_lines, anchor_x=-5.0, top_y=0.0,
                              line_height=0.36, font_size=14)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.1) for ln in source],
            lag_ratio=0.12, run_time=2.6,
        ))
        self.wait(1.5)

        compile_note = Text("[validated · compiled · registered]",
                            font=theme.MONO_FALLBACK, font_size=13,
                            color=Paper.good)
        compile_note.next_to(source, DOWN, buff=0.4)
        self.play(FadeIn(compile_note, shift=UP * 0.05, run_time=0.4))
        self.wait(1.5)

        # Use the new tool
        usage = Text(
            "@quadratic_solver(1, -5, 6, 2, 3)",
            font=theme.MONO_FALLBACK, font_size=18, color=Paper.accent,
        )
        usage.next_to(compile_note, DOWN, buff=0.3)
        self.play(FadeIn(usage, shift=UP * 0.08, run_time=0.4))

        done = Text("@done(\"x = 2 or x = 3\")  ✓",
                    font=theme.MONO_FALLBACK, font_size=18, color=Paper.good)
        done.next_to(usage, DOWN, buff=0.18)
        self.play(FadeIn(done, shift=UP * 0.05, run_time=0.35))
        self.wait(9.0)

        self.play(FadeOut(VGroup(hdr, prob, emit1, switch_label, source,
                                   compile_note, usage, done),
                          run_time=0.5))

    # ----------------------------------------------------------------------
    # CLOSE — thesis (2:55 – 3:00)
    # ----------------------------------------------------------------------

    def _close_thesis(self):
        thesis = VGroup(
            Text("Structured output constrained the shape.",
                 font="Georgia", font_size=26, color=Paper.ink_soft),
            Text("Tool calling constrained the side effect.",
                 font="Georgia", font_size=26, color=Paper.ink_soft),
            Text("orate lets the model enforce",
                 font="Georgia", font_size=32, color=Paper.ink),
            Text("the legality of its own thought.",
                 font="Georgia", slant="ITALIC", font_size=32,
                 color=Paper.accent),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        thesis.move_to(ORIGIN)

        self.play(LaggedStart(
            *[FadeIn(ln, shift=UP * 0.15) for ln in thesis],
            lag_ratio=0.45, run_time=3.6,
        ))
        self.wait(3.5)

        gh = Text("github.com/maltelandgren/orate",
                  font=theme.MONO_FALLBACK, font_size=16,
                  color=Paper.ink_soft)
        gh.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(gh, run_time=0.4))
        self.wait(5.0)
