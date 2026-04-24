"""
Direction 2 — Terminal / code-forward dark aesthetic.

Same beat as d1: LLM free-text slips, grammar-constrained orate locks the
answer. Aesthetic: dark background, mono everywhere, amber/green/red syntax.
"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    LaggedStart,
    Line,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    UP,
    VGroup,
    Write,
    smooth,
)

import theme
from theme import Terminal
from llm import LLMProtagonist, LogitItem


class D2_Terminal(Scene):
    def construct(self):
        self.camera.background_color = Terminal.bg

        # Prompt-style title: no serif, just a blinking caret feel
        prompt = VGroup(
            Text(">", font=theme.MONO_FALLBACK, font_size=36,
                 color=Terminal.accent),
            Text("orate", font=theme.MONO_FALLBACK, font_size=36,
                 color=Terminal.ink),
            Text("  /  constrained decoding", font=theme.MONO_FALLBACK,
                 font_size=22, color=Terminal.ink_soft),
        ).arrange(RIGHT, buff=0.18)
        prompt.move_to(np.array([0, 0, 0]))
        self.play(FadeIn(prompt[0], run_time=0.3))
        self.play(FadeIn(prompt[1], shift=LEFT * 0.1, run_time=0.4))
        self.play(FadeIn(prompt[2], run_time=0.35))
        self.wait(0.4)
        self.play(FadeOut(prompt, run_time=0.35))

        # --- Context problem, code-block style ---
        problem_text = (
            "# find integers x, y\n"
            "2x + 3y == 12\n"
            "x + y   == 5\n"
            "x > y"
        )
        problem = Text(problem_text, font=theme.MONO_FALLBACK, font_size=20,
                        color=Terminal.ink,
                        line_spacing=1.1)
        problem.set_color_by_text_to_color_map = None
        # Hand-color each line cheaply: first line (comment) gray
        problem[0:15].set_color(Terminal.ink_soft)
        panel = RoundedRectangle(
            width=problem.width + 1.2, height=problem.height + 0.7,
            corner_radius=0.1,
            fill_color=Terminal.panel, fill_opacity=1.0,
            stroke_color=Terminal.grid, stroke_width=1.2,
        )
        panel.move_to(problem.get_center())
        panel_group = VGroup(panel, problem)
        panel_group.to_edge(UP, buff=0.5).shift(LEFT * 3.6)
        self.play(FadeIn(panel_group, shift=DOWN * 0.15, run_time=0.5))

        # --- Free-text LLM slips ---
        llm_free = LLMProtagonist(
            palette="terminal", label="qwen-7b  ·  free",
            width=8.0, height=2.6,
        )
        llm_free.shift(DOWN * 0.4 + RIGHT * 1.0)
        self.play(FadeIn(llm_free, run_time=0.4))
        llm_free.stream_tokens(self,
            ["x ", "= ", "5 ", "- ", "y"],
            speed=0.07)
        llm_free.newline()
        llm_free.stream_tokens(self,
            ["2(5-y)", " + ", "3y ", "= ", "12"],
            speed=0.07)
        llm_free.newline()
        llm_free.stream_tokens(self,
            ["10 ", "+ ", "y ", "= ", "12 "],
            speed=0.07)
        llm_free.stream_tokens(self,
            ["→ ", "y=4"],
            speed=0.12, color=Terminal.bad)
        err = Text("✗ arithmetic slip (same seed, same model)",
                   font=theme.MONO_FALLBACK, font_size=16,
                   color=Terminal.bad)
        err.next_to(llm_free, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(err, shift=UP * 0.08, run_time=0.3))
        self.wait(0.5)

        # --- Cut to grammar-constrained variant ---
        self.play(FadeOut(VGroup(llm_free, err), run_time=0.4))

        # Grammar rail on right
        rail_lines = [
            ("outer", Terminal.ink_soft),
            ("@algebra_step", Terminal.accent),
            (" rule: choice[5]", Terminal.amber),
            (" after: string where equivalent_under(...)",
             Terminal.amber),
        ]
        rail = VGroup()
        for i, (t, col) in enumerate(rail_lines):
            ln = Text(t, font=theme.MONO_FALLBACK, font_size=16, color=col)
            ln.align_to(np.array([3.7, 0, 0]), LEFT)
            ln.shift(DOWN * (i * 0.4) + UP * 0.6)
            rail.add(ln)
        rail_title = Text("grammar stack", font=theme.MONO_FALLBACK,
                          font_size=14, color=Terminal.ink_soft)
        rail_title.next_to(rail, UP, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(rail, shift=LEFT * 0.15, run_time=0.4),
                  FadeIn(rail_title, run_time=0.3))

        llm_bound = LLMProtagonist(
            palette="terminal", label="qwen-7b  ·  orate-bound",
            width=5.6, height=1.9,
        )
        llm_bound.shift(DOWN * 0.7 + LEFT * 0.4)
        self.play(FadeIn(llm_bound, run_time=0.35))
        llm_bound.stream_tokens(self, ["10 ", "+ ", "y ", "= ", "12"],
                                 speed=0.08)

        logits = [
            LogitItem("y = 2", 0.44),
            LogitItem("y = 4", 0.25),
            LogitItem("y + 2", 0.14),
            LogitItem("12/y", 0.10),
            LogitItem("???", 0.07),
        ]
        llm_bound.open_logits(self, logits, column_width=2.1, gap=0.3)
        self.wait(0.15)
        llm_bound.apply_grammar_mask(self, mask_indices=[1, 2, 3, 4])
        self.wait(0.3)
        llm_bound.choose_logit(self, 0)
        self.wait(0.25)
        llm_bound.close_logits(self)

        ok = Text("✓ equivalent under: isolate_var",
                  font=theme.MONO_FALLBACK, font_size=16,
                  color=Terminal.good)
        ok.next_to(llm_bound, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(ok, shift=UP * 0.08, run_time=0.3))
        self.wait(0.8)

        # --- Thesis card ---
        self.play(FadeOut(VGroup(panel_group, llm_bound, ok, rail,
                                  rail_title), run_time=0.4))

        thesis = VGroup(
            Text("# structured output  →  shape",
                 font=theme.MONO_FALLBACK, font_size=22,
                 color=Terminal.ink_soft),
            Text("# tool calling       →  side effect",
                 font=theme.MONO_FALLBACK, font_size=22,
                 color=Terminal.ink_soft),
            Text("# orate              →  legality of thought",
                 font=theme.MONO_FALLBACK, font_size=24,
                 color=Terminal.accent),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        thesis.move_to(np.array([0, 0, 0]))
        self.play(LaggedStart(
            *[FadeIn(l, shift=UP * 0.08) for l in thesis],
            lag_ratio=0.4, run_time=1.6,
        ))
        self.wait(1.2)
