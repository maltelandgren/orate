"""
Direction 3 — Hybrid. Paper aesthetic for voice & narrative, terminal
aesthetic for the technical moments (logit column, grammar rail).

The effect I'm going for: warm and product-like until the camera "zooms in"
on the machinery. This is closest to product-marketing + hackathon-honesty.
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
    Polygon,
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
from theme import Paper, Terminal
from llm import LLMProtagonist, LogitItem


class D3_Hybrid(Scene):
    def construct(self):
        self.camera.background_color = Paper.bg
        grid = theme.paper_grid(opacity=0.35)
        self.add(grid)

        # --- Act 1 mini: "Free-text slips. Same model, same prompt." ---
        title = Text("Same model.  Same prompt.  Two runs.",
                     font="Georgia", slant="ITALIC",
                     font_size=30, color=Paper.ink)
        title.to_edge(UP, buff=0.8)
        self.play(FadeIn(title, shift=UP * 0.15, run_time=0.6))

        prob = Text(
            "2x + 3y = 12,  x + y = 5,  x > y     (find integer x, y)",
            font=theme.MONO_FALLBACK, font_size=19, color=Paper.ink_soft,
        )
        prob.next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(prob, run_time=0.4))

        # Two side-by-side LLMs (paper palette)
        llm_a = LLMProtagonist(palette="paper", label="run A",
                                width=5.2, height=2.6)
        llm_b = LLMProtagonist(palette="paper", label="run B",
                                width=5.2, height=2.6)
        llm_a.move_to(np.array([-3.3, -0.8, 0]))
        llm_b.move_to(np.array([3.3, -0.8, 0]))
        self.play(FadeIn(llm_a, run_time=0.35), FadeIn(llm_b, run_time=0.35))

        # Stream both, diverging at the end
        for tokens in [
            ["x = 5 - y"],
            ["2(5-y) + 3y = 12"],
        ]:
            llm_a.stream_tokens(self, tokens, speed=0.04)
            llm_b.stream_tokens(self, tokens, speed=0.04)
            llm_a.newline(); llm_b.newline()

        llm_a.stream_tokens(self, ["10 + y = 12"], speed=0.04)
        llm_b.stream_tokens(self, ["10 + y = 12"], speed=0.04)
        llm_a.newline(); llm_b.newline()
        llm_a.stream_tokens(self, ["y = 2   ✓"], speed=0.08,
                            color=Paper.good)
        llm_b.stream_tokens(self, ["y = 4   ✗"], speed=0.08,
                            color=Paper.bad)

        self.wait(0.6)

        subtitle = Text("The math doesn't constrain the tokens.",
                        font="Georgia", slant="ITALIC",
                        font_size=22, color=Paper.ink_soft)
        subtitle.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(subtitle, shift=UP * 0.1, run_time=0.5))
        self.wait(0.8)

        # --- Transition: the paper world dims and a terminal inset
        #     ZOOMS in from the center. This sells "opening the box." ---
        self.play(
            FadeOut(VGroup(llm_a, llm_b, subtitle, prob, title),
                    run_time=0.5),
        )

        # A terminal inset panel over the paper bg
        inset = RoundedRectangle(
            width=12.5, height=6.6, corner_radius=0.22,
            fill_color=Terminal.bg, fill_opacity=1.0,
            stroke_color=Paper.ink_soft, stroke_width=1.4,
        )
        # Soft shadow for the inset to pop off the paper
        shadows = VGroup(*[
            RoundedRectangle(
                width=12.5 + pad, height=6.6 + pad,
                corner_radius=0.22 + pad / 2,
                fill_color=BLACK, fill_opacity=op, stroke_opacity=0,
            ).shift([0, dy, 0])
            for pad, op, dy in [
                (0.22, 0.02, -0.32),
                (0.14, 0.035, -0.22),
                (0.07, 0.05, -0.13),
            ]
        ])
        inset_tag = Text("  orate · session",
                         font=theme.MONO_FALLBACK, font_size=16,
                         color=Terminal.ink_soft)
        inset_tag.next_to(inset, UP, buff=0.1, aligned_edge=LEFT)
        self.play(
            FadeIn(shadows, scale=0.9, run_time=0.35),
            FadeIn(inset, scale=0.9, run_time=0.35),
            FadeIn(inset_tag, run_time=0.3),
        )

        # Grammar rail along the right edge of the inset
        rail_lines = [
            ("outer:", Terminal.ink_soft),
            ("(text | @call)*", Terminal.ink),
            ("", None),
            ("@algebra_step:", Terminal.accent),
            (" rule: choice[5]", Terminal.amber),
            (" after: str", Terminal.amber),
            (" where: equivalent_under(", Terminal.amber),
            ("        rule, before, after)", Terminal.amber),
        ]
        rail = VGroup()
        for i, (t, col) in enumerate(rail_lines):
            if not t:
                continue
            ln = Text(t, font=theme.MONO_FALLBACK, font_size=14,
                     color=col or Terminal.ink)
            ln.move_to(np.array([3.5, 1.9 - i * 0.36, 0]))
            ln.align_to(np.array([2.7, 0, 0]), LEFT)
            rail.add(ln)
        rail_title = Text("grammar stack",
                         font=theme.MONO_FALLBACK, font_size=12,
                         color=Terminal.ink_soft)
        rail_title.next_to(rail, UP, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(rail_title, run_time=0.25),
                  LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in rail],
                              lag_ratio=0.1, run_time=0.8))

        # The now-constrained LLM, inside the terminal inset
        llm_bound = LLMProtagonist(palette="terminal",
                                    label="qwen-7b  ·  orate-bound",
                                    width=5.4, height=2.0)
        llm_bound.move_to(np.array([-1.7, -0.4, 0]))
        self.play(FadeIn(llm_bound, run_time=0.35))

        # Show the before-context
        ctx = Text("@algebra_step(\"10 + y = 12\")",
                   font=theme.MONO_FALLBACK, font_size=16,
                   color=Terminal.ink_soft)
        ctx.next_to(llm_bound, UP, buff=0.2)
        self.play(FadeIn(ctx, run_time=0.3))

        llm_bound.stream_tokens(self, ["rule ", "= ", "isolate_var"],
                                 speed=0.08, color=Terminal.amber)
        llm_bound.newline()
        llm_bound.stream_tokens(self, ["after ", "= ", "\""],
                                 speed=0.08)

        # Now the logits open
        logits = [
            LogitItem("y = 2", 0.44),
            LogitItem("y = 4", 0.25),
            LogitItem("y + 2", 0.14),
            LogitItem("12/y", 0.10),
            LogitItem("??", 0.07),
        ]
        llm_bound.open_logits(self, logits, column_width=2.1, gap=0.3)
        self.wait(0.2)
        llm_bound.apply_grammar_mask(self, mask_indices=[1, 2, 3, 4])
        self.wait(0.3)
        llm_bound.choose_logit(self, 0)
        llm_bound.stream_tokens(self, ["\""], speed=0.08)
        self.wait(0.2)
        llm_bound.close_logits(self)

        ok = Text("✓ equivalent under isolate_var",
                  font=theme.MONO_FALLBACK, font_size=15,
                  color=Terminal.good)
        ok.next_to(llm_bound, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(ok, shift=UP * 0.08, run_time=0.3))
        self.wait(0.8)

        # --- Zoom back out: fade the inset ---
        self.play(
            FadeOut(VGroup(llm_bound, ctx, ok, rail, rail_title,
                            inset_tag, inset, shadows), run_time=0.5)
        )

        # --- Thesis card (paper, serif, final) ---
        thesis = VGroup(
            Text("Structured output constrained the shape.",
                 font="Georgia", font_size=26, color=Paper.ink_soft),
            Text("Tool calling constrained the side effect.",
                 font="Georgia", font_size=26, color=Paper.ink_soft),
            Text("orate lets the model enforce", font="Georgia",
                 font_size=32, color=Paper.ink),
            Text("the legality of its own thought.", font="Georgia",
                 slant="ITALIC", font_size=32, color=Paper.accent),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        thesis.move_to(ORIGIN)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=UP * 0.15) for ln in thesis],
            lag_ratio=0.35, run_time=2.0,
        ))
        self.wait(1.2)
