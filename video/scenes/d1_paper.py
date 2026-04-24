"""
Direction 1 — Paper / Anthropic-style warm minimalist.

Beat covered: pain → primitive → grammar bites → constrained answer.
Same content rendered in the user's reference aesthetic:
  - #F2EBE5 paper background
  - Georgia serif for headers
  - Terracotta accent #D4704A
  - Soft card shadows, sequential-stagger reveals
"""
from __future__ import annotations

import numpy as np
from manim import (
    AnimationGroup,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    LaggedStart,
    Line,
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
from theme import Paper, serif, sans, mono
from llm import LLMProtagonist, LogitItem


class D1_Paper(Scene):
    def construct(self):
        self.camera.background_color = Paper.bg
        grid = theme.paper_grid(opacity=0.35)
        self.add(grid)

        # --- Beat A: title card (serif, elegant) ---
        title = Text("orate", font="Georgia", font_size=64,
                     color=Paper.ink, slant="ITALIC")
        sub = Text("constrained decoding, authored mid-inference",
                   font="Helvetica", font_size=22, color=Paper.ink_soft)
        title.shift(UP * 0.5)
        sub.next_to(title, DOWN, buff=0.35)

        self.play(FadeIn(title, shift=UP * 0.2, run_time=0.8))
        self.play(FadeIn(sub, run_time=0.5))
        self.wait(0.5)
        self.play(
            FadeOut(title, shift=UP * 0.15),
            FadeOut(sub, shift=UP * 0.1),
            run_time=0.4,
        )

        # --- Beat B: the model, sampling freely, gets the wrong answer ---
        context_card = self._paper_card(width=10, height=0.9)
        context_card.shift(UP * 2.8)
        context = Text(
            "Find integers x, y with  2x + 3y = 12,  x + y = 5,  x > y",
            font=theme.MONO_FALLBACK, font_size=22, color=Paper.ink,
        )
        context.move_to(context_card.get_center())
        self.play(
            FadeIn(context_card, shift=DOWN * 0.15, run_time=0.5),
            FadeIn(context, run_time=0.5),
        )

        # LLM emits a free-text wrong answer
        llm_free = LLMProtagonist(palette="paper", label="LLM  ·  free text",
                                   width=7.5, height=2.6)
        llm_free.shift(DOWN * 0.3)
        self.play(FadeIn(llm_free, shift=DOWN * 0.2, run_time=0.5))
        llm_free.stream_tokens(
            self,
            ["x ", "= ", "5 ", "- ", "y"],
            speed=0.08,
        )
        llm_free.newline()
        llm_free.stream_tokens(
            self,
            ["2(5-y) ", "+ ", "3y ", "= ", "12"],
            speed=0.08,
        )
        llm_free.newline()
        llm_free.stream_tokens(
            self,
            ["10 ", "+ ", "y ", "= ", "12", "  "],
            speed=0.08,
        )
        llm_free.stream_tokens(
            self,
            ["so ", "y", " = ", "4"],
            speed=0.12, color=Paper.bad,
        )
        # Strikethrough the wrong tail
        tail = llm_free.output_group[-4:]
        strike = Line(
            tail[0].get_left() + np.array([-0.03, 0.02, 0]),
            tail[-1].get_right() + np.array([0.03, 0.02, 0]),
            stroke_color=Paper.bad, stroke_width=2.5,
        )
        self.play(FadeIn(strike, run_time=0.25))
        self.wait(0.4)

        # Fade LLM + strike out to make room for the "constrained" LLM
        self.play(FadeOut(VGroup(llm_free, strike), run_time=0.4))

        # --- Beat C: orate steps in. Show the grammar biting. ---
        llm_bound = LLMProtagonist(palette="paper",
                                   label="LLM  ·  grammar-constrained",
                                   width=5.4, height=1.9)
        llm_bound.shift(np.array([1.8, -1.0, 0]))
        self.play(FadeIn(llm_bound, shift=DOWN * 0.2, run_time=0.4))
        llm_bound.stream_tokens(self, ["10 ", "+ ", "y ", "= ", "12"],
                                speed=0.09)

        logits = [
            LogitItem("y = 2", 0.38),
            LogitItem("y = 4", 0.27),
            LogitItem("y + 2", 0.15),
            LogitItem("12/y", 0.12),
            LogitItem("??", 0.08),
        ]
        llm_bound.open_logits(self, logits, column_width=2.4, gap=0.35)
        # Label: "rule = isolate_var  |  where= equivalent_under(...)"
        rule_chip = self._chip(
            "where = equivalent_under(rule, before, after)",
            color=Paper.accent_soft,
        )
        rule_chip.next_to(llm_bound.logit_group, UP, buff=0.25)
        self.play(FadeIn(rule_chip, shift=DOWN * 0.08, run_time=0.35))
        self.wait(0.2)

        # Grammar mask eliminates invalid transformations
        llm_bound.apply_grammar_mask(self, mask_indices=[1, 2, 3, 4])
        self.wait(0.3)
        llm_bound.choose_logit(self, 0)
        self.wait(0.25)
        llm_bound.close_logits(self)

        # --- Beat D: thesis line, closing ---
        self.play(FadeOut(VGroup(llm_bound, rule_chip, context, context_card),
                          run_time=0.5))

        thesis = VGroup(
            serif("Structured output constrained the shape.",
                  font="Georgia", font_size=26, color=Paper.ink_soft),
            serif("Tool calling constrained the side effect.",
                  font="Georgia", font_size=26, color=Paper.ink_soft),
            serif("orate lets the model enforce",
                  font="Georgia", font_size=32, color=Paper.ink),
            serif("the legality of its own thought.",
                  font="Georgia", font_size=32, color=Paper.accent,
                  slant="ITALIC"),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        thesis.move_to(np.array([0, 0, 0]))

        self.play(LaggedStart(
            *[FadeIn(line, shift=UP * 0.15) for line in thesis],
            lag_ratio=0.35, run_time=2.0,
        ))
        self.wait(1.2)

    # ------------------------------------------------------------------
    def _paper_card(self, *, width: float, height: float) -> VGroup:
        return theme.card(width=width, height=height,
                           fill=Paper.card, corner=0.18)

    def _chip(self, text: str, color: str) -> VGroup:
        t = Text(text, font=theme.MONO_FALLBACK, font_size=14,
                 color=Paper.ink)
        pad_x, pad_y = 0.2, 0.08
        pill = RoundedRectangle(
            width=t.width + pad_x * 2, height=t.height + pad_y * 2,
            corner_radius=0.14,
            fill_color=color, fill_opacity=0.6,
            stroke_color=Paper.accent, stroke_width=1.0,
        )
        pill.move_to(t.get_center())
        return VGroup(pill, t)
