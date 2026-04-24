"""
Script variation V2 — "Protagonist-first" cold open.

Instead of opening on the simulacrum pain (Act 1 of the original script),
we open on the LLM itself as the protagonist. The camera is on the box
from frame 1. We see it sample. We see it slip. Then the library arrives
and the grammar bites. This is the 'Keep thinking' framing.

Voiceover sketch (not rendered — subtitles are proxy):
  0:00  "This is a language model. Every token is a sample from a
         distribution over its vocabulary."
  0:08  "Three years in, we're good at shaping those distributions —"
  0:13  "— prompt, fine-tune, RLHF, structured output, tool calls —"
  0:18  "but we're still running a lottery. Same model, same prompt,
         two runs:"
  0:23  [both runs complete — one right, one wrong]
  0:28  "orate closes the gap. Schemas with logic, authored mid-inference,
         binding the model to legal steps."
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
    ORIGIN,
    RIGHT,
    Scene,
    Text,
    UP,
    VGroup,
    Write,
)

import theme
from theme import Paper
from llm import LLMProtagonist, LogitItem


class V2_ProtagonistOpen(Scene):
    def construct(self):
        self.camera.background_color = Paper.bg
        self.add(theme.paper_grid(opacity=0.35))

        # --- Beat 1: single LLM appears, emits a line, opens its logits ---
        llm = LLMProtagonist(palette="paper", label="language model",
                              width=6.0, height=2.2)
        llm.move_to(ORIGIN)
        self.play(FadeIn(llm, shift=UP * 0.25, run_time=0.6))

        sub1 = Text(
            "Every token is a sample from a distribution over its vocabulary.",
            font="Georgia", slant="ITALIC", font_size=20,
            color=Paper.ink_soft,
        )
        sub1.to_edge(DOWN, buff=0.9)
        self.play(FadeIn(sub1, shift=UP * 0.1, run_time=0.5))

        llm.stream_tokens(self, ["the ", "cat ", "sat ", "on ", "the "],
                          speed=0.1)
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
        self.wait(0.5)
        llm.choose_logit(self, 0)
        self.wait(0.25)
        llm.close_logits(self)
        self.wait(0.4)
        self.play(FadeOut(sub1), FadeOut(llm), run_time=0.4)

        # --- Beat 2: "we've spent 3 years shaping these distributions" ---
        sub2 = Text(
            "Three years in, we're good at shaping those distributions.",
            font="Georgia", slant="ITALIC", font_size=22,
            color=Paper.ink,
        )
        sub2.shift(UP * 2.4)
        self.play(FadeIn(sub2, shift=UP * 0.1, run_time=0.5))

        # Four chips: prompt, fine-tune, structured output, tool calls
        chips = VGroup(*[
            self._chip(t) for t in [
                "prompts", "fine-tuning", "structured output", "tool calls",
            ]
        ]).arrange(RIGHT, buff=0.45)
        chips.move_to(UP * 1.1)
        self.play(LaggedStart(
            *[FadeIn(c, shift=UP * 0.1) for c in chips],
            lag_ratio=0.2, run_time=1.2,
        ))
        self.wait(0.3)

        sub3 = Text(
            "But shape and effect aren't the same as legality.",
            font="Georgia", slant="ITALIC", font_size=22,
            color=Paper.ink,
        )
        sub3.shift(DOWN * 0.25)
        self.play(FadeIn(sub3, shift=UP * 0.1, run_time=0.5))
        self.wait(0.6)

        # --- Beat 3: same model, same prompt, two runs ---
        self.play(FadeOut(VGroup(sub2, sub3, chips), run_time=0.4))

        setup = Text("same model · same prompt · two runs",
                     font=theme.MONO_FALLBACK, font_size=18,
                     color=Paper.ink_soft)
        setup.to_edge(UP, buff=0.6)
        self.play(FadeIn(setup, run_time=0.3))

        a = LLMProtagonist(palette="paper", label="run A",
                            width=5.2, height=2.3)
        b = LLMProtagonist(palette="paper", label="run B",
                            width=5.2, height=2.3)
        a.move_to(np.array([-3.3, -0.6, 0]))
        b.move_to(np.array([3.3, -0.6, 0]))
        self.play(FadeIn(a, run_time=0.35), FadeIn(b, run_time=0.35))

        # Stream with small lag between the two
        for line in [
            ["x = 5 - y"],
            ["2(5-y) + 3y = 12"],
            ["10 + y = 12"],
        ]:
            a.stream_tokens(self, line, speed=0.05)
            b.stream_tokens(self, line, speed=0.05)
            a.newline(); b.newline()

        a.stream_tokens(self, ["y = 2   ✓"], speed=0.07,
                          color=Paper.good)
        b.stream_tokens(self, ["y = 4   ✗"], speed=0.07,
                          color=Paper.bad)

        self.wait(0.5)

        # Mini thesis + hook into Act 2
        hook = Text("orate closes the gap.",
                    font="Georgia", font_size=30,
                    color=Paper.accent)
        hook.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(hook, shift=UP * 0.1, run_time=0.5))
        self.wait(1.0)

    # ------------------------------------------------------------------
    def _chip(self, label: str):
        from manim import RoundedRectangle
        t = Text(label, font=theme.SANS_FALLBACK, font_size=16,
                 color=Paper.ink)
        pad_x, pad_y = 0.3, 0.12
        pill = RoundedRectangle(
            width=t.width + pad_x * 2, height=t.height + pad_y * 2,
            corner_radius=0.22,
            fill_color=Paper.card, fill_opacity=1.0,
            stroke_color=Paper.grid, stroke_width=1.0,
        )
        pill.move_to(t.get_center())
        return VGroup(pill, t)
