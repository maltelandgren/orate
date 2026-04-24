"""
Candidate — full narrative arc in the hybrid D3 style.

Roughly 45 seconds, covering:
  Beat A  — title + cold-open hook               (4s)
  Beat B  — the pain: free-text slip             (8s)
  Beat C  — zoom into orate · the primitive      (10s)
  Beat D  — grammar bites (the money shot)        (8s)
  Beat E  — session continues: one KV, mode swap (8s)
  Beat F  — meta: model writes its own program    (5s)
  Close   — thesis                                (4s)

This is a render for the user to *watch* and judge as a direction.
Not final — many rough edges.
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
    RoundedRectangle,
    Scene,
    Text,
    UP,
    VGroup,
    Write,
    smooth,
    there_and_back,
)

import theme
from theme import Paper, Terminal
from llm import LLMProtagonist, LogitItem


class CandidateFull(Scene):
    def construct(self):
        self.camera.background_color = Paper.bg
        self.add(theme.paper_grid(opacity=0.35))

        # ================================================================
        # Beat A: title
        # ================================================================
        title = Text("orate", font="Georgia", slant="ITALIC",
                     font_size=72, color=Paper.ink)
        tagline = Text("grammar-constrained decoding, authored mid-inference",
                        font="Georgia", slant="ITALIC", font_size=22,
                        color=Paper.ink_soft)
        title.shift(UP * 0.5)
        tagline.next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(title, shift=UP * 0.2, run_time=0.9))
        self.play(FadeIn(tagline, run_time=0.5))
        self.wait(0.8)
        self.play(FadeOut(title, shift=UP * 0.1),
                  FadeOut(tagline, shift=UP * 0.05), run_time=0.5)

        # ================================================================
        # Beat B: the pain — free text slips
        # ================================================================
        hdr_b = Text("Same model. Same prompt. Two runs.",
                     font="Georgia", slant="ITALIC",
                     font_size=26, color=Paper.ink)
        hdr_b.to_edge(UP, buff=0.7)
        self.play(FadeIn(hdr_b, shift=UP * 0.1, run_time=0.45))

        prob = Text(
            "2x + 3y = 12,   x + y = 5,   x > y     (integers)",
            font=theme.MONO_FALLBACK, font_size=18, color=Paper.ink_soft,
        )
        prob.next_to(hdr_b, DOWN, buff=0.35)
        self.play(FadeIn(prob, run_time=0.35))

        a = LLMProtagonist(palette="paper", label="run A",
                            width=5.2, height=2.4)
        b = LLMProtagonist(palette="paper", label="run B",
                            width=5.2, height=2.4)
        a.move_to(np.array([-3.3, -0.9, 0]))
        b.move_to(np.array([3.3, -0.9, 0]))
        self.play(FadeIn(a, run_time=0.35), FadeIn(b, run_time=0.35))

        for line in [["x = 5 - y"], ["2(5-y) + 3y = 12"], ["10 + y = 12"]]:
            a.stream_tokens(self, line, speed=0.05)
            b.stream_tokens(self, line, speed=0.05)
            a.newline(); b.newline()
        a.stream_tokens(self, ["y = 2   ✓"], speed=0.08,
                        color=Paper.good)
        b.stream_tokens(self, ["y = 4   ✗"], speed=0.08,
                        color=Paper.bad)

        self.wait(0.5)
        pain = Text("The math doesn't constrain the tokens.",
                    font="Georgia", slant="ITALIC",
                    font_size=22, color=Paper.ink_soft)
        pain.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(pain, shift=UP * 0.08, run_time=0.4))
        self.wait(0.7)

        self.play(FadeOut(VGroup(a, b, pain, prob, hdr_b), run_time=0.45))

        # ================================================================
        # Beat C: zoom into orate's machinery — the primitive
        # ================================================================
        inset_shadow = VGroup(*[
            RoundedRectangle(
                width=13.0 + pad, height=7.0 + pad,
                corner_radius=0.22 + pad / 2,
                fill_color=BLACK, fill_opacity=op, stroke_opacity=0,
            ).shift([0, dy, 0])
            for pad, op, dy in [
                (0.24, 0.02, -0.34),
                (0.15, 0.035, -0.22),
                (0.07, 0.05, -0.13),
            ]
        ])
        inset = RoundedRectangle(
            width=13.0, height=7.0, corner_radius=0.22,
            fill_color=Terminal.bg, fill_opacity=1.0,
            stroke_color=Paper.ink_soft, stroke_width=1.4,
        )
        inset_tag = Text("  orate · session",
                         font=theme.MONO_FALLBACK, font_size=15,
                         color=Terminal.ink_soft)
        inset_tag.next_to(inset, UP, buff=0.1, aligned_edge=LEFT)
        self.play(
            FadeIn(inset_shadow, run_time=0.4),
            FadeIn(inset, scale=0.92, run_time=0.4),
            FadeIn(inset_tag, run_time=0.3),
        )

        # Show the @program code as source-in-prompt
        code_lines = [
            ("@program", Terminal.accent),
            ("def algebra_step(before):", Terminal.ink),
            ("    rule = yield gen.choice(", Terminal.ink),
            ("        [\"isolate_var\", \"simplify\", ...])", Terminal.amber),
            ("    after = yield gen.string(", Terminal.ink),
            ("        where=lambda s: equivalent_under(",
             Terminal.amber),
            ("            rule, before, s))", Terminal.amber),
            ("    return {\"rule\": rule, \"after\": after}", Terminal.ink),
        ]
        code = VGroup()
        for i, (t, col) in enumerate(code_lines):
            ln = Text(t, font=theme.MONO_FALLBACK, font_size=17, color=col)
            ln.align_to(np.array([-5.3, 0, 0]), LEFT)
            ln.shift(DOWN * (i * 0.4) + UP * 1.6)
            code.add(ln)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.1) for ln in code],
            lag_ratio=0.12, run_time=1.4,
        ))
        self.wait(0.3)

        # Callout: the `where=` clause highlight
        where_box = RoundedRectangle(
            width=7.5, height=0.9, corner_radius=0.1,
            fill_color=Terminal.accent, fill_opacity=0.12,
            stroke_color=Terminal.accent, stroke_width=1.4,
        )
        where_box.move_to(code[5].get_center() + DOWN * 0.2)
        where_label = Text("logic constraint", font=theme.SANS_FALLBACK,
                            font_size=13, color=Terminal.accent)
        where_label.next_to(where_box, DOWN, buff=0.12, aligned_edge=LEFT)
        self.play(FadeIn(where_box, run_time=0.3),
                  FadeIn(where_label, run_time=0.3))
        self.wait(0.5)
        self.play(FadeOut(where_box), FadeOut(where_label), run_time=0.3)

        # ================================================================
        # Beat D: grammar bites — the money shot
        # ================================================================
        self.play(FadeOut(code, run_time=0.5))

        llm_bound = LLMProtagonist(palette="terminal",
                                    label="qwen-7b  ·  orate-bound",
                                    width=5.4, height=2.0)
        llm_bound.move_to(np.array([-1.8, -0.3, 0]))
        self.play(FadeIn(llm_bound, run_time=0.35))

        ctx = Text("@algebra_step(\"10 + y = 12\")",
                   font=theme.MONO_FALLBACK, font_size=16,
                   color=Terminal.ink_soft)
        ctx.next_to(llm_bound, UP, buff=0.2)
        self.play(FadeIn(ctx, run_time=0.3))

        llm_bound.stream_tokens(self, ["rule = isolate_var"],
                                 speed=0.07, color=Terminal.amber)
        llm_bound.newline()
        llm_bound.stream_tokens(self, ["after = \""], speed=0.07)

        logits = [
            LogitItem("y = 2", 0.44),
            LogitItem("y = 4", 0.25),
            LogitItem("y + 2", 0.14),
            LogitItem("12/y", 0.10),
            LogitItem("??", 0.07),
        ]
        llm_bound.open_logits(self, logits, column_width=2.0, gap=0.3)
        self.wait(0.25)
        llm_bound.apply_grammar_mask(self, mask_indices=[1, 2, 3, 4])
        self.wait(0.3)
        llm_bound.choose_logit(self, 0)
        llm_bound.stream_tokens(self, ["\""], speed=0.07)
        self.wait(0.25)
        llm_bound.close_logits(self)

        ok = Text("✓ provably equivalent under isolate_var",
                  font=theme.MONO_FALLBACK, font_size=14,
                  color=Terminal.good)
        ok.next_to(llm_bound, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(ok, shift=UP * 0.08, run_time=0.3))
        self.wait(0.6)

        self.play(FadeOut(VGroup(llm_bound, ctx, ok), run_time=0.4))

        # ================================================================
        # Beat E: session continues — one KV, mode swap
        # ================================================================
        # KV cache bar growing at bottom
        kv_bar_bg = RoundedRectangle(
            width=10.0, height=0.2, corner_radius=0.08,
            fill_color=Terminal.grid, fill_opacity=1.0, stroke_opacity=0,
        )
        kv_bar_bg.to_edge(DOWN, buff=1.6)
        kv_bar_fg = RoundedRectangle(
            width=0.01, height=0.2, corner_radius=0.08,
            fill_color=Terminal.accent, fill_opacity=1.0, stroke_opacity=0,
        )
        kv_bar_fg.align_to(kv_bar_bg, LEFT)
        kv_bar_fg.align_to(kv_bar_bg, DOWN).shift(UP * 0.0)
        kv_label = Text("KV: 842 tokens",
                         font=theme.MONO_FALLBACK, font_size=14,
                         color=Terminal.ink_soft)
        kv_label.next_to(kv_bar_bg, UP, buff=0.1, aligned_edge=LEFT)
        self.play(FadeIn(kv_bar_bg), FadeIn(kv_bar_fg), FadeIn(kv_label),
                  run_time=0.3)

        # Simulate the bar growing as "programs" fire on the right
        program_sequence = [
            ("@dm_turn(scene)",           0.18, "1.2k"),
            ("@roll(perception, DC=15)",  0.30, "1.4k"),
            ("@remember({kind: npc})",    0.42, "1.6k"),
            ("@enter_combat(...)",        0.55, "1.9k"),
            ("@algebra_step(\"x+y=5\")",  0.72, "2.3k"),
            ("@inference_step([P→Q, P])", 0.90, "2.7k"),
        ]

        prog_texts = VGroup()
        for i, (prog, _, _) in enumerate(program_sequence):
            t = Text(prog, font=theme.MONO_FALLBACK, font_size=15,
                     color=Terminal.accent)
            t.move_to(np.array([2.8, 1.6 - i * 0.42, 0]))
            t.align_to(np.array([1.8, 0, 0]), LEFT)
            prog_texts.add(t)

        ppanel_hdr = Text("registry",
                          font=theme.MONO_FALLBACK, font_size=13,
                          color=Terminal.ink_soft)
        ppanel_hdr.next_to(prog_texts, UP, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(ppanel_hdr, run_time=0.3))

        # Pre-build all KV labels; swap via FadeOut/FadeIn to avoid the
        # structure-mismatch glitch that .become() causes on text of
        # different lengths.
        for i, (prog, frac, tok_count) in enumerate(program_sequence):
            target_w = frac * 10.0
            new_label = Text(f"KV: {tok_count} tokens",
                             font=theme.MONO_FALLBACK, font_size=14,
                             color=Terminal.ink_soft)
            new_label.next_to(kv_bar_bg, UP, buff=0.1, aligned_edge=LEFT)
            self.play(
                kv_bar_fg.animate.stretch_to_fit_width(target_w).align_to(
                    kv_bar_bg, LEFT),
                FadeIn(prog_texts[i], shift=LEFT * 0.15),
                FadeOut(kv_label, run_time=0.15),
                FadeIn(new_label, run_time=0.2),
                run_time=0.45,
            )
            kv_label = new_label
        self.wait(0.6)

        # Punctuation label
        kv_note = Text("one KV · mid-decode grammar-switches",
                        font="Georgia", slant="ITALIC",
                        font_size=18, color=Paper.ink)
        kv_note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(kv_note, shift=UP * 0.1, run_time=0.4))
        self.wait(0.8)

        # ================================================================
        # Beat F: zoom back out → thesis
        # ================================================================
        self.play(
            FadeOut(VGroup(inset_shadow, inset, inset_tag, kv_bar_bg,
                            kv_bar_fg, kv_label, prog_texts, ppanel_hdr,
                            kv_note), run_time=0.5),
        )

        thesis = VGroup(
            Text("Structured output constrained the shape.",
                 font="Georgia", font_size=28, color=Paper.ink_soft),
            Text("Tool calling constrained the side effect.",
                 font="Georgia", font_size=28, color=Paper.ink_soft),
            Text("orate lets the model enforce",
                 font="Georgia", font_size=34, color=Paper.ink),
            Text("the legality of its own thought.",
                 font="Georgia", slant="ITALIC", font_size=34,
                 color=Paper.accent),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        thesis.move_to(ORIGIN)
        self.play(LaggedStart(
            *[FadeIn(ln, shift=UP * 0.15) for ln in thesis],
            lag_ratio=0.38, run_time=2.2,
        ))
        self.wait(1.5)
