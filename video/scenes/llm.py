"""
LLMProtagonist — the animated autoregressive LLM character.

Two states:
  - CLOSED: a small card that emits tokens one-by-one (default).
  - OPEN:   the card expands left into a 'logit column' revealing candidate
            next tokens with probabilities. Grammar masks can strike out
            invalid candidates; the sampler pulls the winner into the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from manim import (
    AddTextLetterByLetter,
    Animation,
    AnimationGroup,
    BLACK,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    Line,
    Mobject,
    ORIGIN,
    Polygon,
    Rectangle,
    RIGHT,
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
from theme import Paper, Terminal, mono, sans, serif


@dataclass
class LogitItem:
    """One candidate next-token with a scalar probability 0..1."""
    token: str
    prob: float
    masked: bool = False          # grammar says: illegal
    chosen: bool = False          # sampler's pick


class LLMProtagonist(VGroup):
    """An animated autoregressive LLM."""

    def __init__(
        self,
        *,
        width: float = 3.6,
        height: float = 1.9,
        palette: str = "paper",    # "paper" or "terminal"
        label: str = "LLM",
        title_font_size: int = 20,
        output_font_size: int = 22,
        mono_font: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.palette = palette
        if palette == "paper":
            self.col = Paper
            self.card_fill = Paper.card
            self.stroke_col = Paper.grid
            self.ink = Paper.ink
            self.ink_soft = Paper.ink_soft
            self.accent = Paper.accent
        else:
            self.col = Terminal
            self.card_fill = Terminal.panel
            self.stroke_col = Terminal.grid
            self.ink = Terminal.ink
            self.ink_soft = Terminal.ink_soft
            self.accent = Terminal.accent

        self.box_width = width
        self.box_height = height
        self.mono_font = mono_font or theme.MONO_FALLBACK

        # Main box
        self.box = RoundedRectangle(
            width=width, height=height, corner_radius=0.14,
            fill_color=self.card_fill, fill_opacity=1.0,
            stroke_color=self.stroke_col, stroke_width=1.2,
        )

        # Soft shadow — simulate blur by stacking slightly-larger
        # rounded rects beneath the card with decreasing opacities.
        self.shadows = VGroup()
        for i, (pad, op, dy) in enumerate([
            (0.18, 0.015, -0.26),
            (0.12, 0.025, -0.18),
            (0.07, 0.035, -0.12),
            (0.03, 0.04, -0.07),
        ]):
            self.shadows.add(RoundedRectangle(
                width=width + pad, height=height + pad, corner_radius=0.14 + pad / 2,
                fill_color=BLACK, fill_opacity=op,
                stroke_opacity=0,
            ).shift(np.array([0.0, dy, 0.0])))

        # Title label
        self.label = Text(
            label, font_size=title_font_size,
            font=theme.SANS_FALLBACK, color=self.ink_soft,
        )
        self.label.move_to(self.box.get_top() + DOWN * 0.25)

        # Output tokens live in a buffer; we place them as a row of Text
        self.output_group = VGroup()
        self.output_font_size = output_font_size
        self.cursor = Rectangle(
            width=0.03, height=0.32,
            fill_color=self.accent, fill_opacity=1.0,
            stroke_opacity=0,
        )
        self.cursor.move_to(self._cursor_home())

        # Optional logit column (built on-demand)
        self.logit_group: Optional[VGroup] = None
        self.logit_rows: list[VGroup] = []
        self.logit_arrow: Optional[Mobject] = None

        self.add(self.shadows, self.box, self.label, self.output_group,
                 self.cursor)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def stream_tokens(
        self,
        scene: Scene,
        tokens: Sequence[str],
        *,
        speed: float = 0.08,
        color: Optional[str] = None,
    ) -> None:
        """Append tokens one by one with typewriter feel."""
        color = color or self.ink
        for tok in tokens:
            self._append_token(scene, tok, color=color, run_time=speed)

    def open_logits(
        self,
        scene: Scene,
        logits: Sequence[LogitItem],
        *,
        column_width: float = 2.4,
        gap: float = 0.45,
    ) -> VGroup:
        """Reveal the logit column to the LEFT of the box with a slide-in."""
        self.logit_group, self.logit_rows = self._build_logit_column(
            logits, column_width=column_width,
        )
        self.logit_group.next_to(self.box, LEFT, buff=gap)
        # Arrow from column to box
        self.logit_arrow = self._build_arrow(
            self.logit_group.get_right() + np.array([0.05, 0, 0]),
            self.box.get_left() + np.array([-0.05, 0, 0]),
        )
        scene.play(
            FadeIn(self.logit_group, shift=np.array([0.3, 0, 0]),
                   run_time=0.5),
            FadeIn(self.logit_arrow, run_time=0.3),
        )
        return self.logit_group

    def apply_grammar_mask(
        self,
        scene: Scene,
        *,
        mask_indices: Sequence[int],
        run_time: float = 0.8,
    ) -> None:
        """Fade masked logit rows to gray + add a strike-through line."""
        if self.logit_group is None:
            return
        anims = []
        for idx in mask_indices:
            row = self.logit_rows[idx]
            strike = Line(
                row.get_left() + np.array([0.05, 0, 0]),
                row.get_right() + np.array([-0.05, 0, 0]),
                stroke_color=self.col.bad, stroke_width=2.2,
            )
            # Attach strike to the logit group so close_logits takes it too.
            self.logit_group.add(strike)
            row.strike = strike
            anims.append(row.animate.set_opacity(0.35))
            anims.append(FadeIn(strike, run_time=run_time * 0.6))
        scene.play(*anims, run_time=run_time)

    def choose_logit(
        self,
        scene: Scene,
        idx: int,
        *,
        run_time: float = 0.8,
    ) -> None:
        """Highlight the chosen logit and fly it into the output buffer."""
        if self.logit_group is None:
            return
        row = self.logit_rows[idx]
        # Highlight: animate the row's stroke directly — no stray copy.
        scene.play(
            row.animate.set_stroke(self.accent, width=2.2),
            run_time=0.3,
        )
        # Extract token text and append
        token_str = row.meta_token  # set during build
        self._append_token(scene, token_str, color=self.accent,
                           run_time=run_time * 0.6)

    def close_logits(self, scene: Scene) -> None:
        """Dismiss the logit column."""
        if self.logit_group is None:
            return
        targets = [self.logit_group]
        if self.logit_arrow is not None:
            targets.append(self.logit_arrow)
        scene.play(*[FadeOut(t, shift=np.array([-0.3, 0, 0]))
                     for t in targets], run_time=0.4)
        self.logit_group = None
        self.logit_arrow = None

    def pulse_thinking(self, scene: Scene, *, duration: float = 1.0) -> None:
        """Small starburst at the cursor to signal 'thinking'."""
        burst = theme.starburst(radius=0.12, color=self.accent)
        burst.move_to(self.cursor.get_center() + np.array([0.1, 0, 0]))
        scene.play(FadeIn(burst, run_time=0.2))
        scene.play(burst.animate.rotate(np.pi * 2), run_time=duration,
                   rate_func=smooth)
        scene.play(FadeOut(burst, run_time=0.2))

    def clear_output(self, scene: Scene) -> None:
        """Reset output buffer (keep the box in place)."""
        if len(self.output_group) == 0:
            return
        scene.play(FadeOut(self.output_group, run_time=0.3))
        self.output_group.become(VGroup())
        self.cursor.move_to(self._cursor_home())

    def newline(self) -> None:
        """Jump cursor to a new line at the left margin (no animation)."""
        home_x = (self.box.get_corner(UP + LEFT)[0] + 0.25)
        new_y = self.cursor.get_center()[1] - 0.42
        self.cursor.move_to(np.array([home_x, new_y, 0.0]))

    def fade_everything(self, scene: Scene, *, run_time: float = 0.4) -> None:
        """FadeOut the LLM plus any open logit column + arrow."""
        targets = [self]
        if self.logit_group is not None:
            targets.append(self.logit_group)
        if self.logit_arrow is not None:
            targets.append(self.logit_arrow)
        scene.play(*[FadeOut(t) for t in targets], run_time=run_time)
        self.logit_group = None
        self.logit_arrow = None
        self.logit_rows = []

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _cursor_home(self) -> np.ndarray:
        # Left-inside of box, slightly below title
        return (self.box.get_corner(UP + LEFT)
                + np.array([0.25, -0.8, 0.0]))

    def _append_token(self, scene: Scene, tok: str, *,
                      color: str, run_time: float) -> None:
        t = Text(tok, font=self.mono_font,
                 font_size=self.output_font_size, color=color)
        # Position to the right of the current cursor
        cursor_pos = self.cursor.get_center()
        t.move_to(cursor_pos + np.array([t.width / 2 + 0.05, 0, 0]))
        # Handle wrap: if past the right edge, drop to a new line
        right_limit = self.box.get_right()[0] - 0.25
        if t.get_right()[0] > right_limit:
            new_y = cursor_pos[1] - 0.42
            t.move_to(np.array([
                self.box.get_corner(UP + LEFT)[0] + 0.25 + t.width / 2,
                new_y, 0.0,
            ]))
        self.output_group.add(t)
        new_cursor_pos = t.get_right() + np.array([0.06, 0, 0])
        scene.play(
            FadeIn(t, run_time=run_time),
            self.cursor.animate.move_to(
                new_cursor_pos + np.array([0.03, 0, 0])),
            run_time=run_time,
        )

    def _build_logit_column(self, logits: Sequence[LogitItem], *,
                            column_width: float):
        group = VGroup()
        row_list: list[VGroup] = []
        row_h = 0.42
        total_h = row_h * len(logits) + 0.5
        bg = RoundedRectangle(
            width=column_width, height=total_h, corner_radius=0.12,
            fill_color=self.card_fill, fill_opacity=1.0,
            stroke_color=self.stroke_col, stroke_width=1.0,
        )
        group.add(bg)
        header = Text("logits", font=theme.SANS_FALLBACK,
                      font_size=16, color=self.ink_soft)
        header.move_to(bg.get_top() + np.array([0, -0.2, 0]))
        group.add(header)

        for i, item in enumerate(logits):
            y = bg.get_top()[1] - 0.5 - (i * row_h)
            tok = Text(item.token, font=self.mono_font,
                       font_size=18, color=self.ink)
            bar_w = max(0.1, item.prob * (column_width - 1.4))
            bar = Rectangle(
                width=bar_w, height=0.14,
                fill_color=self.accent, fill_opacity=0.85,
                stroke_opacity=0,
            )
            left_x = bg.get_left()[0] + 0.28
            tok.move_to(np.array([left_x + tok.width / 2, y, 0]))
            bar.move_to(np.array([
                bg.get_right()[0] - 0.2 - bar_w / 2, y, 0,
            ]))
            row = VGroup(tok, bar)
            row.meta_token = item.token
            row.meta_prob = item.prob
            group.add(row)
            row_list.append(row)
        return group, row_list

    def _build_arrow(self, start, end):
        """Skinny arrow from logit column to box."""
        line = Line(
            start, end,
            stroke_color=self.ink_soft, stroke_width=1.6,
        )
        # Small arrowhead
        tip = Polygon(
            end,
            end + np.array([-0.15, 0.08, 0]),
            end + np.array([-0.15, -0.08, 0]),
            fill_color=self.ink_soft, fill_opacity=1.0,
            stroke_opacity=0,
        )
        return VGroup(line, tip)
