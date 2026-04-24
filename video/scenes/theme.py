"""
Shared palette, typography, and reusable mobject helpers for the orate demo.

Two palettes:
- PAPER: Anthropic-inspired warm minimalist
- TERMINAL: dark, honest, code-forward
"""

from __future__ import annotations

from manim import (
    BLACK,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    Line,
    Mobject,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Text,
    VGroup,
    VMobject,
)
import numpy as np


# ---------------------------------------------------------------------------
# PALETTES
# ---------------------------------------------------------------------------


class Paper:
    bg = "#F2EBE5"            # warm off-white
    bg_deep = "#ECE3DA"       # card recess
    grid = "#E4DCD1"          # subtle grid line
    ink = "#2A2520"           # primary text
    ink_soft = "#5E554C"      # secondary text
    accent = "#D4704A"        # terracotta
    accent_soft = "#E8A387"   # lighter tint
    mute = "#A89E92"          # muted gray
    good = "#6B8E5A"          # calm green
    bad = "#B8543F"            # clay red
    card = "#FFFFFF"          # card surface


class Terminal:
    bg = "#0B0D10"
    panel = "#14181D"
    grid = "#1E242B"
    ink = "#E4E4E4"
    ink_soft = "#8B95A1"
    accent = "#D4704A"
    amber = "#E0B070"
    good = "#7EC77F"
    bad = "#E16A5F"
    blue = "#7AB8E4"


# ---------------------------------------------------------------------------
# FONTS (best-effort; manim falls back if unavailable)
# ---------------------------------------------------------------------------
SERIF = "Iowan Old Style"      # macOS; falls back to Georgia/Times
SERIF_FALLBACK = "Georgia"
SANS = "Inter"                 # if installed; else system sans
SANS_FALLBACK = "Helvetica"
MONO = "JetBrains Mono"        # common dev font
MONO_FALLBACK = "Menlo"         # macOS default mono


def serif(txt: str, **kwargs) -> Text:
    kwargs.setdefault("font", SERIF_FALLBACK)  # Georgia is universal on mac
    kwargs.setdefault("color", Paper.ink)
    return Text(txt, **kwargs)


def sans(txt: str, **kwargs) -> Text:
    kwargs.setdefault("font", SANS_FALLBACK)
    kwargs.setdefault("color", Paper.ink)
    return Text(txt, **kwargs)


def mono(txt: str, **kwargs) -> Text:
    kwargs.setdefault("font", MONO_FALLBACK)
    kwargs.setdefault("color", Paper.ink)
    return Text(txt, **kwargs)


# ---------------------------------------------------------------------------
# PAPER BACKGROUND w/ subtle grid
# ---------------------------------------------------------------------------


def paper_grid(step: float = 0.5, opacity: float = 0.35) -> VGroup:
    """Subtle graph-paper grid — fades in behind content."""
    grid = VGroup()
    # manim default frame: ~14.22 w x 8 h at 16:9
    half_w, half_h = 8.0, 4.5
    x = -half_w
    while x <= half_w + 1e-6:
        grid.add(Line([x, -half_h, 0], [x, half_h, 0],
                      stroke_color=Paper.grid,
                      stroke_width=0.6,
                      stroke_opacity=opacity))
        x += step
    y = -half_h
    while y <= half_h + 1e-6:
        grid.add(Line([-half_w, y, 0], [half_w, y, 0],
                      stroke_color=Paper.grid,
                      stroke_width=0.6,
                      stroke_opacity=opacity))
        y += step
    return grid


# ---------------------------------------------------------------------------
# CARD (rounded rectangle with soft shadow)
# ---------------------------------------------------------------------------


def card(width: float,
         height: float,
         fill: str = Paper.card,
         corner: float = 0.18,
         shadow: bool = True) -> VGroup:
    """Rounded rectangle with a subtle stacked drop shadow."""
    group = VGroup()
    if shadow:
        for pad, op, dy in [
            (0.18, 0.015, -0.26),
            (0.12, 0.025, -0.18),
            (0.07, 0.035, -0.12),
            (0.03, 0.04, -0.07),
        ]:
            shadow_r = RoundedRectangle(
                width=width + pad, height=height + pad,
                corner_radius=corner + pad / 2,
                fill_color=BLACK, fill_opacity=op,
                stroke_opacity=0,
            ).shift(np.array([0.0, dy, 0.0]))
            group.add(shadow_r)
    body = RoundedRectangle(
        width=width, height=height, corner_radius=corner,
        fill_color=fill, fill_opacity=1.0,
        stroke_color=Paper.grid, stroke_width=1.0,
    )
    group.add(body)
    return group


# ---------------------------------------------------------------------------
# ANTHROPIC-STYLE STARBURST "THINKING" ICON
# ---------------------------------------------------------------------------


def starburst(radius: float = 0.22,
              points: int = 8,
              inner_ratio: float = 0.38,
              color: str = Paper.accent) -> VMobject:
    """Multi-pointed star used as a 'thinking' spinner."""
    verts = []
    for i in range(points * 2):
        r = radius if i % 2 == 0 else radius * inner_ratio
        theta = i * np.pi / points - np.pi / 2
        verts.append([r * np.cos(theta), r * np.sin(theta), 0.0])
    return Polygon(*verts,
                   fill_color=color, fill_opacity=1.0,
                   stroke_opacity=0)


# ---------------------------------------------------------------------------
# CURSOR for typed chat bars
# ---------------------------------------------------------------------------


def text_cursor(height: float = 0.3, color: str = Paper.accent) -> Rectangle:
    return Rectangle(
        width=0.035, height=height,
        fill_color=color, fill_opacity=1.0,
        stroke_opacity=0,
    )
