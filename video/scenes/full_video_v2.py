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
    Indicate,
    LEFT,
    LaggedStart,
    Line,
    ORIGIN,
    RIGHT,
    Polygon,
    Rectangle,
    RegularPolygon,
    Rotate,
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
    # Thin-space tracking on chip labels — without it the glyphs read
    # cramped at 1080p+ (especially when the stroke thickens on
    # highlight, which visually nudges letters together).
    t = Text(_spaced(label), font=theme.SANS_FALLBACK,
             font_size=font_size, color=fg)
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


def _make_sword(scale: float = 1.0, color: str = None) -> VGroup:
    """A simple line-art sword: blade + tip + crossguard + handle + pommel.

    Composed from filled rectangles + a triangle tip + a tiny hex
    pommel — nothing fancy, just a recognisable silhouette at small
    scale. Pointing UP by default; rotate the returned VGroup for
    other orientations.
    """
    if color is None:
        color = Paper.ink
    blade = Rectangle(
        width=0.10, height=2.0,
        fill_color=color, fill_opacity=1.0, stroke_width=0,
    )
    blade.move_to(np.array([0, 0.10, 0]))
    tip = Polygon(
        np.array([-0.05, 1.10, 0]),
        np.array([0.05, 1.10, 0]),
        np.array([0.0, 1.30, 0]),
        fill_color=color, fill_opacity=1.0, stroke_width=0,
    )
    crossguard = Rectangle(
        width=0.55, height=0.08,
        fill_color=color, fill_opacity=1.0, stroke_width=0,
    )
    crossguard.move_to(np.array([0, -0.92, 0]))
    handle = Rectangle(
        width=0.10, height=0.32,
        fill_color=color, fill_opacity=1.0, stroke_width=0,
    )
    handle.move_to(np.array([0, -1.13, 0]))
    pommel = RegularPolygon(
        n=6, color=color, fill_color=color,
        fill_opacity=1.0, stroke_width=0,
    )
    pommel.scale(0.10)
    pommel.move_to(np.array([0, -1.34, 0]))
    sword = VGroup(blade, tip, crossguard, handle, pommel)
    sword.scale(scale)
    return sword


def _stat_card(title: str, rows: list[tuple[str, str]]) -> VGroup:
    """Tabletop-style stat block: title + ruled separator + key/value rows.

    Used in Page 0 to introduce the "rules" idea — sword + stats card
    grounds "behave according to the rules" visually before the
    abstraction conversation starts.
    """
    title_t = Text(title, font="Georgia", weight="BOLD",
                   font_size=18, color=Paper.ink)
    sep_w = 1.6
    sep = Line(
        np.array([-sep_w / 2, 0, 0]),
        np.array([sep_w / 2, 0, 0]),
        stroke_color=Paper.ink_soft, stroke_width=0.8,
    )
    row_mobs = []
    for k, v in rows:
        # Single Text per row keeps spacing predictable. Pad the key so
        # the value column lines up — monospace.
        row_mobs.append(Text(
            f"{k:<10}{v}",
            font=theme.MONO_FALLBACK, font_size=12, color=Paper.ink,
        ))
    rows_grp = VGroup(*row_mobs).arrange(DOWN, buff=0.10, aligned_edge=LEFT)
    content = VGroup(title_t, sep, rows_grp).arrange(
        DOWN, buff=0.16, aligned_edge=LEFT,
    )
    pad_x, pad_y = 0.30, 0.25
    bg = RoundedRectangle(
        width=content.width + 2 * pad_x,
        height=content.height + 2 * pad_y,
        corner_radius=0.10,
        fill_color=Paper.bg, fill_opacity=0.95,
        stroke_color=Paper.ink_soft, stroke_width=1.0,
    )
    bg.move_to(content.get_center())
    return VGroup(bg, content)


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

        # Page 0 carries Malte's 31s personal voiceover: a quiet TTRPG
        # anchor (d20 emblem + name byline), the three-tool chip reveal
        # synced to "structured output / tool calling / xml tagging",
        # the merge-into-? unification beat, and the orate wordmark
        # reveal that hands off to Page 2.
        self._page0_personal_intro()
        # Page 1 (distribution-shaping motivation) is commented out for
        # the submission cut — Page 0 + the personal voiceover do the
        # same job. Re-enable if the intro changes scope.
        # self._page1_distribution_shaping()
        self._page2_structured_output_and_logic()
        self._page3_algebra_predicate_and_contrast()
        self._page4_dnd_session_and_combat_regrammar()
        self._page5_meta_authorship_and_close()

    # ======================================================================
    # PAGE 0 — personal intro (31s voiceover companion)
    # ======================================================================

    def _page0_personal_intro(self):
        """Visual companion to Malte's 31s voiceover.

        Voiceover beat-by-beat (annotated by Malte after a test record):
          0.0  "I'm in Sweden, my name is Malte"
          2.0  "and I'm making a tabletop RPG simulator …with my friend"
          6.0   (small breath)
          7.0  "We really want to make the LLMs in the game"
         10.0  "behave and act according to the rules"
         12.0  "and the project constantly became a tradeoff between"
         14.0  "structured output,"
         15.5  "tool calling, and"
         17.0  "xml tagging."
         18.0  "It bogs up our harness, and it all seemed like it
                should really just be"
         21.5  "unified under one abstraction."
         23.0  "So in this hackathon, Claude and Me have built Orate,
                a programmatic grammar decoding library for LLM
                inference."
         31.0  end

        Visual phases land on those beats:
          0–4   d20 emblem + byline appear; quiet
          4–7   two swords fly across; one settles centre-left
          7–10  pause on the settled sword
         10–12 longsword stat card materialises beside the sword —
                "behave and act according to the rules"
         12–14 sword + card fade; transition into chip lane
         14–18 three chips appear in turn ("structured output",
                "tool calling", "xml tagging")
         18–23 chips slide together and merge into a single `?` chip
         23–31 `?` morphs into the orate wordmark + subtitle, holds
        """
        # ----- 0–4s: byline + d20 anchor ----------------------------
        d20 = RegularPolygon(
            n=6, color=Paper.ink_soft, stroke_width=1.5,
        )
        d20.scale(0.42)
        d20_n = Text("20", font="Georgia", weight="BOLD",
                     font_size=15, color=Paper.ink_soft)
        d20_n.move_to(d20.get_center())
        d20_grp = VGroup(d20, d20_n)
        d20_grp.move_to(np.array([5.5, -3.05, 0]))

        byline = Text("Malte · Sweden",
                      font=theme.SANS_FALLBACK,
                      font_size=11, color=Paper.mute)
        byline.move_to(np.array([-5.7, 3.5, 0]), aligned_edge=LEFT)

        self.play(
            FadeIn(d20_grp, run_time=0.7),
            FadeIn(byline, run_time=0.7),
        )
        # Hold so swords start flying at ~2.0s, on "tabletop rpg
        # simulator" (the 2s mark in Malte's voiceover).
        self.wait(1.3)

        # ----- 4–7s: two swords fly toward centre and clash ---------
        # Sword A enters from left, B from right. They arc inward and
        # cross at the screen centre; a small flash marks the impact.
        # The crossed-swords tableau holds for ~3s; at t≈10s we pause
        # abruptly and surface the longsword stat card next to A.
        sword_a = _make_sword(scale=0.85, color=Paper.ink)
        sword_a.rotate(np.pi / 6)            # tip angled up-right
        sword_a.move_to(np.array([-7.5, -0.5, 0]))

        sword_b = _make_sword(scale=0.85, color=Paper.ink_soft)
        sword_b.rotate(-np.pi / 6 + np.pi)   # tip angled down-left (mirrored)
        sword_b.move_to(np.array([7.5, 0.5, 0]))

        self.add(sword_a, sword_b)
        # Both swords swing toward centre on opposite arcs; they meet
        # in the middle with their blades crossed. Pull the left sword
        # right by 1/3 of its width and the right sword left by 1/4
        # of its width so the X they form sits closer to centre.
        sword_a_w = sword_a.width
        sword_b_w = sword_b.width
        clash_a = np.array([-0.55 + sword_a_w / 3, -0.05, 0])
        clash_b = np.array([0.55 - sword_b_w / 4, 0.05, 0])
        self.play(
            sword_a.animate.move_to(clash_a).rotate(np.pi / 12),
            sword_b.animate.move_to(clash_b).rotate(-np.pi / 12),
            run_time=1.6,
            rate_func=smooth,
        )

        # Tiny clash spark — quick fade, accent colour. Visual rhyme
        # with the strikes used in Page 5's mask flash.
        spark = theme.starburst(radius=0.20, color=Paper.accent)
        spark.move_to(ORIGIN)
        self.play(FadeIn(spark, run_time=0.12))
        self.play(FadeOut(spark, run_time=0.18))

        # Brief recoil (a few pixels) so the tableau reads like a real
        # impact rather than two swords pasted on top of each other.
        self.play(
            sword_a.animate.shift(LEFT * 0.10),
            sword_b.animate.shift(RIGHT * 0.10),
            run_time=0.25,
        )

        # Hold the crossed-swords tableau through "We really want to
        # make the LLMs in the game…" — stat card lands at ~10s on
        # "behave and act according to the rules". The recoil ends
        # near ~4.15s; this fills until ~10s.
        self.wait(5.85)

        # ----- 10s: ABRUPT freeze + info card on sword A ------------
        # The "abrupt" feel comes from a fast simultaneous reveal of
        # the callout line + stat card, plus a soft accent halo around
        # sword A so the viewer's eye locks on to it.
        halo = RoundedRectangle(
            width=sword_a.width + 0.25,
            height=sword_a.height + 0.25,
            corner_radius=0.12,
            fill_opacity=0,
            stroke_color=Paper.accent, stroke_width=1.4,
        )
        halo.move_to(sword_a.get_center())
        # Cumulative rotation applied to sword_a above: π/6 (initial)
        # + π/12 (clash-arc). Halo follows so it looks like a frame
        # snapped to the blade rather than a floating box.
        halo.rotate(np.pi / 6 + np.pi / 12)

        stat_card = _stat_card(
            title="longsword",
            rows=[
                ("damage", "1d8"),
                ("type",   "slashing"),
                ("range",  "melee"),
                ("weight", "3 lb"),
            ],
        )
        # Park the stat card to the LEFT of sword A, leaving sword B
        # unobscured to the right of frame.
        stat_card.move_to(np.array([-3.6, -0.05, 0]))
        callout = Line(
            stat_card.get_right() + np.array([0.05, 0, 0]),
            sword_a.get_left() + np.array([-0.05, 0, 0]),
            stroke_color=Paper.accent, stroke_width=1.2,
        )
        self.play(
            FadeIn(halo, run_time=0.25),
            FadeIn(callout, run_time=0.25),
            FadeIn(stat_card, shift=RIGHT * 0.10, run_time=0.45),
        )
        # Hold from ~10.45s to ~12s — voiceover lands "behave and act
        # according to the rules" during this dwell.
        self.wait(1.55)

        # ----- 12–14s: clear the tabletop set, prepare chip lane ----
        # Voiceover: "and the project constantly became a tradeoff
        # between using…" — we clear at 12s and breathe until 14s,
        # when the first chip lands on "structured output".
        self.play(
            FadeOut(VGroup(sword_a, sword_b, halo,
                           callout, stat_card),
                    run_time=0.7),
        )
        self.wait(1.3)

        # ----- 14–18s: three chips appear -----------------------------
        chips = VGroup(
            _chip("structured output", font_size=14),
            _chip("tool calling", font_size=14),
            _chip("xml tagging", font_size=14),
        ).arrange(RIGHT, buff=0.45)
        chips.move_to(ORIGIN)
        # 14.0 → "structured output"
        self.play(FadeIn(chips[0], shift=UP * 0.08, run_time=0.4))
        self.wait(1.1)
        # 15.5 → "tool calling"
        self.play(FadeIn(chips[1], shift=UP * 0.08, run_time=0.4))
        self.wait(1.1)
        # 17.0 → "xml tagging"
        self.play(FadeIn(chips[2], shift=UP * 0.08, run_time=0.4))
        # Hold until ~18s ("It bogs up our harness…").
        self.wait(0.6)

        # ----- 18–23s: chips merge into ? ----------------------------
        # Hold the trio while voiceover sets up the unification idea
        # — "It bogs up our harness, and it all seemed like it should
        # really just be…" runs from ~18s to ~21.5s.
        self.wait(2.4)
        # ~21.5 → "unified under one abstraction": chips slide together.
        q_chip = _chip("?", font_size=24)
        q_chip.set_stroke(Paper.accent, width=1.8)
        q_chip.move_to(ORIGIN)
        self.play(
            *[c.animate.move_to(ORIGIN).set_opacity(0.0) for c in chips],
            FadeIn(q_chip, scale=0.85),
            run_time=1.1,
        )
        # Linger on `?` until ~26s — that's when Malte says "Orate"
        # and the wordmark transition should land. The 23s mark is
        # the start of the closing sentence ("So in this hackathon,
        # Claude and Me have built…"); the wordmark waits for the
        # actual word.
        self.wait(4.5)

        # ----- 23–31s: orate wordmark + subtitle ---------------------
        wordmark = Text(_spaced("orate", n=2),
                        font="Georgia", slant="ITALIC", weight="BOLD",
                        font_size=64, color=Paper.ink)
        subtitle = Text(
            "programmatic grammar decoding for LLM inference",
            font="Georgia", slant="ITALIC",
            font_size=18, color=Paper.ink_soft,
        )
        title_grp = VGroup(wordmark, subtitle).arrange(DOWN, buff=0.35)
        title_grp.move_to(ORIGIN)

        self.play(
            FadeOut(q_chip, scale=1.2, run_time=0.4),
            FadeIn(wordmark, shift=UP * 0.12, run_time=0.7),
        )
        self.play(FadeIn(subtitle, shift=UP * 0.05, run_time=0.5))
        # Hold through "...programmatic grammar decoding library for
        # LLM inference" — voiceover ends ~31s. Wordmark visible from
        # ~26.7s onward; this dwell carries it to the end of the
        # voiceover plus a small buffer.
        self.wait(4.0)

        # Cleanup; Page 2 has its own composition.
        self.play(
            FadeOut(VGroup(title_grp, d20_grp, byline), run_time=0.45),
        )

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
        # If Page 1 ran, it parked the so_chip ready to dock. If not
        # (current submission cut), materialise it here from scratch
        # so the dock animation still has something to transform.
        so_chip = getattr(self, "_so_chip", None)
        if so_chip is None:
            so_chip = _chip("structured output")
            so_chip.move_to(np.array([0.0, 1.0, 0]))
            self.play(FadeIn(so_chip, shift=UP * 0.05, run_time=0.35))

        # Beat 2.A — dock structured output → top-left header. Drawer drops.
        target = _chip("structured output", font_size=16)
        target.move_to(np.array([-3.8, 3.55, 0]))
        target.set_stroke(Paper.accent, width=1.8)
        self.play(Transform(so_chip, target, run_time=0.6))

        left_drawer = _paper_card(width=6.4, height=4.0)
        left_drawer.move_to(np.array([-3.55, 0.4, 0]))
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
            BOLD in `type_color`; the tail (e.g. "[bard, cleric, rogue]")
            renders in muted text BELOW the type label so it doesn't
            overflow the schema card. Pacing is ~30% slower than the
            original feedback round.
            """
            self.play(FadeIn(schema[idx], shift=LEFT * 0.1, run_time=0.4))
            arrow = Text("← must be a", font=theme.MONO_FALLBACK,
                         font_size=12, color=Paper.ink_soft)
            type_t = Text(_spaced(type_label, n=1),
                          font=theme.MONO_FALLBACK, weight="BOLD",
                          font_size=13, color=type_color)
            top_row = VGroup(arrow, type_t).arrange(RIGHT, buff=0.18)
            if tail:
                tail_t = Text(tail.strip(), font=theme.MONO_FALLBACK,
                              font_size=12, color=Paper.ink_soft)
                # Stack tail under the top row, left-aligned to "← must"
                cap_grp = VGroup(top_row, tail_t).arrange(
                    DOWN, aligned_edge=LEFT, buff=0.06,
                )
            else:
                cap_grp = top_row
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

        # Highlight "typing" so it visually rhymes with "the logic"
        # in the question line below — establishes the two sides of
        # the same coin (typing ↔ the logic) before the dock-up.
        pre_left = Text(
            "With structured output we've put",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink_soft,
        )
        pre_typing = Text(
            _spaced("typing"),
            font="Georgia", weight="BOLD", font_size=18,
            color=Paper.accent,
        )
        bridge_pre = VGroup(pre_left, pre_typing).arrange(
            RIGHT, buff=0.22,
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

        # Beat 2.D — fade everything except "the logic"; dock it to
        # the upper-right header position. Critically, q_logic is the
        # survivor and KEEPS its style (Georgia BOLD italic, _spaced,
        # Paper.accent) all the way to the top — we wrap a pill around
        # it instead of swapping for a generic chip.
        dock_pos = np.array([3.8, 3.55, 0])
        # We DO NOT include `q_logic` in the fade-out — it's the survivor.
        self.play(
            FadeOut(VGroup(bridge_pre, bridge_pre2, q_left, q_right),
                    run_time=0.45),
        )
        # Survivor flies up + scales down. Style preserved end-to-end.
        self.play(
            q_logic.animate.move_to(dock_pos).scale(0.8),
            run_time=0.7,
        )
        # Build a pill around the docked text — mirrors so_chip's
        # geometry, but the label is the un-changed survivor itself.
        pad_x, pad_y = 0.32, 0.14
        logic_pill = RoundedRectangle(
            width=q_logic.width + pad_x * 2,
            height=q_logic.height + pad_y * 2,
            corner_radius=0.22,
            fill_color=Paper.card, fill_opacity=1.0,
            stroke_color=Paper.accent, stroke_width=1.0,
        )
        logic_pill.move_to(q_logic.get_center())
        # Force pill to render BEHIND q_logic from frame 1 of the fade
        # so the text never flickers under a rising pill fill. z_index
        # is renderer-level and beats add-order.
        logic_pill.set_z_index(-1)
        q_logic.set_z_index(1)
        self.play(FadeIn(logic_pill, run_time=0.35))
        logic_chip = VGroup(logic_pill, q_logic)

        right_drawer = _paper_card(width=6.4, height=4.0)
        right_drawer.move_to(np.array([3.55, 0.4, 0]))
        right_drawer.shift(UP * 0.6)
        self.play(
            right_drawer.animate.shift(DOWN * 0.6),
            FadeIn(right_drawer, run_time=0.45),
            run_time=0.45,
        )

        # Beat 2.E — focus shift: structured output → the logic.
        # Dim the upper-left "structured output" chip (it's no longer the
        # focus); the logic chip stays accented. This is the moment the
        # bolding swaps.
        so_dim_target = _chip("structured output", font_size=16,
                              fg=Paper.ink_soft, stroke=Paper.grid)
        so_dim_target.move_to(so_chip.get_center())
        self.play(Transform(so_chip, so_dim_target, run_time=0.4))

        # Beat 2.F — draw the book_meeting @program in the right drawer.
        # Picked for "the logic" framing — pure-logic example, no D&D
        # scaffolding, leads with where= so it clearly anchors the
        # "logic" framing. Only TWO callouts, both impossible in JSON
        # Schema:
        #   - named-predicate where= (in_business_hours)
        #   - cross-field equation closing over a parameter
        #
        # SYNTAX HIGHLIGHTED:
        #   @program (Paper.accent)
        #   def / yield / return / lambda (Paper.accent_soft)
        #   function name (Terminal.amber)
        #   gen.X attribute (Terminal.amber)
        #   strings (Paper.good), numbers (Terminal.blue)
        right_anchor = np.array([0.6, 2.0, 0])
        ACC = Paper.accent
        KW  = Paper.accent_soft   # keywords
        FN  = Terminal.amber      # function / attribute names
        ID  = Paper.ink           # identifiers + neutral
        DIM = Paper.ink_soft      # dimmed
        NUM = Terminal.blue
        STR = Paper.good
        prog_rich = [
            [("business_hours = ", ID), ("range", FN), ("(", DIM),
             ("9", NUM), (", ", DIM), ("18", NUM), (")", DIM),
             ("   # 9..17 inclusive", DIM)],                             # 0
            [],                                                          # blank row (visual spacer)
            [("@program", ACC)],                                         # 1
            [("def ", KW), ("book_meeting", FN), ("(", DIM),
             ("duration_h", ID), (": ", DIM), ("int", FN),
             ("):", DIM)],                                               # 2
            [("    start = ", ID), ("yield ", KW),
             ("gen.datetime", FN), ("(", DIM),
             ("where=", ACC), ("lambda ", KW),
             ("d", ID), (":", DIM)],                                     # 3
            [("        d.hour ", ID), ("in ", KW),
             ("business_hours", ID), (")", DIM)],                        # 4
            [("    end   = ", ID), ("yield ", KW),
             ("gen.datetime", FN), ("(", DIM),
             ("where=", ACC), ("lambda ", KW),
             ("e", ID), (":", DIM)],                                     # 5
            [("        e", ID), (" ", DIM), ("-", DIM), (" ", DIM),
             ("start", ID), (" ", DIM), ("==", DIM), (" ", DIM),
             ("timedelta", FN), ("(", DIM),
             ("hours", ID), ("=", DIM), ("duration_h", ID),
             ("))", DIM)],                                               # 6
            [("    return ", KW),
             ('{"start": start, "end": end}', ID)],                      # 7
        ]
        prog = _rich_block(prog_rich, anchor=right_anchor,
                           line_height=0.34, font_size=12)

        # Two callouts:
        #   start line — predicate constraint, callout sits at the def-line
        #     y-coord (in vacant space) with an arrow pointing DOWN to the
        #     `where=lambda d: d.hour in business_hours` portion.
        #   end lines — cross-field equation (THE moment of the page).
        #     The lambda-body wraps so the whole thing fits the drawer.
        prog_lines = list(prog)
        # Visual layout (after empty-row spacer is dropped by _rich_block):
        #   [0] business_hours = range(9, 18)   # 9..17 inclusive
        #   [1] @program
        #   [2] def book_meeting(duration_h: int):
        #   [3]     start = yield gen.datetime(where=lambda d:
        #   [4]         d.hour in business_hours)
        #   [5]     end   = yield gen.datetime(where=lambda e:
        #   [6]         e - start == timedelta(hours=duration_h))
        #   [7]     return {"start": start, "end": end}

        # Stage 1: business_hours definition (alone) — establish the range.
        self.play(FadeIn(prog_lines[0], shift=LEFT * 0.08, run_time=0.45))
        self.wait(0.25)

        # Stage 2: @program + def signature.
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in prog_lines[1:3]],
            lag_ratio=0.18, run_time=0.55,
        ))
        self.wait(0.3)

        frame_right = 7.0

        # Lines 3 + 4 — start = yield gen.datetime(where=lambda d:
        #                   d.hour in business_hours)
        # Reveal both together so the wrapped predicate reads as one.
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in prog_lines[3:5]],
            lag_ratio=0.20, run_time=0.6,
        ))

        # Predicate-constraint callout: just the label "predicate
        # constraint" at the def-line y-coord. Diagonal down-arrow from
        # the BOTTOM-LEFT of the label to the top of the "where="
        # token on the start line, with "where=" underlined while the
        # callout holds.
        callout1 = Text(
            "predicate constraint",
            font="Georgia", slant="ITALIC",
            font_size=12, color=Paper.accent,
        )
        # Anchor at def-line y, in vacant space to the right of the def
        # signature.
        callout1.next_to(prog_lines[2], RIGHT, buff=0.30)
        if callout1.get_right()[0] > frame_right:
            callout1.shift(LEFT * (callout1.get_right()[0] - frame_right))

        # Find the "where=" token on the start line (prog_lines[3]).
        # Token order in the rich-line VGroup mirrors the prog_rich
        # entry: index 4 is "where=".
        where_mob = prog_lines[3][4]

        arrow_start = np.array([callout1.get_left()[0], callout1.get_bottom()[1], 0]) + np.array([0.05, -0.04, 0])
        arrow_end = where_mob.get_top() + np.array([0, 0.04, 0])
        callout1_arrow = Line(
            arrow_start, arrow_end,
            color=Paper.accent, stroke_width=1.6,
        )
        # Arrowhead — V-tip oriented along the arrow's direction.
        _dir = arrow_end - arrow_start
        _dir_norm = _dir / np.linalg.norm(_dir)
        _perp = np.array([-_dir_norm[1], _dir_norm[0], 0])
        _head_len = 0.10
        _head_spread = 0.07
        head_l = Line(
            arrow_end,
            arrow_end - _dir_norm * _head_len + _perp * _head_spread,
            color=Paper.accent, stroke_width=1.6,
        )
        head_r = Line(
            arrow_end,
            arrow_end - _dir_norm * _head_len - _perp * _head_spread,
            color=Paper.accent, stroke_width=1.6,
        )
        # Underline of "where=" token.
        where_underline = Line(
            np.array([where_mob.get_left()[0],
                      where_mob.get_bottom()[1] - 0.04, 0]),
            np.array([where_mob.get_right()[0],
                      where_mob.get_bottom()[1] - 0.04, 0]),
            color=Paper.accent, stroke_width=1.6,
        )
        callout1_grp = VGroup(callout1, callout1_arrow, head_l, head_r,
                              where_underline)
        self.play(FadeIn(callout1_grp, shift=DOWN * 0.05, run_time=0.3))
        self.wait(2.4)
        self.play(FadeOut(callout1_grp, run_time=0.25))

        # Lines 5 + 6 — end = yield gen.datetime(where=lambda e:
        #                   e - start == timedelta(hours=duration_h))
        # Reveal both together so the wrapped equation reads as one.
        self.play(LaggedStart(
            *[FadeIn(ln, shift=LEFT * 0.08) for ln in prog_lines[5:7]],
            lag_ratio=0.20, run_time=0.65,
        ))
        # Cross-field equation callout: positioned just LEFT of the
        # equation line (prog_lines[6]), at the same y. Sits in the
        # gap-and-drawer-edge area; a drawer-card-colored background
        # panel sits behind the text so it reads cleanly even where it
        # overlaps the right drawer's fill. Short horizontal arrow
        # lands on "e - start ==" without crossing any program text.
        callout2 = Text(
            "cross-field equation",
            font="Georgia", slant="ITALIC",
            font_size=12, color=Paper.accent,
        )
        # Target: "e - start == " token on prog_lines[6] (index 0 —
        # the first non-empty token of that row).
        equation_mob = prog_lines[6][0]
        equation_y = equation_mob.get_center()[1]
        # Position: callout right edge sits ~0.18u left of the
        # equation token's left edge — minus the width of the word
        # "equation" so the label hangs well clear of the equation
        # token (with a longer arrow connecting the two).
        callout2.move_to(np.array([0, equation_y, 0]))
        _equation_word_w = Text(
            "equation", font="Georgia", slant="ITALIC", font_size=12,
        ).width
        target_right_x = (
            equation_mob.get_left()[0] - 0.18 - _equation_word_w
        )
        callout2.shift(
            np.array([target_right_x - callout2.get_right()[0], 0, 0])
        )
        # Background panel (drawer-card color) so the text reads cleanly
        # against the right drawer's fill below.
        callout2_bg = RoundedRectangle(
            width=callout2.width + 0.22,
            height=callout2.height + 0.14,
            corner_radius=0.06,
            fill_color=Paper.card,
            fill_opacity=1.0,
            stroke_opacity=0,
        )
        callout2_bg.move_to(callout2.get_center())

        # Short horizontal arrow: bg right edge → equation left edge.
        arrow2_start = np.array(
            [callout2_bg.get_right()[0] + 0.04, equation_y, 0]
        )
        arrow2_end = np.array(
            [equation_mob.get_left()[0] - 0.06, equation_y, 0]
        )
        callout2_arrow = Line(
            arrow2_start, arrow2_end,
            color=Paper.accent, stroke_width=1.6,
        )
        # Arrowhead: V-tip pointing right at the equation token.
        _dir2 = arrow2_end - arrow2_start
        _dir2_norm = _dir2 / np.linalg.norm(_dir2)
        _perp2 = np.array([-_dir2_norm[1], _dir2_norm[0], 0])
        head_l2 = Line(
            arrow2_end,
            arrow2_end - _dir2_norm * 0.10 + _perp2 * 0.07,
            color=Paper.accent, stroke_width=1.6,
        )
        head_r2 = Line(
            arrow2_end,
            arrow2_end - _dir2_norm * 0.10 - _perp2 * 0.07,
            color=Paper.accent, stroke_width=1.6,
        )
        # Order matters: bg first (back), then text + arrow on top.
        callout2_grp = VGroup(callout2_bg, callout2, callout2_arrow,
                              head_l2, head_r2)
        self.play(FadeIn(callout2_grp, shift=RIGHT * 0.05, run_time=0.35))
        self.wait(3.0)
        self.play(FadeOut(callout2_grp, run_time=0.3))

        # Line 7 — return {...}
        self.play(FadeIn(prog_lines[7], shift=LEFT * 0.08, run_time=0.4))
        self.wait(0.6)

        # Punchline caption
        punch = Text(
            "Types, tool calls, control flow — same yield stream.",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.ink,
        )
        punch.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(punch, shift=UP * 0.06, run_time=0.5))
        self.wait(3.0)

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
        self.wait(0.4)

        # Punchline caption — short, single-line, italic terracotta.
        # Reveals later, paired with the algebra-step highlight box on
        # the right column (see Beat 3.C). Sits where the longer 3-line
        # narration used to be, but reads as a single punchline.
        punchline = Text(
            "literally can't mess up",
            font="Georgia", slant="ITALIC", font_size=18,
            color=Paper.accent,
        )
        punchline.next_to(intro_grp, DOWN, buff=0.28)
        # NOTE: punchline reveal happens later (after the constrained
        # trace finishes drawing). Just instantiate here.

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

        # Highlight the where= line specifically.
        # Tighter vertical padding — earlier 1.18 height left a gap that
        # almost touched the lines above and below; ~0.95 hugs the 3-line
        # where= block more cleanly without clipping into the
        # `gen.string(` line above or `return {...}` line below.
        where_box = RoundedRectangle(
            width=5.4, height=0.95, corner_radius=0.10,
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
        # All three right-side labels — "Solve:", "free text",
        # "under @algebra_step" — anchor to the SAME left x-coordinate
        # so the column reads aligned. Earlier each used a different
        # alignment (some via move_to center, some via align_to LEFT).
        right_left_x = 1.5  # shared LEFT-edge for the right column
        anchor_r_top = 1.5
        prob = Text("Solve:   3x + 5 = 14",
                    font=theme.MONO_FALLBACK, font_size=18,
                    color=Paper.ink_soft)
        prob.shift(np.array([0, anchor_r_top, 0]) - prob.get_center()
                   * np.array([0, 1, 0]))
        prob.align_to(np.array([right_left_x, 0, 0]), LEFT)
        self.play(FadeIn(prob, run_time=0.3))

        # Two stacked outputs
        free_label = Text("Normal freetext, qwen 2.5 7b",
                          font=theme.SANS_FALLBACK,
                          font_size=12, color=Paper.ink_soft)
        free_label.move_to(np.array([0, anchor_r_top - 0.6, 0]))
        free_label.align_to(np.array([right_left_x, 0, 0]), LEFT)
        free_lines = [
            ("3x = 14 - 5", Paper.ink),
            ("3x = 9", Paper.ink),
            ("x = 9 / 3 = 4", Paper.bad),
        ]
        free_block = VGroup()
        for i, (t, c) in enumerate(free_lines):
            ln = _code_text(t, c, font_size=14)
            ln.move_to(np.array([0, anchor_r_top - 1.05 - 0.36 * i, 0]))
            ln.align_to(np.array([right_left_x, 0, 0]), LEFT)
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
        cons_label.move_to(np.array([0, anchor_r_top - 2.5, 0]))
        cons_label.align_to(np.array([right_left_x, 0, 0]), LEFT)
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
        for i, tokens in enumerate(cons_specs):
            ln = _rich_line(tokens, font_size=12)
            ln.align_to(np.array([right_left_x, 0, 0]), LEFT)
            ln.move_to(np.array([
                ln.get_center()[0],
                anchor_r_top - 2.92 - 0.34 * i,
                0,
            ]))
            ln.align_to(np.array([right_left_x, 0, 0]), LEFT)
            cons_block.add(ln)
        self.play(FadeIn(cons_label, run_time=0.3))
        self.play(LaggedStart(
            *[FadeIn(l, shift=LEFT * 0.06) for l in cons_block],
            lag_ratio=0.22, run_time=1.4,
        ))
        self.wait(0.6)

        # Orange highlight box around the algebra-step chain. Visual
        # rhyme with the where_box on Page 2 / Beat 3.B (same stroke
        # color, same corner radius, same stroke width). Singular box
        # around the WHOLE chain — "this whole sequence is correct by
        # construction" — paired with the "literally can't mess up"
        # punchline fading in concurrently.
        cons_top = cons_block[0].get_top()[1]
        cons_bot = cons_block[-1].get_bottom()[1]
        cons_left = min(ln.get_left()[0] for ln in cons_block)
        cons_right = max(ln.get_right()[0] for ln in cons_block)
        algebra_box_buff = 0.12
        cons_box = RoundedRectangle(
            width=(cons_right - cons_left) + 2 * algebra_box_buff,
            height=(cons_top - cons_bot) + 2 * algebra_box_buff,
            corner_radius=0.10,
            fill_color=Paper.accent, fill_opacity=0.10,
            stroke_color=Paper.accent, stroke_width=1.4,
        )
        cons_box.move_to(np.array([
            (cons_left + cons_right) / 2,
            (cons_top + cons_bot) / 2,
            0,
        ]))
        self.play(FadeIn(cons_box, run_time=0.45),
                  FadeIn(punchline, shift=UP * 0.06, run_time=0.45))
        self.wait(2.0)

        # Beat 3.D — benchmark line + same weights, different gate.
        # Source: bench/results/legal_steps_2026-04-26_1759.md
        # (Qwen2.5-7B-Instruct-Q4_K_M, T=0 by default with Session-
        # level escalation on rejection; 10 algebra problems × 2
        # modes). The single constrained miss is eq_negative — model
        # escapes the locked `x = -2` once T ramps up but then
        # meanders through valid no-progress steps without committing
        # to @done. A clean reminder: step correctness ≠ solution.
        bench = Text(
            "free-text 5/10   ·   constrained 9/10   ·   16 illegal-step rejections",
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
                                  punchline,
                                  prob, free_label, free_block, free_x_mark,
                                  cons_label, cons_block, cons_box,
                                  bench, gate),
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
        roll_line = emit('@roll("persuasion", dc=14)', color=Paper.ink)
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
        self.wait(0.6)

        # ====================================================================
        # Unification callout — the @roll line is structurally a
        # `gen.tool` yield with ends_turn=True. Same syntax as @narrate
        # / @meta, just with the flag flipped. We pause, point at the
        # @roll line, and surface the thesis.
        # ====================================================================
        unify_arrow = Text("←", font=theme.MONO_FALLBACK,
                           font_size=18, color=Paper.accent, weight="BOLD")
        unify_arrow.next_to(roll_line, RIGHT, buff=0.18)
        unify_l1 = Text(
            _spaced("structured output & tools"),
            font="Georgia", weight="BOLD", font_size=15,
            color=Paper.accent,
        )
        unify_l2 = Text(
            "same yield syntax —",
            font="Georgia", slant="ITALIC", font_size=14,
            color=Paper.ink,
        )
        unify_l3 = Text(
            "only `ends_turn=True` distinguishes them.",
            font="Georgia", slant="ITALIC", font_size=14,
            color=Paper.ink_soft,
        )
        unify_grp = VGroup(unify_l1, unify_l2, unify_l3).arrange(
            DOWN, aligned_edge=LEFT, buff=0.08,
        )
        unify_grp.next_to(unify_arrow, RIGHT, buff=0.2)
        # Shift the whole callout (arrow + 3 text lines) DOWN so the
        # top line doesn't overlap the @narrate continuation line above
        # the @roll line. The arrow tail/head still points cleanly at
        # the @roll line's right edge from below.
        unify_shift = DOWN * 0.3
        unify_arrow.shift(unify_shift)
        unify_grp.shift(unify_shift)
        # Flash: fade-in arrow + label, hold ~2s, fade out.
        self.play(
            FadeIn(unify_arrow, shift=LEFT * 0.1, run_time=0.4),
            FadeIn(unify_grp, shift=LEFT * 0.06, run_time=0.5),
        )
        self.wait(2.2)
        self.play(FadeOut(VGroup(unify_arrow, unify_grp), run_time=0.4))
        self.wait(0.4)

        # Meta emission — subbar bolds "meta".
        set_subbar(active_idx=2, run_time=0.3)
        meta_line = emit(
            '@meta("Haha — a 1. Sorry, won\'t cut it.")',
            color=Paper.accent_soft, hold=1.4,
        )

        # Brief caption — placed to the RIGHT of the @meta emission,
        # at the same y, so the eye doesn't have to dart to the bottom
        # of the page. The trace lives on the left half; the caption
        # reads as a sidebar gloss to that exact line.
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
        # Centered horizontally on the screen, at the y of @meta.
        meta_grp.move_to(np.array([0, meta_line.get_center()[1], 0]))
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

        # Third reveal — same visual weight as "Predicates are Python."
        # so all three lines stack as a single moment. Wrapped onto
        # two visual rows so the line fits within the frame.
        bonus_l1 = Text(
            _spaced("Fields can reference any"),
            font="Georgia", slant="ITALIC", weight="BOLD",
            font_size=22, color=Paper.accent,
        )
        bonus_l2 = Text(
            _spaced("Python program state."),
            font="Georgia", slant="ITALIC", weight="BOLD",
            font_size=22, color=Paper.accent,
        )
        bonus = VGroup(bonus_l1, bonus_l2).arrange(DOWN, buff=0.08)
        bonus.next_to(ours, DOWN, buff=0.4)
        self.play(FadeIn(bonus, shift=UP * 0.06, run_time=0.5))
        self.wait(4.8)

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
        self.wait(2.9)

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
        self.wait(4.0)
        self.play(FadeOut(VGroup(bridge, question_grp), run_time=0.45))

        # ====================================================================
        # Grammar selector — same idiom as Page 4 (D&D). Lives at the top
        # throughout Page 5; reflects the current outer-grammar alternation
        # the model is decoding under. Starts as [done | make_new_program],
        # grows by one alternative when @_factor_1147 registers.
        #
        # Both active and inactive tabs use BOLD weight so the bar's width
        # doesn't shift when active state moves — only color and underline
        # differentiate. Avoids the flimsy resize feel.
        # ====================================================================
        SEL_Y = 2.85

        def _tab_p5(name: str, active: bool, font_size: int = 13) -> VGroup:
            color = Paper.accent if active else Paper.ink_soft
            t = Text(name, font=theme.MONO_FALLBACK,
                     weight="BOLD",
                     font_size=font_size, color=color)
            grp = VGroup(t)
            if active:
                ul = Line(
                    t.get_corner(np.array([-1, -1, 0])) + np.array([0, -0.06, 0]),
                    t.get_corner(np.array([1, -1, 0])) + np.array([0, -0.06, 0]),
                    stroke_color=Paper.accent, stroke_width=2.0,
                )
                grp.add(ul)
            return grp

        def _selector(names: list[str], active_idx: int | None,
                      font_size: int = 13) -> VGroup:
            """Render `[ a | b | c ]`; if active_idx is None, no tab is hot."""
            lb = Text("[", font=theme.MONO_FALLBACK, font_size=font_size,
                      color=Paper.ink_soft)
            rb = Text("]", font=theme.MONO_FALLBACK, font_size=font_size,
                      color=Paper.ink_soft)
            row = VGroup(lb)
            for i, n in enumerate(names):
                row.add(_tab_p5(n, active=(i == active_idx),
                                 font_size=font_size))
                if i < len(names) - 1:
                    row.add(Text("|", font=theme.MONO_FALLBACK,
                                  font_size=font_size, color=Paper.mute))
            row.add(rb)
            row.arrange(RIGHT, buff=0.18)
            return row

        # "Grammar" label sits to the left of the bracketed alternatives,
        # in the same style as Page 4's "Many grammars" outer label.
        sel_label = Text("Grammar", font=theme.SANS_FALLBACK,
                         font_size=14, color=Paper.ink_soft)
        sel_label.move_to(np.array([-3.7, SEL_Y, 0]))

        sel_state = {"current": None, "names": ["done", "make_new_program"],
                     "active": None}

        def set_selector(names: list[str] | None = None,
                         active_idx: int | None = None,
                         run_time: float = 0.35) -> VGroup:
            new_names = names if names is not None else sel_state["names"]
            new_bar = _selector(new_names, active_idx)
            # Anchor the bar's left edge to a fixed x so growth extends to
            # the right rather than reflowing around the screen centre.
            new_bar.move_to(np.array([0.0, SEL_Y, 0]))
            new_bar.align_to(np.array([-2.7, 0, 0]), LEFT)
            old = sel_state["current"]
            if old is None:
                self.play(
                    FadeIn(sel_label, run_time=run_time),
                    FadeIn(new_bar, shift=DOWN * 0.06, run_time=run_time),
                )
            else:
                self.play(
                    FadeOut(old, run_time=run_time * 0.4),
                    FadeIn(new_bar, run_time=run_time),
                )
            sel_state["current"] = new_bar
            sel_state["names"] = new_names
            sel_state["active"] = active_idx
            return new_bar

        # Initial selector — neutral (no tab hot yet).
        set_selector(active_idx=None, run_time=0.45)
        self.wait(0.3)

        # Blink the make_new_program tab to telegraph what's about to fire.
        # The tab is at position [, done, |, make_new_program, ]  → child #3.
        bar = sel_state["current"]
        self.play(Indicate(bar[3], color=Paper.accent, scale_factor=1.18),
                  run_time=0.7)
        self.wait(0.3)

        # ====================================================================
        # BEAT 5.A — making the new program
        # ====================================================================
        # The trace below is verbatim from /tmp/factorize_run_6.log
        # (Qwen2.5-7B local, deterministic argmax) — see
        # docs/video_script.md Beat 3 for shot-by-shot timing.
        #
        # Layout invariant: model generation lives in the LEFT column
        # (anchor x ≈ -6.0); grammar explanations / mask flashes / the
        # predicate flash live in the RIGHT column (x ≈ +1.5). The
        # vertical stack on the left reads as one continuous KV cache.
        LEFT_X = -6.0  # left column anchor for left-aligned text

        prob = Text("Factor:   1147 = p × q   (p, q > 1)",
                    font=theme.MONO_FALLBACK, font_size=16,
                    color=Paper.ink_soft)
        prob.move_to(np.array([LEFT_X, 2.25, 0]), aligned_edge=LEFT)
        self.play(FadeIn(prob, run_time=0.35))
        self.wait(0.6)

        # The model emits @make_new_program first.
        emit_call = Text(
            '@make_new_program("factor_1147",',
            font=theme.MONO_FALLBACK, font_size=13, color=Paper.accent,
        )
        emit_call2 = Text(
            '                  "two factors of 1147 greater than 1")',
            font=theme.MONO_FALLBACK, font_size=13, color=Paper.accent,
        )
        emit_grp = VGroup(emit_call, emit_call2).arrange(
            DOWN, aligned_edge=LEFT, buff=0.06,
        )
        emit_grp.next_to(prob, DOWN, buff=0.30, aligned_edge=LEFT)
        self.play(FadeIn(emit_grp, shift=UP * 0.08, run_time=0.45))
        # Selector hot: the model just took the make_new_program branch.
        set_selector(active_idx=1, run_time=0.30)
        self.wait(0.4)

        # The grammar selector at the top makes the grammar switch
        # visible; no in-line label needed. Source anchors directly
        # under the meta-call so the generation history stays a single
        # column.
        anchor_src = np.array([
            LEFT_X,
            emit_grp.get_bottom()[1] - 0.40,
            0,
        ])
        ACC = Paper.accent
        KW  = Paper.accent_soft
        FN  = Terminal.amber
        ID  = Paper.ink
        DIM = Paper.ink_soft
        NUM = Terminal.blue
        STR = Paper.good
        PRED = Paper.good
        source_rich = [
            [("@program", ACC)],
            [("def ", KW), ("_factor_1147", FN), ("():", DIM)],
            [("    n = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("1147", NUM), (", ", DIM), ("1147", NUM), (")", DIM)],
            [("    p = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("2", NUM), (", ", DIM), ("1146", NUM), (", ", DIM),
             ("where=", KW), ("divides", PRED), ("(", DIM),
             ("n", FN), (")", DIM), (")", DIM)],
            [("    q = ", ID), ("yield ", KW),
             ("gen.integer", FN), ("(", DIM),
             ("2", NUM), (", ", DIM), ("1146", NUM), (", ", DIM),
             ("where=", KW), ("multiplies_to", PRED), ("(", DIM),
             ("n", FN), (", ", DIM), ("p", FN), (")", DIM), (")", DIM)],
            [("    return ", KW),
             ("{", DIM), ("'p'", STR), (": p, ", ID),
             ("'q'", STR), (": q", ID), ("}", DIM)],
        ]
        source = _rich_block(source_rich, anchor=anchor_src,
                             line_height=0.32, font_size=13)

        # Phase 1a: lines 0..2 fade in line-by-line (header + first yield).
        self.play(LaggedStart(
            *[FadeIn(source[i], shift=LEFT * 0.08) for i in range(3)],
            lag_ratio=0.18, run_time=1.6,
        ))

        # Phase 1b: line 3 prefix — token-by-token until just after
        # `divides(`. The model has typed
        #   `    p = yield gen.integer(2, 1146, where=divides(`
        # at this point, with the cursor sitting at the predicate's arg
        # slot. We pause here for the mask flash before the line
        # finishes. Token indices match the source_rich line-3 list:
        #   0..10 = `    p = ` … `divides(`
        #   11..13 = `n` `)` `)`
        LINE3_PREFIX_COUNT = 11
        self.play(LaggedStart(
            *[FadeIn(source[3][i]) for i in range(LINE3_PREFIX_COUNT)],
            lag_ratio=0.05, run_time=0.45,
        ))

        # === MASK FLASH ============================================
        # The cursor is at the where= arg slot. Show a 7-row logit
        # column to the right of the source. Six masked candidates get
        # clay-red strikes + opacity drop; the survivor `n` glows
        # accent. The grammar rule that does the masking is
        # `var-name ::= [a-z]` (see src/orate/meta.py); the validator
        # additionally enforces "name must be bound." Caption conflates
        # both for clarity.
        mask_data = [
            ("number", "−2.1", False),
            ("value",  "−2.4", False),
            ("int",    "−3.0", False),
            ("the",    "−3.1", False),
            ("target", "−3.5", False),
            ("1147",   "−4.0", False),
            ("n",      "−1.4", True),
        ]
        mask_rows = []
        rows_vg = VGroup()
        for tok, logit, kept in mask_data:
            tok_t = Text(tok, font=theme.MONO_FALLBACK, font_size=13,
                         color=(Paper.accent if kept else Paper.ink))
            logit_t = Text(logit, font=theme.MONO_FALLBACK, font_size=11,
                           color=Paper.ink_soft)
            row = VGroup(tok_t, logit_t).arrange(
                RIGHT, buff=0.5, aligned_edge=DOWN,
            )
            rows_vg.add(row)
            mask_rows.append((row, tok_t, logit_t, kept))
        rows_vg.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rows_vg.next_to(source, RIGHT, buff=0.9)
        rows_vg.align_to(source, UP)

        mask_caption = Text(
            "grammar enforces only previously\ndeclared variables at this token",
            font="Georgia", slant="ITALIC", font_size=12,
            color=Paper.ink_soft,
        )
        mask_caption.next_to(rows_vg, UP, buff=0.22, aligned_edge=LEFT)

        all_mask = VGroup(mask_caption, rows_vg)
        self.play(FadeIn(all_mask, shift=LEFT * 0.10, run_time=0.45))

        # Apply strikes + opacity drop on masked rows; pulse the survivor.
        strike_anims = []
        for row, tok_t, logit_t, kept in mask_rows:
            if kept:
                continue
            strike = Line(
                tok_t.get_left() + np.array([-0.05, 0, 0]),
                logit_t.get_right() + np.array([0.05, 0, 0]),
                stroke_color=Paper.bad, stroke_width=2.0,
            )
            row.add(strike)
            strike_anims.append(FadeIn(strike, run_time=0.5))
            strike_anims.append(tok_t.animate.set_opacity(0.35))
            strike_anims.append(logit_t.animate.set_opacity(0.35))
        for row, tok_t, logit_t, kept in mask_rows:
            if not kept:
                continue
            strike_anims.append(
                tok_t.animate.scale(1.18).set_color(Paper.accent),
            )
        self.play(*strike_anims, run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(all_mask, run_time=0.35))
        # === /MASK FLASH ===========================================

        # Phase 2a: line 3 finishes — the surviving `n)` snaps in,
        # plus the closing paren. Token indices 11..13.
        self.play(LaggedStart(
            *[FadeIn(source[3][i])
              for i in range(LINE3_PREFIX_COUNT, len(source[3]))],
            lag_ratio=0.10, run_time=0.4,
        ))

        # Phase 2b: lines 4..5 (q-yield + return) come in line-by-line.
        self.play(LaggedStart(
            *[FadeIn(source[i], shift=LEFT * 0.08) for i in range(4, 6)],
            lag_ratio=0.18, run_time=1.2,
        ))
        self.wait(0.4)

        # Compile callout — RIGHT column, level with the top of source.
        # Three pieces appearing in succession: validated → compiled →
        # registered. Each piece is "<verb> ✓" with the ✓ in green. The
        # *registered* piece appears in sync with the selector growing
        # to include @_factor_1147; the new tab green-flashes at that
        # exact moment to make the registry growth pop.
        val_piece = _rich_line(
            [("validated ", Paper.good), ("✓", Paper.good)],
            font_size=13,
        )
        cmp_piece = _rich_line(
            [("compiled ", Paper.good), ("✓", Paper.good)],
            font_size=13,
        )
        reg_piece = _rich_line(
            [("registered ", Paper.good), ("✓", Paper.good)],
            font_size=13,
        )
        note_row = VGroup(val_piece, cmp_piece, reg_piece).arrange(
            RIGHT, buff=0.45, aligned_edge=DOWN,
        )
        # Anchor near the END of the make_program generation — right
        # column, level with the last line of the authored source. Reads
        # as: "the source just finished; here's what the runtime did to
        # it" rather than a separate floating banner at the top.
        note_row.move_to(np.array([1.7, source.get_bottom()[1] + 0.10, 0]),
                         aligned_edge=LEFT)
        # Stage compiled and registered invisible; reveal in succession.
        cmp_piece.set_opacity(0)
        reg_piece.set_opacity(0)

        self.play(FadeIn(val_piece, shift=LEFT * 0.06, run_time=0.25))
        self.wait(0.10)
        self.play(cmp_piece.animate.set_opacity(1.0), run_time=0.25)
        self.wait(0.10)

        # Selector grows + reg_piece appears + green flash, all in sync.
        new_bar = _selector(
            ["done", "make_new_program", "_factor_1147"], active_idx=1,
        )
        new_bar.move_to(np.array([0.0, SEL_Y, 0]))
        new_bar.align_to(np.array([-2.7, 0, 0]), LEFT)
        old_bar = sel_state["current"]
        self.play(
            FadeOut(old_bar, run_time=0.20),
            FadeIn(new_bar, run_time=0.30),
            reg_piece.animate.set_opacity(1.0),
        )
        sel_state["current"] = new_bar
        sel_state["names"] = ["done", "make_new_program", "_factor_1147"]
        sel_state["active"] = 1
        # Green flash on the just-registered tab — same colour as the ✓.
        # Children layout for 3 names: [, done, |, make_new_program, |, _factor_1147, ]
        # → the new tab is at index 5.
        self.play(Indicate(new_bar[5], color=Paper.good, scale_factor=1.18),
                  run_time=0.6)
        self.wait(0.3)
        # Bind compile_note for the cleanup fadeout below.
        compile_note = note_row

        # ====================================================================
        # BEAT 5.B — solving the query by it
        # ====================================================================
        # Brief pause; demote the meta-call history (it's done its job),
        # fade out the validated/compiled/registered notes (they belong
        # to Beat 5.A; the right column is about to host the rapid mask
        # cycle and predicate flash), then flip the selector to the
        # freshly-registered tab. The model is about to sample under
        # @_factor_1147's body grammar.
        self.play(
            emit_grp.animate.set_opacity(0.30),
            FadeOut(note_row, run_time=0.35),
            run_time=0.35,
        )
        set_selector(active_idx=2, run_time=0.30)
        self.wait(0.3)

        # The usage line lives on the LEFT (continues the model
        # generation stack). Built as a row of pieces so each token
        # picked by the rapid mask cycle on the right can fade into
        # position simultaneously. The stub `@_factor_1147(` appears
        # immediately; arg pieces fade in synced with their cycles.
        usage_part_specs = [
            ("@_factor_1147(", "stub"),
            ("1147",            "arg"),
            (", ",              "sep"),
            ("31",               "arg"),
            (", ",              "sep"),
            ("37",               "arg"),
            (")",                "post"),
        ]
        usage_parts = [
            Text(text, font=theme.MONO_FALLBACK, font_size=13,
                 color=Paper.accent)
            for text, _ in usage_part_specs
        ]
        usage = VGroup(*usage_parts).arrange(RIGHT, buff=0.0,
                                              aligned_edge=DOWN)
        usage.next_to(source, DOWN, buff=0.30, aligned_edge=LEFT)
        # Stage everything-but-stub invisible; cycles will reveal them.
        for p in usage_parts[1:]:
            p.set_opacity(0)
        self.play(FadeIn(usage_parts[0], shift=UP * 0.08, run_time=0.30))

        # === RAPID MASK CYCLE =====================================
        # Per-token grammar masking, fast enough to be more rhythm than
        # legible. The underlying mechanism: each emitted arg position
        # has its own grammar mask (forced int range, comma-separator,
        # close-paren) plus a where= predicate run at verification time.
        # Visual takeaway: many candidates get struck, one survives, the
        # corresponding token *simultaneously* lands in the usage line on
        # the left — making the link between mask resolution and
        # generation step explicit.
        cycle_caption = Text(
            "@_factor_1147(...) — sampled under the body grammar",
            font="Georgia", slant="ITALIC", font_size=11,
            color=Paper.ink_soft,
        )
        cycle_anchor = np.array([1.7,
                                  usage.get_center()[1] + 0.1, 0])
        cycle_caption.move_to(cycle_anchor + np.array([0, 0.65, 0]),
                               aligned_edge=LEFT)
        self.play(FadeIn(cycle_caption, run_time=0.30))

        # Five cycles, one per arg-token in `(1147, 31, 37)`. Each cycle
        # is paired with the usage-line piece that should appear when
        # that token "wins" — the kept survivor in the cycle and the
        # piece on the left fade in together with the strike anim. One
        # cycle (the `31` slot) carries an explanation caption: this is
        # the moment the where=divides(1147) predicate gates the sample,
        # mirroring the source-authoring mask-flash beat.
        cycle_specs = [
            # (kept, masked, left_piece, explanation)
            ("1147", ["int", "the", "n", "p"],    usage_parts[1], None),
            (", ",   [".",   ":",   "0", "1"],    usage_parts[2], None),
            ("31",   ["7",   "11",  "13", "32"],  usage_parts[3],
             "grammar enforces only divisors\nof 1147 at this token"),
            (", ",   [";",   ":",   ".", " "],    usage_parts[4], None),
            ("37",   ["1147", "5",  "23", "41"],  usage_parts[5], None),
        ]
        for kept, masked, left_piece, explanation in cycle_specs:
            items = []
            rows_vg = VGroup()
            for tok in masked:
                t = Text(tok, font=theme.MONO_FALLBACK, font_size=12,
                         color=Paper.ink)
                rows_vg.add(t)
                items.append((t, False))
            surv = Text(kept, font=theme.MONO_FALLBACK, font_size=12,
                         color=Paper.accent, weight="BOLD")
            rows_vg.add(surv)
            items.append((surv, True))
            rows_vg.arrange(DOWN, aligned_edge=LEFT, buff=0.07)
            rows_vg.move_to(cycle_anchor, aligned_edge=LEFT)

            self.play(FadeIn(rows_vg, run_time=0.12))
            strikes = VGroup()
            strike_anims = []
            for tok_t, kept_flag in items:
                if kept_flag:
                    continue
                s = Line(
                    tok_t.get_left() + np.array([-0.05, 0, 0]),
                    tok_t.get_right() + np.array([0.05, 0, 0]),
                    stroke_color=Paper.bad, stroke_width=1.8,
                )
                strikes.add(s)
                strike_anims.append(FadeIn(s, run_time=0.13))
                strike_anims.append(tok_t.animate.set_opacity(0.4))
            rows_vg.add(strikes)
            # The kept-token survives on the right at the same instant
            # as the corresponding piece appears on the left.
            strike_anims.append(left_piece.animate.set_opacity(1.0))
            self.play(*strike_anims, run_time=0.16)

            if explanation:
                # Educational pause: hold the panel + show explanation.
                ex_caption = Text(
                    explanation, font="Georgia", slant="ITALIC",
                    font_size=12, color=Paper.ink_soft,
                )
                ex_caption.next_to(rows_vg, DOWN, buff=0.20,
                                    aligned_edge=LEFT)
                self.play(FadeIn(ex_caption, shift=UP * 0.05,
                                  run_time=0.30))
                self.wait(1.8)
                self.play(FadeOut(ex_caption, run_time=0.25))
            else:
                self.wait(0.04)
            self.play(FadeOut(rows_vg, run_time=0.10))

        self.play(FadeOut(cycle_caption, run_time=0.20))
        # === /RAPID MASK CYCLE ====================================

        # Closing `)` lands on the left after the cycle ends.
        self.play(usage_parts[6].animate.set_opacity(1.0), run_time=0.20)

        result = Text(
            "  → {'p': 31, 'q': 37}",
            font=theme.MONO_FALLBACK, font_size=12, color=Paper.ink,
        )
        result.next_to(usage, DOWN, buff=0.10, aligned_edge=LEFT)
        self.play(FadeIn(result, run_time=0.35))
        self.wait(0.3)

        # === PREDICATE FLASH ======================================
        # The contract-honoring moment. Voiceover lands here:
        # "The model wrote down a contract — then was forced to honor it."
        # See docs/video_script.md Beat 3 for cue timing.
        # Lives in the RIGHT column, level with the usage/result on the
        # left — visually pairs "the result that just landed" with "the
        # math that the runtime ran on every candidate emission."
        pf_line1 = _rich_line(
            [("divides(1147)(31)", PRED),
             ("         → ", DIM),
             ("1147 % 31 == 0", ID),
             ("   ", DIM),
             ("✓", Paper.good)],
            font_size=12,
        )
        pf_line2 = _rich_line(
            [("multiplies_to(1147, 31)(37)", PRED),
             (" → ", DIM),
             ("37 × 31 == 1147", ID),
             ("  ", DIM),
             ("✓", Paper.good)],
            font_size=12,
        )
        pf_anchor_y = (usage.get_center()[1] + result.get_center()[1]) / 2
        pf_line1.move_to(np.array([1.7, pf_anchor_y + 0.18, 0]),
                         aligned_edge=LEFT)
        pf_line2.next_to(pf_line1, DOWN, buff=0.16, aligned_edge=LEFT)
        self.play(LaggedStart(
            FadeIn(pf_line1, shift=UP * 0.05),
            FadeIn(pf_line2, shift=UP * 0.05),
            lag_ratio=0.55, run_time=1.4,
        ))
        self.wait(2.5)  # voiceover holds here
        # === /PREDICATE FLASH =====================================

        # Selector closes the chain: model picks @done.
        set_selector(active_idx=0, run_time=0.30)
        # Done line stays in the LEFT column (continues the model
        # generation stack), not under the predicate flash.
        done_t = Text("@done(\"31 and 37\")",
                      font=theme.MONO_FALLBACK, font_size=13,
                      color=Paper.mute)
        done_t.next_to(result, DOWN, buff=0.20, aligned_edge=LEFT)
        self.play(FadeIn(done_t, shift=UP * 0.05, run_time=0.4))
        self.wait(1.0)

        # Clean for thesis card. Pull the selector + label too.
        self.play(FadeOut(VGroup(prob, emit_grp, source,
                                  compile_note, usage, result,
                                  pf_line1, pf_line2, done_t,
                                  sel_state["current"], sel_label),
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
        # Tighter cascade — was lag_ratio=0.55 / run_time=5.2 / hold=5.5,
        # now 0.45 / 4.0 / 3.5 to claw back ~3s for the personal intro
        # without making the punchline feel rushed.
        self.play(LaggedStart(
            *[FadeIn(ln, shift=UP * 0.12) for ln in thesis],
            lag_ratio=0.45, run_time=4.0,
        ))
        self.wait(3.5)

        gh = Text("github.com/maltelandgren/orate",
                  font=theme.MONO_FALLBACK, font_size=14,
                  color=Paper.ink_soft)
        gh.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(gh, run_time=0.45))
        self.wait(3.5)
