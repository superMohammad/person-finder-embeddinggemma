"""Small SVG drawing kit for the paper-style method figures (pure functions that return SVG strings)."""
import hashlib
from dataclasses import dataclass
from html import escape

FONT: str = "Inter, 'Helvetica Neue', Arial, sans-serif"
INK: str = "#1f2328"
INK_2: str = "#57606a"
ARROW: str = "#3d4148"


@dataclass(frozen=True)
class Palette:
    """Fill, stroke and three vector-cell shades of one block family."""

    fill: str
    stroke: str
    text: str
    cells: tuple[str, str, str]


TEXT: Palette = Palette("#e4edf9", "#4a76b8", "#24497f", ("#c6d8f0", "#86a9dc", "#3c6db6"))
IMAGE: Palette = Palette("#fce8d7", "#d9822b", "#8a4a12", ("#f7d3b2", "#eda564", "#c8691d"))
MATCH: Palette = Palette("#e7f3eb", "#4c9a6a", "#1f5c3a", ("#cfe6d7", "#7fbf97", "#3a8f5c"))
LOSS: Palette = Palette("#f1ebf8", "#8064ad", "#5b3f8c", ("#ddd0ee", "#b49bd8", "#7d5fb0"))
FROZEN: Palette = Palette("#eef0f3", "#8a93a0", "#3d4148", ("#dde1e6", "#b4bcc6", "#7f8996"))
PANEL_FILL: str = "#f7f8fa"
PANEL_STROKE: str = "#d3d8de"
GREEN: str = "#3a8f5c"


def defs() -> str:
    """Arrow markers shared by every figure."""
    return (f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
            f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{ARROW}"/></marker>'
            f'<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
            f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{GREEN}"/></marker></defs>')


def text(x: float, y: float, s: str, size: float, weight: int, color: str, anchor: str, italic: bool) -> str:
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}"{style}>{escape(s)}</text>')


def math(x: float, y: float, parts: list[tuple[str, str]], size: float, color: str, anchor: str) -> str:
    """Inline maths: each part is (body, subscript); a subscript starting with ^ is a superscript instead.

    The body is italic when it is a single letter.
    """
    spans: list[str] = []
    for body, sub in parts:
        italic = len(body.strip()) == 1 and body.strip().isalpha()
        spans.append(f'<tspan font-style="{"italic" if italic else "normal"}">{escape(body)}</tspan>')
        if sub:
            shift, body_sub = ("super", sub[1:]) if sub.startswith("^") else ("sub", sub)
            spans.append(f'<tspan baseline-shift="{shift}" font-size="{size * 0.68:.1f}">{escape(body_sub)}</tspan>')
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" text-anchor="{anchor}">'
            f'{"".join(spans)}</text>')


def lines(x: float, y: float, rows: list[str], size: float, weight: int, color: str, anchor: str, italic: bool,
          gap: float) -> str:
    return "".join(text(x, y + i * gap, r, size, weight, color, anchor, italic) for i, r in enumerate(rows))


def panel(x: float, y: float, w: float, h: float, title: str) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{PANEL_FILL}" stroke="{PANEL_STROKE}" '
            f'stroke-width="2"/>' + text(x + 30, y + 46, title, 27, 700, INK, "start", False))


def box(x: float, y: float, w: float, h: float, p: Palette, title: str, sub: str, title_size: float) -> str:
    cx = x + w / 2
    if sub:
        body = (text(cx, y + h / 2 - 4, title, title_size, 700, INK, "middle", False)
                + text(cx, y + h / 2 + 26, sub, 19, 400, INK_2, "middle", False))
    else:
        body = text(cx, y + h / 2 + title_size * 0.35, title, title_size, 600, INK, "middle", False)
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{p.fill}" stroke="{p.stroke}" '
            f'stroke-width="2.4"/>' + body)


def encoder(x: float, y: float, w: float, h: float, p: Palette, title: str, sub: str) -> str:
    """Trapezoid narrowing to the right: an encoder that compresses its input into a vector."""
    inset = h * 0.13
    pts = f"{x},{y} {x + w},{y + inset} {x + w},{y + h - inset} {x},{y + h}"
    return (f'<polygon points="{pts}" fill="{p.fill}" stroke="{p.stroke}" stroke-width="2.4" stroke-linejoin="round"/>'
            + text(x + w / 2, y + h / 2 - 2, title, 22, 700, INK, "middle", False)
            + text(x + w / 2, y + h / 2 + 27, sub, 19, 400, INK_2, "middle", False))


def arrow(x1: float, y1: float, x2: float, y2: float, green: bool) -> str:
    c = GREEN if green else ARROW
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2.6" '
            f'marker-end="url(#{"ahg" if green else "ah"})"/>')


def curve(x1: float, y1: float, x2: float, y2: float, bend: float) -> str:
    """S-shaped cubic arrow; `bend` is the horizontal reach of both control points."""
    return (f'<path d="M{x1} {y1} C{x1 + bend} {y1} {x2 - bend} {y2} {x2} {y2}" fill="none" stroke="{ARROW}" '
            f'stroke-width="2.6" marker-end="url(#ah)"/>')


def vector(x: float, y: float, p: Palette, cols: int, rows: int, seed: int, cell: float) -> str:
    """A few embedding vectors drawn as columns of shaded cells (deterministic pseudo-random shades)."""
    out: list[str] = []
    s = seed
    for c in range(cols):
        for r in range(rows):
            s = (s * 1103515245 + 12345) % 2147483648
            shade = ("#ffffff",) + p.cells
            fill = shade[(s >> 16) % 4]
            out.append(f'<rect x="{x + c * (cell + 5)}" y="{y + r * cell}" width="{cell}" height="{cell}" fill="{fill}" '
                       f'stroke="{p.stroke}" stroke-width="1.3"/>')
    return "".join(out)


def photo(x: float, y: float, w: float, h: float, href: str, border: str, bw: float) -> str:
    """A real image with rounded corners and a border (green for a match)."""
    cid = "c" + hashlib.md5(f"{x},{y},{href}".encode()).hexdigest()[:10]
    return (f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7"/></clipPath>'
            f'<image href="{href}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" '
            f'clip-path="url(#{cid})"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="none" stroke="{border}" stroke-width="{bw}"/>')


def flame(x: float, y: float) -> str:
    """Trainable marker."""
    return (f'<path d="M{x} {y + 24} C{x - 11} {y + 24} {x - 13} {y + 10} {x - 4} {y + 2} C{x - 3} {y + 9} {x + 1} {y + 10} '
            f'{x + 1} {y + 5} C{x + 7} {y + 10} {x + 11} {y + 22} {x} {y + 24} z" fill="#f08a24"/>'
            f'<path d="M{x} {y + 23} C{x - 5} {y + 23} {x - 6} {y + 17} {x - 1} {y + 12} C{x + 2} {y + 17} {x + 5} {y + 21} '
            f'{x} {y + 23} z" fill="#e0452b"/>')


def snowflake(x: float, y: float) -> str:
    """Frozen (not trained here) marker."""
    arms = "".join(f'<line x1="{x}" y1="{y}" x2="{x + 13 * dx}" y2="{y + 13 * dy}" stroke="#3c6db6" stroke-width="3" '
                   f'stroke-linecap="round"/>' for dx, dy in ((0, 1), (0, -1), (0.87, 0.5), (-0.87, 0.5), (0.87, -0.5), (-0.87, -0.5)))
    return arms


def matrix(x: float, y: float, n: int, cell: float, diag: str, off: tuple[str, ...]) -> str:
    out: list[str] = []
    k = 0
    for r in range(n):
        for c in range(n):
            fill = diag if r == c else off[(r * 3 + c * 5) % len(off)]
            out.append(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" height="{cell}" fill="{fill}" '
                       f'stroke="#ffffff" stroke-width="3"/>')
            k += 1
    out.append(f'<rect x="{x}" y="{y}" width="{n * cell}" height="{n * cell}" fill="none" stroke="#9aa3ad" stroke-width="1.5"/>')
    return "".join(out)


def chip(x: float, y: float, w: float, h: float, p: Palette, body: str, sub: str) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{p.fill}" stroke="{p.stroke}" stroke-width="2"/>'
            + math(x + w / 2, y + h / 2 + 8, [(body, sub)], 22, INK, "middle"))


def page(title: str, width: int, height: int, body: str) -> str:
    """A standalone HTML page holding one figure; render.py screenshots it."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&display=block" rel="stylesheet">
<style>html,body{{margin:0;background:#ffffff}}svg{{display:block;width:1600px;height:auto}}</style></head>
<body><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">{defs()}{body}</svg></body></html>
"""
