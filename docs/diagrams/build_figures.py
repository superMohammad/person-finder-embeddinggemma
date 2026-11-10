"""Write the paper-style method figure (fig1_overview) as an HTML page with SVG inside; render.py turns it into a PNG.

Run from the project root:  .venv/bin/python docs/diagrams/build_figures.py && .venv/bin/python docs/diagrams/render.py
Every image and score drawn here is a real model output saved in assets/.
"""
from pathlib import Path

from svgkit import (FROZEN, GREEN, IMAGE, INK, INK_2, LOSS, MATCH, TEXT, Palette, arrow, box, chip, curve, encoder,
                    flame, lines, math, matrix, page, panel, photo, snowflake, text, vector)

HERE: Path = Path(__file__).resolve().parent
SHARED: Palette = Palette("#eceff6", "#5b6b8f", "#2f3a55", ("#d5dbe9", "#a5b0cb", "#5b6b8f"))
GRAY_CELL: str = "#e3e6e9"


def legend_marks(x: float, y: float) -> str:
    return (flame(x, y - 20) + text(x + 22, y, "trainable (full fine-tune)", 21, 400, INK_2, "start", False)
            + snowflake(x + 330, y - 8) + text(x + 352, y, "frozen / not trained", 21, 400, INK_2, "start", False))


def fig1() -> str:
    o: list[str] = []
    # ---------- (a) training ----------
    o.append(panel(18, 18, 1964, 520, "(a) Training: contrastive text–image alignment on RSTPReid (train split)"))
    o.append(legend_marks(1290, 64))
    for i, dy in enumerate((24, 12, 0)):
        o.append(f'<rect x="{58 + dy}" y="{98 + dy}" width="372" height="118" rx="12" fill="#ffffff" stroke="{TEXT.stroke}" stroke-width="2.4"/>')
    o.append(lines(244, 146, ["“The man is wearing a red coat,", "black trousers and black shoes …”"], 21, 400, INK, "middle", True, 30))
    o.append(math(244, 262, [("English descriptions ", ""), ("T", "1"), (" … ", ""), ("T", "N")], 21, INK_2, "middle"))
    for i, n in enumerate(("3920_c7_0005", "4022_c14_0026", "4021_c14_0032", "3939_c14_0012")):
        o.append(photo(58 + i * 94, 300, 82, 160, f"assets/{n}.jpg", "#c9ced4", 1.6))
    o.append(math(244, 494, [("Person photos ", ""), ("I", "1"), (" … ", ""), ("I", "N"), ("  (one per person)", "")], 21, INK_2, "middle"))

    o.append(encoder(480, 304, 230, 150, IMAGE, "Vision tower", "→ 70 vision tokens"))
    o.append(flame(700, 288))
    o.append(arrow(436, 380, 474, 380, False))
    o.append(arrow(442, 157, 774, 157, False))
    o.append(arrow(714, 380, 774, 380, False))
    # shared backbone
    o.append(f'<rect x="780" y="96" width="250" height="364" rx="14" fill="{SHARED.fill}" stroke="{SHARED.stroke}" stroke-width="2.4"/>')
    o.append(text(905, 150, "EmbeddingGemma-2", 21.5, 700, INK, "middle", False))
    o.append(text(905, 178, "shared Gemma backbone", 19, 400, INK_2, "middle", False))
    o.append(text(905, 202, "24 layers", 19, 400, INK_2, "middle", False))
    for k, (lbl, yy) in enumerate((("mean pool", 248), ("Linear → 768", 300), ("L2 normalize", 352))):
        o.append(f'<rect x="812" y="{yy}" width="186" height="38" rx="8" fill="#ffffff" stroke="{SHARED.stroke}" stroke-width="1.6"/>')
        o.append(text(905, yy + 26, lbl, 19, 500, INK, "middle", False))
    o.append(arrow(905, 212, 905, 244, False))
    o.append(f'<line x1="905" y1="286" x2="905" y2="300" stroke="{SHARED.stroke}" stroke-width="2"/><line x1="905" y1="338" x2="905" y2="352" stroke="{SHARED.stroke}" stroke-width="2"/>')
    o.append(text(905, 428, "one model, two inputs", 17, 400, INK_2, "middle", True))
    o.append(flame(1020, 80))
    # vectors
    o.append(arrow(1030, 157, 1072, 157, False)); o.append(arrow(1030, 380, 1072, 380, False))
    o.append(vector(1080, 110, TEXT, 4, 6, 3, 16))
    o.append(math(1116, 236, [("t", "1"), (" … ", ""), ("t", "N")], 22, INK, "middle"))
    o.append(vector(1080, 332, IMAGE, 4, 6, 9, 16))
    o.append(math(1116, 458, [("v", "1"), (" … ", ""), ("v", "N")], 22, INK, "middle"))
    # similarity matrix
    mx, my, cell = 1340, 168, 52
    for i in range(4):
        o.append(chip(mx + i * cell + 4, my - 58, cell - 8, 42, TEXT, "t", str(i + 1)))
        o.append(chip(mx - 62, my + i * cell + 5, 50, cell - 10, IMAGE, "v", str(i + 1)))
    o.append(matrix(mx, my, 4, cell, GREEN, ("#e9ecef", "#f3f5f7", "#e2e6ea")))
    o.append(curve(1160, 157, 1336, my - 37, 70))
    o.append(curve(1160, 380, 1270, my + 2 * cell, 60))
    o.append(math(mx + 2 * cell, my + 4 * cell + 40, [("S", "ij"), (" = cos(", ""), ("t", "i"), (", ", ""), ("v", "j"), (") / ", ""), ("τ", ""), (",  τ = 0.05", "")], 22, INK, "middle"))
    o.append(f'<rect x="{mx - 70}" y="{my + 4 * cell + 64}" width="20" height="20" fill="{GREEN}"/>' + text(mx - 42, my + 4 * cell + 81, "matched pair", 19, 400, INK_2, "start", False))
    o.append(f'<rect x="{mx + 100}" y="{my + 4 * cell + 64}" width="20" height="20" fill="#e6e9ec" stroke="#c9ced4"/>' + text(mx + 128, my + 4 * cell + 81, "in-batch negative", 19, 400, INK_2, "start", False))
    # loss
    o.append(arrow(mx + 4 * cell + 18, my + 2 * cell, 1590, my + 2 * cell, False))
    o.append(f'<rect x="1596" y="128" width="366" height="290" rx="14" fill="{LOSS.fill}" stroke="{LOSS.stroke}" stroke-width="2.4"/>')
    o.append(text(1779, 172, "Symmetric InfoNCE", 25, 700, INK, "middle", False))
    o.append(math(1779, 212, [("ℒ = ½ (ℒ", "t→v"), (" + ℒ", "v→t"), (")", "")], 23, INK, "middle"))
    o.append(text(1779, 248, "pull the diagonal up, push the rest down", 18, 400, INK_2, "middle", True))
    o.append(f'<line x1="1620" y1="270" x2="1938" y2="270" stroke="{LOSS.stroke}" stroke-opacity=".35"/>')
    o.append(text(1779, 300, "Matryoshka: summed over the first", 18.5, 400, LOSS.text, "middle", False))
    o.append(text(1779, 326, "768, 512, 256 and 128 dimensions", 18.5, 400, LOSS.text, "middle", False))
    o.append(text(1779, 362, "N = 128 pairs, one photo per person,", 18.5, 400, LOSS.text, "middle", False))
    o.append(text(1779, 388, "cached in mini-batches of 16", 18.5, 400, LOSS.text, "middle", False))

    # ---------- (b) inference ----------
    o.append(panel(18, 560, 1964, 548, "(b) Inference: text-driven person search in live video"))
    o.append(f'<rect x="58" y="628" width="364" height="104" rx="12" fill="#ffffff" stroke="{TEXT.stroke}" stroke-width="2.4"/>')
    o.append(lines(240, 672, ["“a man in a red and white", "striped shirt”"], 23, 400, INK, "middle", True, 32))
    o.append(math(240, 764, [("Text query ", ""), ("Q", "")], 21, INK_2, "middle"))
    o.append(arrow(426, 680, 486, 680, False))
    o.append(box(492, 636, 300, 88, TEXT, "EmbeddingGemma-2", "fine-tuned · text in", 22))
    o.append(arrow(796, 680, 846, 680, False))
    o.append(vector(854, 634, TEXT, 1, 6, 5, 16))
    o.append(math(862, 760, [("q", "")], 22, INK, "middle"))
    # video frame with film strip
    o.append(f'<rect x="54" y="826" width="336" height="268" rx="8" fill="#dfe3e8" stroke="#9aa3ad" stroke-width="1.6"/>')
    for i in range(11):
        o.append(f'<rect x="{66 + i * 30}" y="834" width="16" height="9" rx="2" fill="#aab2bc"/><rect x="{66 + i * 30}" y="1077" width="16" height="9" rx="2" fill="#aab2bc"/>')
    o.append(photo(62, 850, 320, 222, "assets/mall_790.jpg", "#9aa3ad", 1))
    o.append(arrow(394, 960, 432, 960, False))
    o.append(box(438, 912, 206, 96, FROZEN, "Detector", "YOLO26s · persons", 23)); o.append(snowflake(632, 902))
    o.append(arrow(648, 960, 680, 960, False))
    o.append(box(686, 912, 196, 96, FROZEN, "Tracker", "ByteTrack ids", 23)); o.append(snowflake(870, 902))
    o.append(arrow(886, 960, 918, 960, False))
    for i, (tid, border) in enumerate((("276", "#c9ced4"), ("31", "#c9ced4"), ("434", "#c9ced4"))):
        o.append(photo(924 + i * 66, 902, 58, 116, f"assets/mall_790_t{tid}.jpg", border, 1.6))
        o.append(text(953 + i * 66, 1040, f"#{tid}", 17, 500, INK_2, "middle", False))
    o.append(math(1022, 1072, [("crops ", ""), ("c", "1"), (" … ", ""), ("c", "K")], 20, INK_2, "middle"))
    o.append(arrow(1126, 960, 1158, 960, False))
    o.append(box(1164, 912, 236, 96, IMAGE, "EmbeddingGemma-2", "every 10 frames, EMA", 21))
    o.append(arrow(1404, 960, 1436, 960, False))
    o.append(vector(1442, 912, IMAGE, 3, 6, 11, 16))
    o.append(math(1476, 1036, [("v", "1"), (" … ", ""), ("v", "K")], 22, INK, "middle"))
    o.append(curve(876, 680, 1560, 742, 330))
    o.append(curve(1504, 960, 1560, 862, 40))
    # similarity + gate
    o.append(f'<rect x="1566" y="660" width="226" height="318" rx="14" fill="{MATCH.fill}" stroke="{MATCH.stroke}" stroke-width="2.4"/>')
    o.append(text(1679, 700, "Cosine similarity", 22, 700, INK, "middle", False))
    o.append(math(1679, 734, [("s", "k"), (" = cos(", ""), ("q", ""), (", ", ""), ("v", "k"), (")", "")], 21, INK, "middle"))
    base = 884
    for i, (sc, col, lab) in enumerate(((0.72, GREEN, "#276"), (0.40, "#b8c2bc", "#31"), (0.36, "#b8c2bc", "#434"))):
        hh = sc * 160
        o.append(f'<rect x="{1606 + i * 58}" y="{base - hh}" width="40" height="{hh}" fill="{col}"/>')
        o.append(text(1626 + i * 58, (base - hh - 8) if i == 0 else (base - 10), f"{sc:.2f}", 15, 600 if i == 0 else 500, INK if i == 0 else "#ffffff", "middle", False))
        o.append(text(1626 + i * 58, base + 22, lab, 15, 400, INK_2, "middle", False))
    o.append(f'<line x1="1590" y1="{base}" x2="1784" y2="{base}" stroke="#7f8a84" stroke-width="1.6"/>')
    o.append(f'<line x1="1590" y1="{base - 0.45 * 160}" x2="1784" y2="{base - 0.45 * 160}" stroke="{INK}" stroke-width="1.6" stroke-dasharray="6 5"/>')
    o.append(text(1784, base - 0.45 * 160 - 6, "0.45", 14, 600, INK, "end", False))
    o.append(lines(1679, 944, ["wins only if s ≥ 0.45", "and box h/w ≥ 1.2"], 16, 400, MATCH.text, "middle", True, 21))
    # output crop of the real frame
    o.append(arrow(1794, 810, 1814, 810, False))
    o.append('<svg x="1820" y="660" width="156" height="318" viewBox="221 312 140 170" preserveAspectRatio="xMidYMid slice">'
             '<image href="assets/mall_790.jpg" width="640" height="480"/>'
             f'<rect x="266" y="340" width="51" height="116" fill="none" stroke="{GREEN}" stroke-width="3.5"/>'
             f'<rect x="266" y="326" width="51" height="14" fill="{GREEN}"/>'
             '<text x="291.5" y="337" font-family="Inter" font-size="10" font-weight="700" fill="#fff" text-anchor="middle">top-1</text></svg>')
    o.append(f'<rect x="1820" y="660" width="156" height="318" rx="6" fill="none" stroke="#9aa3ad" stroke-width="1.6"/>')
    o.append(text(1898, 1008, "Localized match", 19, 500, INK, "middle", False))
    o.append(math(1898, 1034, [("k", "*"), (" = argmax ", ""), ("s", "k")], 19, INK_2, "middle"))
    return page("Figure 1 · Method overview", 2000, 1126, "".join(o))


FIGURES: dict[str, object] = {"fig1_overview": fig1}


def main() -> None:
    for name, build in FIGURES.items():
        (HERE / f"{name}.html").write_text(build(), encoding="utf-8")
        print(f"docs/diagrams/{name}.html", flush=True)


if __name__ == "__main__":
    main()
