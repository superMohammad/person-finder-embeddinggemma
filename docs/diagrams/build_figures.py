"""Write the paper-style method figures fig1..fig6 as HTML pages (SVG inside); render.py turns them into PNGs.

Run from the project root:  .venv/bin/python docs/diagrams/build_figures.py && .venv/bin/python docs/diagrams/render.py
Every number drawn here comes from the notebooks or from real model outputs saved in assets/.
"""
from pathlib import Path

from svgkit import (FROZEN, GREEN, IMAGE, INK, INK_2, LOSS, MATCH, TEXT, Palette, arrow, box, chip, curve, encoder,
                    flame, lines, math, matrix, page, panel, photo, poly, snowflake, text, vector)

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


def grid(x: float, y: float, rows: int, cols: int, cell: float, p: Palette, seed: int) -> str:
    """An embedding matrix: one row per item."""
    out: list[str] = []
    st = seed
    for r in range(rows):
        for c in range(cols):
            st = (st * 1103515245 + 12345) % 2147483648
            fill = ("#ffffff",) + p.cells
            out.append(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" height="{cell}" fill="{fill[(st >> 16) % 4]}" '
                       f'stroke="{p.stroke}" stroke-width="1"/>')
    return "".join(out)


def tick(x: float, y: float, ok: bool) -> str:
    color = GREEN if ok else "#c2412b"
    return text(x, y, "✓" if ok else "✗", 22, 700, color, "middle", False)


def fig2() -> str:
    o: list[str] = []
    # (a) offline gallery encoding
    o.append(panel(18, 18, 1964, 300, "(a) Offline, once: encode every gallery photo"))
    gal = ["4022_c14_0026", "3961_c14_0012", "3920_c1_0001", "4040_c14_0000", "4021_c14_0032", "4052_c14_0002", "3939_c14_0012"]
    for i, n in enumerate(gal):
        o.append(photo(58 + i * 58, 96, 50, 100, f"assets/{n}.jpg", "#c9ced4", 1.4))
    o.append(text(58 + 3.5 * 58, 236, "gallery: 1,000 test photos (or all 20,505)", 20, 400, INK_2, "middle", False))
    o.append(arrow(470, 146, 520, 146, False))
    o.append(encoder(526, 76, 260, 140, IMAGE, "EmbeddingGemma-2", "image input · 70 tokens"))
    o.append(arrow(790, 146, 840, 146, False))
    o.append(grid(848, 82, 8, 14, 16, IMAGE, 21))
    o.append(math(848 + 7 * 16, 238, [("G", ""), (" ∈ ℝ", "^1000 × 768")], 22, INK, "middle"))
    o.append(text(848 + 7 * 16, 268, "rows are L2-normalized", 18, 400, INK_2, "middle", True))
    o.append(f'<rect x="1150" y="96" width="320" height="110" rx="12" fill="#ffffff" stroke="#9aa3ad" stroke-width="2" stroke-dasharray="7 6"/>')
    o.append(text(1310, 140, "cached to disk", 22, 600, INK, "middle", False))
    o.append(text(1310, 172, "image_embeddings.npy", 19, 400, INK_2, "middle", False))
    o.append(arrow(1086, 146, 1144, 146, False))
    o.append(lines(1520, 116, ["≈ 15 min for 20,505 photos on an RTX 3090.",
                                "Photos never change, so this runs once,",
                                "and every later search reuses it."], 19, 400, INK_2, "start", False, 30))

    # (b) online query
    o.append(panel(18, 368, 1964, 420, "(b) Online, per query: one matrix–vector product, then sort"))
    o.append(f'<rect x="58" y="450" width="330" height="110" rx="12" fill="#ffffff" stroke="{TEXT.stroke}" stroke-width="2.4"/>')
    o.append(lines(223, 494, ["“a man wearing a red jacket,", "black trousers and black shoes”"], 20, 400, INK, "middle", True, 30))
    o.append(text(223, 592, "+ prompt “task: search result | query:”", 16.5, 400, INK_2, "middle", False))
    o.append(arrow(392, 505, 432, 505, False))
    o.append(encoder(438, 436, 240, 138, TEXT, "EmbeddingGemma-2", "text input"))
    o.append(arrow(682, 505, 722, 505, False))
    o.append(vector(730, 458, TEXT, 1, 6, 4, 16))
    o.append(math(738, 582, [("q", ""), (" ∈ ℝ", "^768")], 20, INK, "middle"))
    o.append(arrow(760, 505, 800, 505, False))
    o.append(box(806, 452, 226, 106, MATCH, "Score all", "", 23))
    o.append(math(919, 538, [("s = G q", "")], 22, INK, "middle"))
    o.append(poly([(848 + 7 * 16, 278), (848 + 7 * 16, 340), (919, 340), (919, 446)], True))
    o.append(arrow(1036, 505, 1076, 505, False))
    o.append(box(1082, 462, 130, 86, FROZEN, "top-k", "", 23))
    o.append(arrow(1216, 505, 1256, 505, False))
    hits = [("3920_c1_0001", 0.579, True), ("3920_c14_0015", 0.559, True), ("3920_c7_0005", 0.537, True),
            ("3920_c15_0022", 0.528, True), ("3946_c7_0005", 0.476, False)]
    for i, (n, sc, ok) in enumerate(hits):
        x = 1268 + i * 140
        o.append(photo(x, 420, 116, 232, f"assets/{n}.jpg", GREEN if ok else "#c9ced4", 4 if ok else 1.6))
        o.append(f'<rect x="{x}" y="420" width="34" height="26" rx="4" fill="{INK}"/>' + text(x + 17, 439, str(i + 1), 17, 700, "#ffffff", "middle", False))
        o.append(text(x + 46, 684, f"{sc:.3f}", 19, 500, INK, "middle", False) + tick(x + 98, 685, ok))
    o.append(text(1608, 740, "real output: ranks 1–4 are the same man (id 3920), seen by four cameras", 18, 400, INK_2, "middle", True))

    # (c) evaluation
    o.append(panel(18, 808, 1964, 420, "(c) Evaluation, protocol A: 2,000 test descriptions × 1,000 test photos, 200 unseen people"))
    order = [("3920_c1_0001", True), ("3920_c14_0015", True), ("3920_c7_0005", True), ("3920_c15_0022", True),
             ("3946_c7_0005", False), ("4039_c14_0000", False), ("4040_c14_0000", False), ("4052_c14_0002", False),
             ("3920_c5_0008", True), ("4022_c14_0026", False)]
    for i, (n, ok) in enumerate(order):
        x = 58 + i * 84
        o.append(photo(x, 890, 72, 144, f"assets/{n}.jpg", GREEN if ok else "#c9ced4", 4 if ok else 1.4))
        o.append(text(x + 36, 1066, str(i + 1), 18, 600 if ok else 400, INK if ok else INK_2, "middle", False))
    o.append(text(58, 1106, "one query's ranking (illustrative): the person's 5 photos land at ranks 1, 2, 3, 4 and 9", 18.5, 400, INK_2, "start", True))
    o.append(f'<rect x="58" y="1124" width="20" height="20" rx="3" fill="none" stroke="{GREEN}" stroke-width="3"/>' + text(88, 1141, "photo of the right person", 18, 400, INK_2, "start", False))
    rows = [("Rank-k", "first correct photo within the top k", "Rank-1 = 1,  Rank-5 = 1"),
            ("AP", "mean of precision at each correct photo", "(1+1+1+1+5/9) / 5 = 0.911"),
            ("INP", "correct photos ÷ rank of the last one", "5 / 9 = 0.556")]
    o.append(f'<rect x="940" y="874" width="1016" height="330" rx="14" fill="#ffffff" stroke="#d3d8de" stroke-width="2"/>')
    o.append(text(970, 914, "per query", 18, 600, INK_2, "start", False) + text(1580, 914, "for this ranking", 18, 600, INK_2, "start", False))
    for i, (k, d, v) in enumerate(rows):
        yy = 964 + i * 62
        o.append(text(970, yy, k, 23, 700, INK, "start", False) + text(1080, yy, d, 20, 400, INK_2, "start", False)
                 + text(1580, yy, v, 20, 500, INK, "start", False))
        o.append(f'<line x1="970" y1="{yy + 22}" x2="1926" y2="{yy + 22}" stroke="#e3e6e9"/>')
    o.append(lines(970, 1150, ["Averaged over all 2,000 queries → Rank-1, Rank-5, Rank-10, mAP, mINP.",
                                "fine-tuned: 71.4 / 87.3 / 93.2 / 54.8 / 29.5  ·  zero-shot: 40.1 / 65.5 / 75.4 / 28.8 / 9.7"], 18.5, 400, INK, "start", False, 30))
    return page("Figure 2 · Retrieval and evaluation", 2000, 1246, "".join(o))


def fig3() -> str:
    o: list[str] = []
    people = {3920: ["3920_c1_0001", "3920_c5_0008", "3920_c7_0005", "3920_c14_0015", "3920_c15_0022"],
              4022: ["4022_c14_0026", "4022_c1_0040", "4022_c3_0010", "4022_c5_0034", "4022_c7_0035"],
              4021: ["4021_c14_0032", "4021_c1_0004", "4021_c3_0016", "4021_c5_0029", "4021_c7_0022"],
              3939: ["3939_c14_0012", "3939_c14_0013", "3939_c15_0026", "3939_c5_0002", "3939_c5_0003"]}
    # (a) unique-person sampler
    o.append(panel(18, 18, 1964, 490, "(a) Batch construction: at most one photo per person"))
    for k, (pid, imgs) in enumerate(people.items()):
        x0, y0 = 58 + k * 150, 96
        for j, n in enumerate(reversed(imgs)):
            o.append(photo(x0 + (4 - j) * 9, y0 + (4 - j) * 9, 92, 184, f"assets/{n}.jpg", "#ffffff", 2))
        o.append(text(x0 + 64, y0 + 254, f"id {pid}", 19, 600, INK, "middle", False))
        o.append(text(x0 + 64, y0 + 278, "5 photos", 17, 400, INK_2, "middle", False))
    o.append(text(330, 436, "train split: 3,701 people, ≈ 5 photos each, 2 descriptions per photo", 18.5, 400, INK_2, "middle", True))
    o.append(arrow(662, 210, 712, 210, False))
    o.append(f'<rect x="718" y="140" width="300" height="140" rx="12" fill="{LOSS.fill}" stroke="{LOSS.stroke}" stroke-width="2.4"/>')
    o.append(text(868, 186, "Unique-person", 22, 700, INK, "middle", False) + text(868, 212, "batch sampler", 22, 700, INK, "middle", False))
    o.append(lines(868, 244, ["shuffle ids each epoch,", "draw ≤ 1 photo per id"], 17, 400, LOSS.text, "middle", True, 22))
    o.append(arrow(1022, 210, 1072, 210, False))
    for k, pid in enumerate(people):
        x = 1084 + k * 118
        o.append(photo(x, 100, 92, 184, f"assets/{people[pid][2]}.jpg", "#c9ced4", 1.6))
        o.append(chip(x + 6, 296, 80, 40, TEXT, "T", str(k + 1)))
    o.append(text(1084 + 4 * 118 + 10, 200, "…", 34, 700, INK_2, "start", False))
    o.append(text(1340, 378, "batch of N = 128 (photo, description) pairs, 128 different people", 18.5, 400, INK_2, "middle", True))
    # why: false negative
    o.append(f'<rect x="1640" y="88" width="320" height="400" rx="14" fill="#ffffff" stroke="#e3b8ae" stroke-width="2"/>')
    o.append(text(1800, 122, "without the sampler", 19, 700, "#a5321f", "middle", False))
    mx, my, cell = 1716, 176, 42
    for i in range(4):
        o.append(chip(mx + i * cell + 3, my - 44, cell - 6, 34, TEXT, "t", str(i + 1)))
    o.append(matrix(mx, my, 4, cell, GREEN, ("#e9ecef", "#f3f5f7", "#e2e6ea")))
    o.append(f'<rect x="{mx + 2 * cell}" y="{my}" width="{cell}" height="{cell}" fill="#e0533d" stroke="#ffffff" stroke-width="3"/>')
    o.append(text(mx + 2.5 * cell, my + cell * 0.66, "✗", 22, 700, "#ffffff", "middle", False))
    o.append(photo(mx - 52, my + 2, 38, 76, "assets/3920_c7_0005.jpg", "#c9ced4", 1.2))
    o.append(photo(mx - 52, my + 2 * cell + 2, 38, 76, "assets/3920_c1_0001.jpg", "#e0533d", 2.4))
    o.append(lines(1800, my + 4 * cell + 44, ["Photo 3 is the same man as", "description 1, yet the loss calls", "it a wrong answer and pushes", "the two apart: a false negative."], 17, 400, INK_2, "middle", False, 24))

    # (b) cached MNRL
    o.append(panel(18, 528, 960, 560, "(b) Cached loss: 128 negatives at the memory cost of 16"))
    steps = [("1", "Embed all 128 pairs in mini-batches of 16", "no gradients kept, so memory stays small"),
             ("2", "Loss on the full 128 × 128 matrix", "store ∂ℒ/∂embedding for every item"),
             ("3", "Re-embed each mini-batch with gradients", "multiply in the stored ∂ℒ, backprop, free memory")]
    for i, (n, t1, t2) in enumerate(steps):
        yy = 606 + i * 150
        o.append(f'<circle cx="86" cy="{yy + 40}" r="24" fill="{INK}"/>' + text(86, yy + 48, n, 22, 700, "#ffffff", "middle", False))
        o.append(text(130, yy + 34, t1, 22, 600, INK, "start", False) + text(130, yy + 64, t2, 19, 400, INK_2, "start", True))
        for j in range(8):
            o.append(f'<rect x="{620 + j * 40}" y="{yy + 16}" width="34" height="48" rx="5" fill="{IMAGE.fill if i != 1 else LOSS.fill}" '
                     f'stroke="{IMAGE.stroke if i != 1 else LOSS.stroke}" stroke-width="1.6"/>')
        o.append(text(780, yy + 96, "8 mini-batches × 16 = 128" if i != 1 else "one 128 × 128 similarity matrix", 16.5, 400, INK_2, "middle", False))
        if i < 2:
            o.append(arrow(86, yy + 70, 86, yy + 140, False))
    o.append(text(58, 1062, "CachedMultipleNegativesRankingLoss (GradCache); ≈ 18 GB of 24 GB, 6.8 s per 128-pair step", 17.5, 400, INK_2, "start", True))

    # (c) Matryoshka
    o.append(panel(998, 528, 984, 560, "(c) Matryoshka: one vector, usable at four sizes"))
    dims = [(768, 71.4), (512, 71.1), (256, 70.1), (128, 65.1)]
    x0, y0, full = 1040, 620, 600
    for i, (d, r1) in enumerate(dims):
        w = full * d / 768
        yy = y0 + i * 80
        o.append(f'<rect x="{x0}" y="{yy}" width="{w}" height="44" rx="6" fill="{["#c8691d", "#e08a3e", "#eda564", "#f7d3b2"][i]}"/>')
        o.append(text(x0 + w + 14, yy + 30, f"first {d}", 20, 600, INK, "start", False))
        o.append(math(x0 + full + 150, yy + 30, [("ℒ", str(d))], 22, LOSS.text, "middle"))
        o.append(text(x0 + full + 290, yy + 30, f"{r1}", 21, 600, INK, "end", False))
    o.append(text(x0 + full + 150, y0 - 18, "loss", 17, 600, INK_2, "middle", False) + text(x0 + full + 290, y0 - 18, "Rank-1", 17, 600, INK_2, "end", False))
    o.append(math(1400, y0 + 4 * 80 + 30, [("ℒ = ℒ", "768"), (" + ℒ", "512"), (" + ℒ", "256"), (" + ℒ", "128")], 24, LOSS.text, "middle"))
    o.append(lines(1040, y0 + 4 * 80 + 78, ["The first numbers carry the most information, so a 256-number", "index is 3× smaller and loses about one Rank-1 point."], 18, 400, INK_2, "start", True, 26))
    return page("Figure 3 · Training details", 2000, 1106, "".join(o))


def fig4() -> str:
    o: list[str] = []
    o.append(panel(18, 18, 1964, 560, "(a) Over time: each track is re-encoded every 10 frames and smoothed"))
    x0, x1, base = 260, 1900, 520
    o.append(f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="{INK}" stroke-width="2" marker-end="url(#ah)"/>')
    for k in range(0, 41, 5):
        x = x0 + k * (x1 - x0 - 30) / 40
        o.append(f'<line x1="{x}" y1="{base - 6}" x2="{x}" y2="{base + 6}" stroke="{INK}" stroke-width="1.6"/>' + text(x, base + 32, f"t+{k}" if k else "t", 17, 400, INK_2, "middle", False))
    lanes = [("#276", 0, 40, [0, 10, 20, 30, 40], True), ("#31", 0, 34, [0, 10, 20, 30], False), ("#434", 12, 40, [12, 22, 32], False)]
    xs = lambda f: x0 + f * (x1 - x0 - 30) / 40
    for i, (tid, a, b, enc, win) in enumerate(lanes):
        yy = 120 + i * 120
        o.append(text(70, yy + 8, f"track {tid}", 21, 700 if win else 500, INK, "start", False))
        o.append(f'<rect x="{xs(a)}" y="{yy - 14}" width="{xs(b) - xs(a)}" height="28" rx="14" fill="{MATCH.fill if win else "#eef0f3"}" stroke="{MATCH.stroke if win else "#aab2bc"}" stroke-width="1.6"/>')
        for f in enc:
            o.append(f'<circle cx="{xs(f)}" cy="{yy}" r="10" fill="{IMAGE.stroke}" stroke="#ffffff" stroke-width="2.5"/>')
    o.append(text(xs(12), 330, "new track: encoded on its first frame", 16.5, 400, INK_2, "middle", True))
    o.append(f'<circle cx="80" cy="420" r="10" fill="{IMAGE.stroke}" stroke="#ffffff" stroke-width="2.5"/>' + text(100, 427, "crop encoded (one GPU batch for all due tracks)", 18, 400, INK_2, "start", False))
    o.append(f'<rect x="70" y="448" width="22" height="16" rx="8" fill="{MATCH.fill}" stroke="{MATCH.stroke}"/>' + text(100, 462, "track visible", 18, 400, INK_2, "start", False))
    o.append(f'<rect x="1290" y="388" width="650" height="96" rx="12" fill="{IMAGE.fill}" stroke="{IMAGE.stroke}" stroke-width="2"/>')
    o.append(math(1615, 428, [("e", "k"), (" ← normalize(0.7 · ", ""), ("e", "k"), (" + 0.3 · ", ""), ("v", "new"), (")", "")], 22, INK, "middle"))
    o.append(text(1615, 462, "EMA: one blurry crop cannot flip the decision", 17.5, 400, INK_2, "middle", True))

    # (b) decision per frame
    o.append(panel(18, 598, 1964, 520, "(b) Every frame: score all tracks, then two gates decide who gets the box"))
    o.append(photo(58, 680, 360, 270, "assets/mall_790.jpg", "#9aa3ad", 1.4))
    o.append(text(238, 984, "frame t (Mall, real tracker output)", 18, 400, INK_2, "middle", True))
    o.append(arrow(424, 815, 470, 815, False))
    ppl = [("276", 0.72, 2.27), ("31", 0.40, 1.39), ("434", 0.36, 2.55), ("467", 0.26, 2.36)]
    for i, (tid, sc, asp) in enumerate(ppl):
        yy = 670 + i * 74
        o.append(photo(480, yy, 30, 60, f"assets/mall_790_t{tid}.jpg", "#c9ced4", 1))
        o.append(text(522, yy + 26, f"#{tid}", 18, 600, INK, "start", False))
        o.append(math(522, yy + 52, [("s", ""), (f" = {sc:.2f}  ·  h/w = {asp:.2f}", "")], 16.5, INK_2, "start"))
    o.append(arrow(780, 815, 830, 815, False))
    o.append(box(836, 765, 230, 100, MATCH, "best eligible", "", 22))
    o.append(math(951, 846, [("k", "*"), (" = argmax ", ""), ("s", "k")], 19, INK_2, "middle"))
    o.append(arrow(1070, 815, 1120, 815, False))
    diamonds = [(1230, "s ≥ 0.45 ?", "strangers peak at 0.40–0.42"), (1520, "h/w ≥ 1.2 ?", "full body, not a head or two people")]
    for cx, q, why in diamonds:
        o.append(f'<polygon points="{cx - 104},815 {cx},745 {cx + 104},815 {cx},885" fill="#ffffff" stroke="{INK}" stroke-width="2.2"/>')
        o.append(text(cx, 822, q, 20, 700, INK, "middle", False))
        o.append(text(cx, 730, why, 16, 400, INK_2, "middle", True))
        o.append(poly([(cx, 885), (cx, 960), (cx, 1010)], False))
        o.append(text(cx + 14, 980, "no", 17, 600, "#a5321f", "start", False))
    o.append(arrow(1334, 815, 1414, 815, True) + text(1374, 804, "yes", 17, 600, GREEN, "middle", False))
    o.append(arrow(1624, 815, 1690, 815, True) + text(1657, 804, "yes", 17, 600, GREEN, "middle", False))
    for cx in (1230, 1520):
        o.append(f'<rect x="{cx - 92}" y="1016" width="184" height="54" rx="10" fill="#eef0f3" stroke="#aab2bc" stroke-width="1.6"/>' + text(cx, 1050, "no green box", 18, 500, INK_2, "middle", False))
    o.append('<svg x="1696" y="690" width="250" height="250" viewBox="200 300 180 180" preserveAspectRatio="xMidYMid slice">'
             '<image href="assets/mall_790.jpg" width="640" height="480"/>'
             f'<rect x="266" y="340" width="51" height="116" fill="none" stroke="{GREEN}" stroke-width="3"/>'
             f'<rect x="266" y="327" width="66" height="13" fill="{GREEN}"/>'
             '<text x="269" y="337" font-family="Inter" font-size="10" font-weight="700" fill="#fff">MATCH .72</text></svg>')
    o.append(f'<rect x="1696" y="690" width="250" height="250" rx="6" fill="none" stroke="#9aa3ad" stroke-width="1.6"/>')
    o.append(text(1821, 976, "green box on #276", 19, 600, INK, "middle", False))
    o.append(text(1821, 1004, "≈ 26 fps end to end", 17, 400, INK_2, "middle", False))
    o.append(text(1821, 1030, "new query: next frame (≈ 150 ms)", 17, 400, INK_2, "middle", False))
    return page("Figure 4 · Video grounding over time", 2000, 1136, "".join(o))


def fig5() -> str:
    import json
    h = json.loads((HERE / "assets" / "mall_790_heads.json").read_text())
    o: list[str] = []
    o.append(panel(18, 18, 1964, 790, "Detector evaluation on the Mall dataset: one-to-one matching of person boxes to labelled heads"))
    # (a) frame with heads and boxes
    sx = 760 / 640
    o.append(text(58, 98, "(a) frame 790: 22 labelled heads, 7 YOLO boxes", 21, 600, INK, "start", False))
    o.append(f'<svg x="58" y="116" width="760" height="570" viewBox="0 0 640 480"><image href="assets/mall_790.jpg" width="640" height="480"/>')
    for b in h["boxes"]:
        o.append(f'<rect x="{b[0]}" y="{b[1]}" width="{b[2] - b[0]}" height="{b[3] - b[1]}" fill="none" stroke="#ffffff" stroke-width="2"/>')
        o.append(f'<rect x="{b[0]}" y="{b[1]}" width="{b[2] - b[0]}" height="{0.45 * (b[3] - b[1])}" fill="#ffffff" fill-opacity=".18"/>')
    for (hx, hy), hit in zip(h["heads"], h["gt_hit"]):
        o.append(f'<circle cx="{hx}" cy="{hy}" r="4.5" fill="{GREEN if hit else "#e0533d"}" stroke="#ffffff" stroke-width="1.4"/>')
    o.append('</svg>')
    o.append(f'<rect x="58" y="116" width="760" height="570" fill="none" stroke="#9aa3ad" stroke-width="1.4"/>')
    # (b) rule
    o.append(text(880, 98, "(b) the matching rule", 21, 600, INK, "start", False))
    o.append(f'<rect x="900" y="140" width="120" height="270" rx="4" fill="none" stroke="{INK}" stroke-width="2.4"/>')
    o.append(f'<rect x="900" y="140" width="120" height="{0.45 * 270}" fill="{MATCH.fill}" stroke="none"/>')
    o.append(f'<line x1="890" y1="{140 + 0.45 * 270}" x2="1030" y2="{140 + 0.45 * 270}" stroke="{MATCH.stroke}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    o.append(f'<circle cx="960" cy="170" r="9" fill="{GREEN}" stroke="#fff" stroke-width="2"/>' + f'<circle cx="1008" cy="330" r="9" fill="#e0533d" stroke="#fff" stroke-width="2"/>')
    o.append(text(1046, 200, "top 45 % of the box:", 18.5, 600, INK, "start", False) + text(1046, 226, "a head here may match it", 18.5, 400, INK_2, "start", False))
    o.append(text(1046, 336, "a head lower down may not", 18.5, 400, INK_2, "start", False))
    o.append(lines(880, 470, ["Each head matches at most one box,", "and each box at most one head.", "Among the allowed pairs, the Hungarian", "algorithm picks the closest set."], 18.5, 400, INK_2, "start", False, 28))
    # (c) bipartite
    o.append(text(1460, 98, "(c) bipartite assignment", 21, 600, INK, "start", False))
    bx = [(1500, 150 + i * 56) for i in range(5)]
    hx_ = [(1760, 140 + i * 48) for i in range(6)]
    pairs = [(0, 0), (1, 2), (2, 3), (3, 4)]
    for a, b in pairs:
        o.append(f'<line x1="{bx[a][0] + 40}" y1="{bx[a][1]}" x2="{hx_[b][0] - 10}" y2="{hx_[b][1]}" stroke="{GREEN}" stroke-width="3"/>')
    o.append(f'<line x1="{bx[1][0] + 40}" y1="{bx[1][1]}" x2="{hx_[1][0] - 10}" y2="{hx_[1][1]}" stroke="#c9ced4" stroke-width="1.6" stroke-dasharray="5 4"/>')
    for i, (x, y) in enumerate(bx):
        o.append(f'<rect x="{x}" y="{y - 18}" width="40" height="36" rx="5" fill="#ffffff" stroke="{INK}" stroke-width="2"/>')
    for i, (x, y) in enumerate(hx_):
        hit = i in (0, 2, 3, 4)
        o.append(f'<circle cx="{x}" cy="{y}" r="10" fill="{GREEN if hit else "#e0533d"}" stroke="#fff" stroke-width="2"/>')
    o.append(text(1520, 450, "boxes", 18, 600, INK_2, "middle", False) + text(1760, 450, "heads", 18, 600, INK_2, "middle", False))
    o.append(lines(1460, 498, ["matched pair → true positive (TP)", "head left alone → missed (FN)", "box left alone → false alarm (FP)"], 18, 400, INK_2, "start", False, 28))
    # summary
    o.append(f'<rect x="880" y="640" width="1076" height="94" rx="12" fill="#ffffff" stroke="#d3d8de" stroke-width="2"/>')
    o.append(text(910, 678, f"This frame: TP = {h['tp']}, FP = {h['fp']}, FN = {h['fn']}.", 20, 600, INK, "start", False))
    o.append(text(910, 712, "Over 2,000 frames: precision 97.7 %, recall 39.0 % at 640 px · recall 66.3 % at 1280 px", 18.5, 400, INK_2, "start", False))
    o.append(f'<circle cx="80" cy="726" r="7" fill="{GREEN}"/>' + text(94, 732, "head matched to a box", 17, 400, INK_2, "start", False))
    o.append(f'<circle cx="320" cy="726" r="7" fill="#e0533d"/>' + text(334, 732, "head with no box (missed)", 17, 400, INK_2, "start", False))
    return page("Figure 5 · Mall detector evaluation", 2000, 826, "".join(o))


def bars(x: float, y: float, w: float, h: float, groups: list[tuple[str, float, float]], vmax: float, step: int) -> str:
    """Grouped bars, zero-shot (blue) vs fine-tuned (orange), with value labels."""
    o: list[str] = []
    gw = w / len(groups)
    for k in range(0, int(vmax) + 1, step):
        yy = y + h - h * k / vmax
        o.append(f'<line x1="{x}" y1="{yy}" x2="{x + w}" y2="{yy}" stroke="#e3e6e9"/>' + text(x - 10, yy + 6, str(k), 16, 400, INK_2, "end", False))
    for i, (name, a, b) in enumerate(groups):
        gx = x + i * gw + gw / 2
        for j, (v, col) in enumerate(((a, "#2a78d6"), (b, "#eb6834"))):
            bh = h * v / vmax
            bx = gx - 46 + j * 48
            o.append(f'<rect x="{bx}" y="{y + h - bh}" width="44" height="{bh}" rx="3" fill="{col}"/>')
            o.append(text(bx + 22, y + h - bh - 8, f"{v:.1f}", 15.5, 600 if j else 400, INK, "middle", False))
        o.append(text(gx, y + h + 30, name, 19, 600, INK, "middle", False))
    o.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="{INK}" stroke-width="1.6"/>')
    return "".join(o)


def fig6() -> str:
    o: list[str] = []
    o.append(panel(18, 18, 1180, 560, "(a) Test split (protocol A): zero-shot vs fine-tuned, 70 vision tokens"))
    o.append(bars(110, 110, 1040, 360, [("Rank-1", 40.1, 71.4), ("Rank-5", 65.5, 87.3), ("Rank-10", 75.4, 93.2), ("mAP", 28.8, 54.8), ("mINP", 9.7, 29.5)], 100, 20))
    o.append(f'<rect x="420" y="530" width="18" height="18" rx="3" fill="#2a78d6"/>' + text(446, 545, "zero-shot", 18, 400, INK_2, "start", False))
    o.append(f'<rect x="600" y="530" width="18" height="18" rx="3" fill="#eb6834"/>' + text(626, 545, "fine-tuned (this work)", 18, 400, INK_2, "start", False))
    o.append(panel(1218, 18, 764, 270, "(b) Vision tokens per image (zero-shot, test)"))
    o.append(bars(1290, 104, 400, 120, [("Rank-1", 35.4, 40.1), ("mAP", 25.0, 28.8)], 60, 20).replace("#2a78d6", "#9aa3ad").replace("#eb6834", "#5b6b8f"))
    o.append(f'<rect x="1730" y="130" width="16" height="16" rx="3" fill="#9aa3ad"/>' + text(1754, 144, "280 tokens", 17, 400, INK_2, "start", False))
    o.append(f'<rect x="1730" y="162" width="16" height="16" rx="3" fill="#5b6b8f"/>' + text(1754, 176, "70 tokens (used)", 17, 400, INK_2, "start", False))
    o.append(panel(1218, 308, 764, 270, "(c) Mall crowd: detector quality (%)"))
    o.append(bars(1290, 394, 400, 120, [("precision", 97.7, 92.4), ("recall", 39.0, 66.3)], 100, 50).replace("#2a78d6", "#9aa3ad").replace("#eb6834", GREEN))
    o.append(f'<rect x="1730" y="420" width="16" height="16" rx="3" fill="#9aa3ad"/>' + text(1754, 434, "640 px, conf 0.40", 17, 400, INK_2, "start", False))
    o.append(f'<rect x="1730" y="452" width="16" height="16" rx="3" fill="{GREEN}"/>' + text(1754, 466, "1280 px, conf 0.25", 17, 400, INK_2, "start", False))
    return page("Figure 6 · Results", 2000, 596, "".join(o))


FIGURES: dict[str, object] = {"fig1_overview": fig1, "fig2_retrieval": fig2, "fig3_training": fig3,
                               "fig4_video": fig4, "fig5_mall": fig5, "fig6_results": fig6}


def main() -> None:
    for name, build in FIGURES.items():
        (HERE / f"{name}.html").write_text(build(), encoding="utf-8")
        print(f"docs/diagrams/{name}.html", flush=True)


if __name__ == "__main__":
    main()
