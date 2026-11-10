"""Render every docs/diagrams/fig*.html to a 2x PNG with headless Chrome, trimmed to its content height.

Run from the project root:  .venv/bin/python docs/diagrams/render.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE: Path = Path(__file__).resolve().parent
CHROME: str = "google-chrome"
WIDTH: int = 1600
MAX_HEIGHT: int = 2400
SCALE: int = 2
BOTTOM_PAD: int = 40 * SCALE


def screenshot(html: Path, out: Path) -> None:
    cmd = [CHROME, "--headless=new", "--hide-scrollbars", "--disable-gpu", f"--window-size={WIDTH},{MAX_HEIGHT}",
           f"--force-device-scale-factor={SCALE}", "--virtual-time-budget=8000", f"--screenshot={out}", html.as_uri()]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not out.is_file():
        raise RuntimeError(f"chrome failed for {html.name}: exit {res.returncode}\n{res.stderr[-2000:]}")


def trim_bottom(img: Image.Image) -> Image.Image:
    """Cut the empty paper below the last row that differs from the bottom-left background pixel."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    bg = a[-1, 0]
    rows = np.flatnonzero((np.abs(a - bg).sum(axis=2) > 12).any(axis=1))
    if rows.size == 0:
        raise ValueError("rendered figure is blank")
    return img.crop((0, 0, img.width, min(img.height, int(rows[-1]) + BOTTOM_PAD)))


def main() -> None:
    figs = sorted(HERE.glob("fig*.html")) if len(sys.argv) == 1 else [HERE / a for a in sys.argv[1:]]
    with tempfile.TemporaryDirectory() as tmp:
        for html in figs:
            raw = Path(tmp) / f"{html.stem}.png"
            screenshot(html, raw)
            out = HERE / f"{html.stem}.png"
            trim_bottom(Image.open(raw)).save(out, optimize=True)
            print(f"{out.relative_to(HERE.parent.parent)}", flush=True)


if __name__ == "__main__":
    main()
