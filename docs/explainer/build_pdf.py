"""Print docs/explainer/explainer.html to docs/Person_Finder_Explained.pdf with headless Chrome.

Run from the project root:  .venv/bin/python docs/explainer/build_pdf.py
"""
import subprocess
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent
SOURCE: Path = HERE / "explainer.html"
OUT: Path = HERE.parent / "Person_Finder_Explained.pdf"


def main() -> None:
    cmd = ["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
           f"--print-to-pdf={OUT}", SOURCE.as_uri()]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0 or not OUT.is_file():
        raise RuntimeError(f"chrome PDF export failed: exit {res.returncode}\n{res.stderr[-2000:]}")
    print(OUT.relative_to(HERE.parent.parent), f"{OUT.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
