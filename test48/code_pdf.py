#!/usr/bin/env python3
"""Dump every test48 *.go file into a readable PDF.

    .venv/bin/python code_pdf.py
    .venv/bin/python code_pdf.py --out test48_code.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-test48")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NAVY = "#1a365d"
SLATE = "#2d3748"
MUTED = "#718096"
GREEN = "#276749"

LINES_PER_PAGE = 64
WRAP = 108
FONTSIZE = 6.3
LINE_H = 0.0128
LEFT = 0.055
TOP = 0.90

# DejaVu Sans Mono has no emoji; keep the PDF from painting tofu.
_EMOJI = {
    "❌": "[x]",
    "✅": "[ok]",
    "🧠": "[layers]",
    "🔢": "[n]",
    "📐": "[cam]",
    "🧪": "[tasks]",
    "📊": "[jobs]",
    "🏆": "[win]",
}


def collect_go(root: Path) -> list[Path]:
    files = sorted(p for p in root.glob("*.go") if p.is_file())
    order = ["test48.go", "model.go", "hemis.go", "workclock.go", "test48_test.go"]
    rank = {n: i for i, n in enumerate(order)}
    files.sort(key=lambda p: (rank.get(p.name, 100), p.name))
    return files


def wrap_line(text: str, width: int) -> list[str]:
    text = text.replace("\t", "    ").rstrip("\n")
    if len(text) <= width:
        return [text]
    out = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        if cut < width // 2:
            cut = width
        out.append(text[:cut])
        text = "        " + text[cut:].lstrip()
    if text.strip():
        out.append(text)
    return out or [""]


def pdf_safe(s: str) -> str:
    for a, b in _EMOJI.items():
        s = s.replace(a, b)
    return s


def new_page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=12, fontweight="bold", color=NAVY, x=0.055, ha="left", y=0.97)
    if subtitle:
        fig.text(0.055, 0.942, subtitle, fontsize=7.5, color=MUTED, ha="left")
    return fig


def footer(fig, page: int, total: int, name: str):
    fig.text(0.055, 0.022, f"welvet test48  ·  {name}", fontsize=6.5, color=MUTED)
    fig.text(0.94, 0.022, f"{page} / {total}", fontsize=7, color=MUTED, ha="right")


def draw_line(fig, y: float, ln: int, text: str):
    fig.text(LEFT, y, f"{ln:4d}", fontsize=FONTSIZE, color=MUTED,
             family="DejaVu Sans Mono", va="top", ha="left")
    color = GREEN if text.lstrip().startswith("//") else SLATE
    fig.text(LEFT + 0.048, y, pdf_safe(text), fontsize=FONTSIZE, color=color,
             family="DejaVu Sans Mono", va="top", ha="left")


def paginate_file(path: Path) -> list[tuple[int, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[tuple[int, str]] = []
    for i, line in enumerate(raw, 1):
        wrapped = wrap_line(line, WRAP)
        rows.append((i, wrapped[0]))
        for cont in wrapped[1:]:
            rows.append((i, cont))
    if not rows:
        rows = [(1, "")]
    return rows


def count_pages(files: list[Path]) -> int:
    n = 1  # cover
    for p in files:
        rows = paginate_file(p)
        n += max(1, (len(rows) + LINES_PER_PAGE - 1) // LINES_PER_PAGE)
    return n


def page_cover(pdf, files: list[Path], page: int, total: int):
    fig = new_page("Test 48 — Go source", "Every *.go in apps/aai/test48. Engine credit lives in layers/parallel.")
    lines = [
        "This PDF is the harness, not the Jacobian. Credit math is in",
        "layers/parallel (TrainStackMSE, OpenSplitTape, HemispheresFrom).",
        "",
        f"{'file':<22} {'lines':>6}  role",
        "-" * 72,
    ]
    roles = {
        "test48.go": "Lucy loop, jobs, xor/sine/copy, flags",
        "model.go": "Sandwich: Dense stem → mid/Hemispheres → Dense head",
        "hemis.go": "20 mid Ops + View wrap + dtype parse",
        "workclock.go": "thread-CPU duty clock (RUSAGE_THREAD)",
        "test48_test.go": "mode/dtype/cameral parse tests",
    }
    total_lines = 0
    for p in files:
        n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        total_lines += n
        lines.append(f"{p.name:<22} {n:6d}  {roles.get(p.name, '')}")
    lines += [
        "-" * 72,
        f"{'total':<22} {total_lines:6d}",
        "",
        "Arch names: nHemi=1 → Dense (one mid Op), 2 → Bicameral, 3 → Tricameral.",
        "CombineAdd: h = h1 + h2 + … + hn. Extra cams are extra merge, not free Acc.",
        "Weight dtype is FormatNone storage; activations stay float32.",
    ]
    y = 0.90
    for line in lines:
        fig.text(0.06, y, line, fontsize=8.2, color=SLATE, family="DejaVu Sans Mono", va="top")
        y -= 0.028
    footer(fig, page, total, "source")
    pdf.savefig(fig)
    plt.close(fig)


def emit_file(pdf, path: Path, start_page: int, total: int) -> int:
    rows = paginate_file(path)
    chunks = [rows[i:i + LINES_PER_PAGE] for i in range(0, len(rows), LINES_PER_PAGE)]
    page = start_page
    n_chunks = len(chunks)
    for ci, chunk in enumerate(chunks):
        suffix = f"  ({ci + 1}/{n_chunks})" if n_chunks > 1 else ""
        fig = new_page(path.name + suffix, str(path))
        y = TOP
        for ln, text in chunk:
            draw_line(fig, y, ln, text)
            y -= LINE_H
        footer(fig, page, total, path.name)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1
    return page


def main():
    ap = argparse.ArgumentParser(description="Pack test48 Go sources into a PDF")
    ap.add_argument("--out", default="test48_code.pdf")
    ap.add_argument("--root", default="")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    root = Path(args.root) if args.root else here
    out = Path(args.out) if os.path.isabs(args.out) else here / args.out
    files = collect_go(root)
    if not files:
        sys.exit(f"no *.go under {root}")
    total = count_pages(files)
    print(f"writing {out}  ({len(files)} files, {total} pages) …", flush=True)
    with PdfPages(out) as pdf:
        page_cover(pdf, files, 1, total)
        page = 2
        for p in files:
            page = emit_file(pdf, p, page, total)
        d = pdf.infodict()
        d["Title"] = "Welvet test48 Go source"
        d["Author"] = "welvet test48"
        d["Subject"] = "Harness sources for Lucy credit × layer × dtype"
    print(f"PDF  {out}  ({out.stat().st_size / 1e3:.0f} KB, {total} pages)")


if __name__ == "__main__":
    main()
