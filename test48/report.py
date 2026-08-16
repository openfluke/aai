#!/usr/bin/env python3
"""Test48 Lucy report — tide / live_mnist axes, PDF + graphs.

Streams test48_results.json (windows stripped), then writes a multi-page PDF
of domain winners, Pareto fronts, vs-StepBP, and honesty checks.

    .venv/bin/python report.py
    .venv/bin/python report.py --json test48_results.json --out test48_report.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-test48")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAVY = "#1a365d"
TEAL = "#2c7a7b"
GOLD = "#b7791f"
RED = "#c53030"
SLATE = "#2d3748"
MUTED = "#718096"

SHORT_MODE = {
    "NormalBP": "NormalBP",
    "StepBP": "StepBP",
    "Tween": "Tween",
    "StepTween": "StepTween",
    "TweenChain": "TweenChain",
    "StepTweenChain": "StepTweenChain",
    "TweenSplit": "TweenSplit",
    "StepTweenSplit": "StepTweenSplit",
    "TweenAlt": "TweenAlt",
    "StepTweenAlt": "StepTweenAlt",
    "TweenSplitHeadProxy": "HeadProxy",
    "TweenSplitLinear": "Linear",
    "TweenSplitFastProxy": "FastProxy",
    "TweenSplitLinearCache": "LinearCache",
    "TweenSplitHeadProxyAsync": "ProxyAsync",
    "TweenSplitSparse": "Sparse",
}

FAMILY = {
    "NormalBP": "BP",
    "StepBP": "BP",
    "Tween": "Tween",
    "StepTween": "Tween",
    "TweenChain": "TweenChain",
    "StepTweenChain": "TweenChain",
    "TweenSplit": "TweenSplit",
    "StepTweenSplit": "TweenSplit",
    "TweenAlt": "TweenAlt",
    "StepTweenAlt": "TweenAlt",
    "TweenSplitHeadProxy": "HeadProxy",
    "TweenSplitLinear": "Linear",
    "TweenSplitFastProxy": "FastProxy",
    "TweenSplitLinearCache": "LinearCache",
    "TweenSplitHeadProxyAsync": "ProxyAsync",
    "TweenSplitSparse": "Sparse",
}

HEADLINE = ["StepBP", "FastProxy", "Sparse", "HeadProxy", "Linear", "TweenSplit", "Tween", "LinearCache"]

MODE_ORDER = [
    "NormalBP", "StepBP",
    "Tween", "StepTween",
    "TweenChain", "StepTweenChain",
    "TweenSplit", "StepTweenSplit",
    "TweenAlt", "StepTweenAlt",
    "HeadProxy", "Linear", "FastProxy", "LinearCache", "ProxyAsync", "Sparse",
]

MODE_ABBR = {
    "NormalBP": "NBP", "StepBP": "SBP",
    "Tween": "Tw", "StepTween": "STw",
    "TweenChain": "TC", "StepTweenChain": "STC",
    "TweenSplit": "TS", "StepTweenSplit": "STS",
    "TweenAlt": "TA", "StepTweenAlt": "STA",
    "HeadProxy": "HP", "Linear": "Lin",
    "FastProxy": "FP", "LinearCache": "LC",
    "ProxyAsync": "PA", "Sparse": "Sp",
}

ABBR_NOTE = (
    "NBP=NormalBP  SBP=StepBP  Tw=Tween  STw=StepTween  TC=TweenChain  STC=StepTweenChain  "
    "TS=TweenSplit  STS=StepTweenSplit  TA=TweenAlt  STA=StepTweenAlt  HP=HeadProxy  "
    "Lin=Linear  FP=FastProxy  LC=LinearCache  PA=ProxyAsync  Sp=Sparse"
)

MAX_TABLE_ROWS = 22

DTYPE_BITS = {
    "float64": 64, "float32": 32, "float16": 16, "bfloat16": 16,
    "fp8e4m3": 8, "fp8e5m2": 8,
    "int64": 64, "int32": 32, "int16": 16, "int8": 8,
    "uint64": 64, "uint32": 32, "uint16": 16, "uint8": 8,
    "int4": 4, "uint4": 4, "fp4": 4, "nf4": 4,
    "int2": 2, "uint2": 2, "ternary": 2, "binary": 1,
    "int": 64, "uint": 64, "uintptr": 64,
    "complex64": 64, "complex128": 128,
    "fp6": 6, "int6": 6, "uint6": 6, "int5": 5, "uint5": 5, "int3": 3, "uint3": 3,
}

FLOOR_LAYERS = {"cnn1", "convt1"}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def iter_rows(path: str):
    dec = json.JSONDecoder()
    with open(path, "r") as f:
        buf = ""
        while True:
            piece = f.read(1 << 20)
            if not piece:
                raise ValueError(f"{path}: no rows array")
            buf += piece
            i = buf.find('"rows"')
            if i < 0:
                continue
            j = buf.find("[", i)
            if j >= 0:
                buf = buf[j + 1 :]
                break
        n = 0
        t0 = time.time()
        while True:
            buf = buf.lstrip(" \r\n\t,")
            while True:
                if buf.startswith("]"):
                    return
                try:
                    obj, end = dec.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    more = f.read(1 << 20)
                    if not more:
                        return
                    buf += more
            yield obj
            n += 1
            if n % 5000 == 0:
                print(f"  streamed {n} rows  ({time.time() - t0:.1f}s)", flush=True)
            buf = buf[end:]


def _f(x, default=float("nan")):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def flatten_row(obj: dict) -> dict | None:
    lucy = obj.get("lucy") or {}
    if isinstance(lucy, dict):
        lucy.pop("windows", None)
        lucy.pop("soft_acc_blocks", None)
        lucy.pop("phase_blocks", None)
        lucy.pop("switch_blocks", None)
    err = obj.get("error") or ""
    mode = obj.get("mode") or ""
    short = SHORT_MODE.get(mode, mode)
    ram_kib = _f(obj.get("ram_kib"), 0.0)
    w_mib = _f(lucy.get("weight_mib"))
    if not w_mib or w_mib <= 0:
        w_mib = ram_kib / 1024.0 if ram_kib > 0 else float("nan")
    score = _f(obj.get("score"))
    soft = _f(obj.get("soft_acc"))
    avail = _f(obj.get("availability"))
    tput = _f(obj.get("throughput"))
    acc = _f(obj.get("acc"))
    mobile = _f(lucy.get("mobile_score"))
    if (not mobile or math.isnan(mobile)) and w_mib and w_mib > 1e-12 and not math.isnan(score):
        mobile = score / w_mib
    zd = _f(lucy.get("zero_downtime"))
    if math.isnan(zd) and not math.isnan(soft) and not math.isnan(avail):
        zd = soft * avail / 100.0
    dur_ns = lucy.get("duration") or 0
    try:
        dur_s = float(dur_ns) / 1e9 if dur_ns else float("nan")
    except (TypeError, ValueError):
        dur_s = float("nan")
    acc_per_sec = _f(lucy.get("acc_per_sec"))
    if math.isnan(acc_per_sec) and dur_s and dur_s > 0 and not math.isnan(acc):
        acc_per_sec = acc / dur_s
    mobile_acc_per_sec = _f(lucy.get("mobile_acc_per_sec"))
    if math.isnan(mobile_acc_per_sec) and w_mib and w_mib > 1e-12 and not math.isnan(acc_per_sec):
        mobile_acc_per_sec = acc_per_sec / w_mib
    collapsed = (
        not err
        and acc < 16
        and _f(obj.get("consistency"), 0) >= 70
        and _f(obj.get("stability"), 0) >= 90
    )
    layer = obj.get("layer") or ""
    task = obj.get("task") or ""
    floor = (
        task == "copy"
        and layer in FLOOR_LAYERS
        and abs(acc - 47.7) < 1.0
        and abs(soft - 50.0) < 1.5
    )
    return {
        "task": task,
        "dtype": obj.get("dtype") or "float32",
        "layer": layer,
        "arch": obj.get("arch") or "",
        "mode": mode,
        "short": short,
        "family": FAMILY.get(mode, short),
        "cell": f"{obj.get('dtype','')}/{layer}/{obj.get('arch','')}/{short}",
        "id": f"{task}/{obj.get('dtype','')}/{layer}/{obj.get('arch','')}/{short}",
        "acc": acc,
        "soft": soft,
        "adapt": _f(obj.get("adapt_pct")),
        "avail": avail,
        "stab": _f(obj.get("stability")),
        "cons": _f(obj.get("consistency")),
        "tput": tput,
        "score": score,
        "steps": _f(obj.get("steps"), 0),
        "infer_ms": _f(obj.get("infer_ms")),
        "train_ms": _f(obj.get("train_ms")),
        "ram_kib": ram_kib,
        "weight_mib": w_mib,
        "mobile_score": mobile,
        "mobile_tput": _f(lucy.get("mobile_throughput")),
        "mobile_avail": _f(lucy.get("mobile_availability")),
        "mobile_acc": _f(lucy.get("mobile_accuracy")),
        "zero_dt": zd,
        "t25": _f(lucy.get("time_to_acc25_sec")),
        "t50": _f(lucy.get("time_to_acc50_sec")),
        "acc_per_sec": acc_per_sec,
        "mobile_acc_per_sec": mobile_acc_per_sec,
        "err": err,
        "ok": not bool(err),
        "collapsed": bool(collapsed),
        "floor": bool(floor),
        "bits": DTYPE_BITS.get(obj.get("dtype") or "", 32),
    }


def load_df(json_path: str, cache_path: str, refresh: bool) -> tuple[pd.DataFrame, dict]:
    meta = {}
    if (
        not refresh
        and cache_path
        and os.path.exists(cache_path)
        and os.path.exists(json_path)
        and os.path.getmtime(cache_path) >= os.path.getmtime(json_path)
    ):
        print(f"cache hit {cache_path}", flush=True)
        df = pd.read_pickle(cache_path)
        meta_path = cache_path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return df, meta

    print(f"streaming {json_path} ({os.path.getsize(json_path) / 1e9:.2f} GB)…", flush=True)
    rows = []
    for obj in iter_rows(json_path):
        flat = flatten_row(obj)
        if flat:
            rows.append(flat)
    df = pd.DataFrame(rows)
    # header meta: peek start of file
    with open(json_path) as f:
        head = f.read(800)
    for key in ("duration", "engine", "dtypes", "workers", "score_formula", "adapt_windows"):
        # best-effort; full meta is in the object but we skipped it
        pass
    meta = {
        "n_raw": len(df),
        "json": os.path.abspath(json_path),
        "head": head[:400],
    }
    if cache_path:
        df.to_pickle(cache_path)
        with open(cache_path + ".meta.json", "w") as f:
            json.dump(meta, f)
        print(f"cached {len(df)} rows → {cache_path}", flush=True)
    return df, meta


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["ok"]].copy()


def pick_best(df: pd.DataFrame, col: str, higher: bool = True, min_n: int = 1) -> pd.Series | None:
    s = df.dropna(subset=[col])
    if s.empty or len(s) < min_n:
        return None
    idx = s[col].idxmax() if higher else s[col].idxmin()
    return s.loc[idx]


def cell_line(r: pd.Series) -> str:
    return f"{r['id']}   acc={r['acc']:.1f} soft={r['soft']:.1f} avail={r['avail']:.1f} score={r['score']:.0f}"


def pareto_mask(x: np.ndarray, y: np.ndarray, xmax=True, ymax=True) -> np.ndarray:
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dx = (x >= x[i]) if xmax else (x <= x[i])
        dy = (y >= y[i]) if ymax else (y <= y[i])
        dominated = dx & dy
        dominated[i] = False
        # strictly better in one dim
        better = dominated & ((x != x[i]) | (y != y[i]))
        if np.any(better):
            keep[i] = False
    return keep


def vs_bp(df: pd.DataFrame) -> pd.DataFrame:
    bp = df[df["short"] == "StepBP"][
        ["task", "dtype", "layer", "arch", "acc", "soft", "avail", "score", "id"]
    ].rename(
        columns={
            "acc": "bp_acc",
            "soft": "bp_soft",
            "avail": "bp_avail",
            "score": "bp_score",
            "id": "bp_id",
        }
    )
    m = df.merge(bp, on=["task", "dtype", "layer", "arch"], how="inner")
    m = m[m["short"] != "StepBP"]
    m["d_acc"] = m["acc"] - m["bp_acc"]
    m["d_soft"] = m["soft"] - m["bp_soft"]
    m["d_avail"] = m["avail"] - m["bp_avail"]
    m["d_score"] = m["score"] - m["bp_score"]
    return m


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def new_page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=NAVY, x=0.06, ha="left", y=0.97)
    if subtitle:
        fig.text(0.06, 0.935, subtitle, fontsize=8, color=MUTED, ha="left")
    return fig


def footer(fig, page: int, total: int):
    fig.text(0.06, 0.02, "welvet test48  ·  Lucy = tide / live_mnist  ·  Score = Tput × Avail × SoftAcc / 10_000",
             fontsize=6.5, color=MUTED)
    fig.text(0.94, 0.02, f"{page} / {total}", fontsize=7, color=MUTED, ha="right")


def text_block(fig, lines: list[str], x=0.06, y=0.90, dy=0.028, size=8.5, color=SLATE):
    for i, line in enumerate(lines):
        fig.text(x, y - i * dy, line, fontsize=size, color=color, family="DejaVu Sans", va="top")


def _col_widths(headers, rows):
    n = len(headers)
    lens = [len(str(h)) for h in headers]
    for row in rows:
        for c in range(n):
            lens[c] = max(lens[c], len(str(row[c])))
    id_names = {"cell", "winner", "id", "#1", "#2", "#3", "layer", "mode", "dtype", "arch", "task"}
    weights = []
    for c, h in enumerate(headers):
        hl = h.lower()
        if (
            h in ("#", "n")
            or len(h) <= 3
            or hl in ("acc", "soft", "avail", "adapt", "tput", "score", "kib", "mobile")
            or hl.endswith(" acc")
            or hl.endswith(" score")
            or "−bp" in hl
            or "-bp" in hl
        ):
            weights.append(max(3, min(lens[c], 8)))
        elif hl in id_names or "winner" in hl or "mode" in hl or "layer" in hl or "arch" in hl:
            weights.append(max(lens[c], 8) * 1.25)
        else:
            weights.append(max(lens[c], 6))
    s = sum(weights) or 1
    return [w / s for w in weights]


def table_page(fig, headers, rows, col_w=None, y0=0.88, fontsize=6.2):
    # leave room for the footer so long tables cannot paint over it
    ax = fig.add_axes([0.03, 0.075, 0.94, max(0.12, y0 - 0.085)])
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "(empty)", ha="center")
        return
    widths = col_w or _col_widths(headers, rows)
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper left",
        cellLoc="left",
        colWidths=widths,
    )
    tbl.auto_set_font_size(False)
    n_rows = len(rows)
    fs = fontsize
    rs = 1.22
    if n_rows > 24:
        fs = min(fs, 5.6)
        rs = 1.10
    if n_rows > 30:
        fs = min(fs, 5.2)
        rs = 1.02
    tbl.set_fontsize(fs)
    tbl.scale(1, rs)
    id_idx = {i for i, h in enumerate(headers) if h.lower() in {
        "cell", "winner", "layer", "mode", "dtype", "arch", "task", "#1", "#2", "#3",
        "domain",
    }}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        cell.PAD = 0.012
        if c in id_idx:
            cell.get_text().set_fontfamily("DejaVu Sans Mono")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")


def emit_tables(pdf, title, note, headers, rows, n_fn, fontsize=5.6):
    """Write one or more PDF pages so long tables never paint the footer."""
    chunks = [rows[i : i + MAX_TABLE_ROWS] for i in range(0, max(len(rows), 1), MAX_TABLE_ROWS)]
    if not rows:
        chunks = [[]]
    n_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        page, total = n_fn()
        suffix = f"  ({i + 1}/{n_chunks})" if n_chunks > 1 else ""
        fig = new_page(title + suffix, note)
        table_page(fig, headers, chunk, fontsize=fontsize)
        footer(fig, page, total)
        pdf.savefig(fig)
        plt.close(fig)


def heatmap(ax, pivot: pd.DataFrame, title: str, cmap="YlGnBu", vmin=None, vmax=None, cbar=True):
    if pivot.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        return
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=75, ha="right", fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=6)
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def barh_means(ax, series: pd.Series, title: str, color=TEAL):
    s = series.sort_values()
    ax.barh(s.index.astype(str), s.values, color=color, height=0.7)
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_cover(pdf, df, meta, page, total):
    fig = new_page("Test 48 — Lucy credit × layer × dtype report")
    n = len(df)
    n_ok = int(df["ok"].sum())
    n_err = n - n_ok
    n_col = int(df["collapsed"].sum())
    n_floor = int(df["floor"].sum())
    tasks = sorted(df["task"].unique())
    dtypes = list(OrderedDict.fromkeys(df["dtype"]))
    lines = [
        "Same measuring as tide / live_mnist / test41: SoftAcc, hard Acc, Availability,",
        "AdaptPct, Throughput, Score, ZeroDowntime, MobileScore (Score / WeightMiB).",
        "",
        f"Jobs          {n:,}   ok {n_ok:,}   errors {n_err:,}   collapsed {n_col:,}   copy-floor {n_floor:,}",
        f"Tasks         {', '.join(tasks)}",
        f"Dtypes        {len(dtypes)}  (activations stay float32; this axis is weight storage)",
        f"Layers        {df['layer'].nunique()}    arches {', '.join(sorted(df['arch'].unique()))}",
        f"Modes         {df['mode'].nunique()}     families {df['family'].nunique()}",
        "",
        "Score = Throughput × Availability × SoftAcc / 10_000",
        "Availability = InferMs / (InferMs + TrainMs) × 100",
        "MobileScore = Score / WeightMiB     ZeroDowntime = SoftAcc × Availability / 100",
        "",
        "Two questions. Acc / SoftAcc / AdaptPct = did it learn. Score / Avail = duty clock.",
        "Sparse winning Score while losing Acc is skipping GEMVs, not a better chain rule.",
        "XOR Acc is a 50/75/100 lottery. cnn1/convt1 copy 47.7/50 is a View-wrap floor.",
        "13% Acc + Cons~75 + Stab~98 = collapsed job. LinearCache is dead on sine.",
        "",
        "This sweep is the short perm race (2s, shared CPU). Lucy-honest Score needs",
        "  -dtypes float32 -duration 10s -workers 1",
        "Do not write “we beat backprop” from Score or from a 2s XOR cell.",
    ]
    text_block(fig, lines, y=0.90, dy=0.032, size=9)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_bests(pdf, df, page, total, title, specs, subtitle=""):
    """specs: list of (label, column, higher)."""
    fig = new_page(title, subtitle)
    headers = ["domain", "task", "dtype", "layer", "arch", "mode", "acc", "soft", "avail", "adapt", "score", "mobile", "KiB"]
    rows = []
    for label, col, higher in specs:
        r = pick_best(df, col, higher=higher)
        if r is None:
            rows.append([label, "—", "—", "—", "—", "—", "", "", "", "", "", "", ""])
            continue
        rows.append([
            label,
            r["task"],
            r["dtype"],
            r["layer"],
            r["arch"],
            r["short"],
            f"{r['acc']:.1f}",
            f"{r['soft']:.1f}",
            f"{r['avail']:.1f}",
            f"{r['adapt']:.1f}",
            f"{r['score']:.0f}",
            f"{r['mobile_score']:.0f}",
            f"{r['ram_kib']:.1f}",
        ])
    table_page(fig, headers, rows, fontsize=6.5)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_leaderboard(pdf, df, col, title, page, total, n=20, note=""):
    fig = new_page(title, note)
    s = df.dropna(subset=[col]).sort_values(col, ascending=False).head(n)
    metric_hdrs = ["acc", "soft", "avail", "adapt", "tput", "score"]
    extra = [] if col in metric_hdrs else [col]
    headers = ["#", "task", "dtype", "layer", "arch", "mode"] + metric_hdrs + extra
    rows = []
    for i, (_, r) in enumerate(s.iterrows(), 1):
        row = [
            str(i),
            r["task"],
            r["dtype"],
            r["layer"],
            r["arch"],
            r["short"],
            f"{r['acc']:.1f}",
            f"{r['soft']:.1f}",
            f"{r['avail']:.1f}",
            f"{r['adapt']:.1f}",
            f"{r['tput']:.0f}",
            f"{r['score']:.0f}",
        ]
        if extra:
            v = r[col]
            row.append(f"{v:.1f}" if abs(v) < 1e6 else f"{v:.0f}")
        rows.append(row)
    table_page(fig, headers, rows, fontsize=6.4)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def _top_rows(sub: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    return sub.dropna(subset=[col]).sort_values(col, ascending=False).head(n)


def _fmt_slot(r: pd.Series, col: str) -> str:
    v = r[col]
    vs = f"{v:.0f}" if abs(v) >= 20 else f"{v:.1f}"
    return f"{r['task']} {r['dtype']} {r['layer']} {r['arch']} {r['short']}  {vs}"


def page_topn_by_group(pdf, df, group, col, title, n_fn, *, top=3, note="", group_order=None):
    """One row per group: top-n cells on `col`. Identity columns follow the grouping axis."""
    peers = {
        "dtype": [("task", "task"), ("layer", "layer"), ("short", "mode")],
        "short": [("task", "task"), ("dtype", "dtype"), ("layer", "layer")],
        "layer": [("task", "task"), ("dtype", "dtype"), ("short", "mode")],
    }.get(group, [("task", "task"), ("dtype", "dtype"), ("layer", "layer"), ("short", "mode")])
    label = "mode" if group == "short" else group
    headers = [label]
    for i in range(1, top + 1):
        headers += [f"#{i} {hdr}" for _, hdr in peers]
        headers.append(f"#{i} {col}")
    rows = []
    keys = group_order if group_order is not None else list(dict.fromkeys(df[group].tolist()))
    for g in keys:
        sub = df[df[group] == g]
        found = _top_rows(sub, col, top)
        if found.empty:
            continue
        slots = [str(g)]
        shown = 0
        for _, r in found.iterrows():
            for src, _ in peers:
                slots.append(str(r[src]))
            v = r[col]
            slots.append(f"{v:.0f}" if abs(v) >= 20 else f"{v:.1f}")
            shown += 1
        while shown < top:
            slots += ["—"] * (len(peers) + 1)
            shown += 1
        rows.append(slots)
    emit_tables(pdf, title, note, headers, rows, n_fn, fontsize=5.6)


def page_task_topn(pdf, df, page, total, n=10):
    fig = new_page("Top 10 per task — Acc and Score", "Every numerical type and every mode sit in these pools. Acc = learn, Score = clock.")
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    # 3 tasks × 2 metrics as small tables via text is messy; 6 mini tables
    for i, task in enumerate(tasks):
        sub = df[df["task"] == task]
        for j, (col, name) in enumerate((("acc", "Acc"), ("score", "Score"))):
            ax = fig.add_subplot(3, 2, i * 2 + j + 1)
            ax.axis("off")
            ax.set_title(f"{task}  top {n} {name}", loc="left", fontsize=9, color=NAVY)
            top = _top_rows(sub, col, n)
            lines = []
            for k, (_, r) in enumerate(top.iterrows(), 1):
                lines.append(
                    f"{k:2d}  {r['dtype']:<12} {r['layer']:<14} {r['arch']:<11} {r['short']:<12}  "
                    f"acc={r['acc']:5.1f}  sc={r['score']:6.0f}"
                )
            ax.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=6.2, family="DejaVu Sans Mono",
                    transform=ax.transAxes, color=SLATE)
    fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.92])
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_task_mode_top(pdf, df, page, total):
    fig = new_page("Best cell per task × mode (hard Acc)", "16 modes × 3 tasks. Winner can be any dtype/layer/arch.")
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    headers = ["mode"] + [f"{t} winner (acc)" for t in tasks]
    rows = []
    for mode in MODE_ORDER:
        slots = [mode]
        empty = True
        for task in tasks:
            sub = df[(df["short"] == mode) & (df["task"] == task)]
            r = pick_best(sub, "acc")
            if r is None:
                slots.append("—")
            else:
                empty = False
                slots.append(f"{r['dtype']} {r['layer']} {r['arch']} {r['acc']:.0f}")
        if not empty:
            rows.append(slots)
    table_page(fig, headers, rows)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_task_mode_top_score(pdf, df, page, total):
    fig = new_page("Best cell per task × mode (Lucy Score)", "Compare to the Acc page. If Sparse everywhere, that is Avail.")
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    headers = ["mode"] + [f"{t} winner (score)" for t in tasks]
    rows = []
    for mode in MODE_ORDER:
        slots = [mode]
        empty = True
        for task in tasks:
            sub = df[(df["short"] == mode) & (df["task"] == task)]
            r = pick_best(sub, "score")
            if r is None:
                slots.append("—")
            else:
                empty = False
                slots.append(f"{r['dtype']} {r['layer']} {r['arch']} {r['score']:.0f}")
        if not empty:
            rows.append(slots)
    table_page(fig, headers, rows)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_mode_bars(pdf, df, page, total):
    fig = new_page("Mode means — Acc vs Score (two questions)", "Mean over layer × arch × dtype. Score crowns skip-GEMV.")
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    for i, task in enumerate(tasks):
        sub = df[df["task"] == task]
        g = sub.groupby("short", observed=True)[["acc", "score", "avail", "soft"]].mean()
        g = g.reindex([m for m in MODE_ORDER if m in g.index])
        ax1 = fig.add_subplot(2, 3, i + 1)
        ax1.bar(range(len(g)), g["acc"], color=TEAL, width=0.7)
        ax1.set_xticks(range(len(g)))
        ax1.set_xticklabels(g.index, rotation=75, ha="right", fontsize=6)
        ax1.set_title(f"{task}  mean Acc")
        ax1.set_ylim(0, 100)
        ax2 = fig.add_subplot(2, 3, i + 4)
        ax2.bar(range(len(g)), g["score"], color=GOLD, width=0.7)
        ax2.set_xticks(range(len(g)))
        ax2.set_xticklabels(g.index, rotation=75, ha="right", fontsize=6)
        ax2.set_title(f"{task}  mean Score")
    fig.tight_layout(rect=[0.03, 0.05, 0.99, 0.92])
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_heat_dtype_mode(pdf, df, page, total):
    fig = new_page("Heatmap — mean Acc  dtype × mode", "Sine is the adaptive task. White/yellow = low Acc.")
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(1, 3, i + 1)
        sub = df[df["task"] == task]
        pv = sub.pivot_table(index="dtype", columns="short", values="acc", aggfunc="mean")
        cols = [m for m in MODE_ORDER if m in pv.columns]
        pv = pv[cols]
        heatmap(ax, pv, task, cmap="RdYlGn", vmin=20, vmax=100, cbar=(i == 2))
    fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.92])
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_heat_layer_mode(pdf, df, page, total):
    fig = new_page("Heatmap — mean Acc  layer × mode", "cnn1/convt1 copy often 47.7 for every mode (floor).")
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(1, 3, i + 1)
        sub = df[df["task"] == task]
        pv = sub.pivot_table(index="layer", columns="short", values="acc", aggfunc="mean")
        cols = [m for m in MODE_ORDER if m in pv.columns]
        pv = pv[cols]
        heatmap(ax, pv, task, cmap="RdYlGn", vmin=20, vmax=100, cbar=(i == 2))
    fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.92])
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_pareto(pdf, df, page, total):
    fig = new_page("Pareto fronts", "Undominated edge. Score vs Acc: Sparse sits high-Score / mid-Acc.")
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    pairs = [("acc", "avail", "Acc vs Avail"), ("acc", "score", "Acc vs Score"), ("soft", "avail", "SoftAcc vs Avail")]
    for r, (xcol, ycol, ttl) in enumerate(pairs):
        for c, task in enumerate(tasks):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1)
            sub = df[df["task"] == task].dropna(subset=[xcol, ycol])
            if sub.empty:
                continue
            # subsample for scatter if huge
            plot = sub.sample(min(len(sub), 4000), random_state=1) if len(sub) > 4000 else sub
            ax.scatter(plot[xcol], plot[ycol], s=6, alpha=0.15, c=MUTED, linewidths=0)
            # pareto on all
            mask = pareto_mask(sub[xcol].values, sub[ycol].values)
            front = sub[mask]
            ax.scatter(front[xcol], front[ycol], s=18, c=GOLD, zorder=3, label="pareto")
            ax.set_title(f"{task}  {ttl}", fontsize=8)
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(ycol)
            if r == 2:
                ax.set_xlabel(xcol)
    fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.92])
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_vs_bp(pdf, vs, page, total):
    fig = new_page("vs StepBP — AccΔ and ScoreΔ", "Matched task×dtype×layer×arch. ScoreΔ almost always + (Avail). AccΔ is the honest column.")
    focus = ["FastProxy", "Sparse", "HeadProxy", "Linear", "ProxyAsync", "LinearCache", "TweenSplit", "Tween"]
    tasks = [t for t in ("xor", "sine", "copy") if t in set(vs["task"])]
    headers = ["task", "mode", "n", "mean AccΔ", "Acc win%", "mean SoftΔ", "mean AvailΔ", "mean ScoreΔ", "Score win%"]
    rows = []
    for task in tasks:
        for m in focus:
            s = vs[(vs["task"] == task) & (vs["short"] == m)]
            if s.empty:
                continue
            n = len(s)
            aw = 100 * (s["d_acc"] > 0.5).mean()
            sw = 100 * (s["d_score"] > 1).mean()
            rows.append([
                task, m, str(n),
                f"{s['d_acc'].mean():+.1f}",
                f"{aw:.0f}",
                f"{s['d_soft'].mean():+.1f}",
                f"{s['d_avail'].mean():+.1f}",
                f"{s['d_score'].mean():+.0f}",
                f"{sw:.0f}",
            ])
    table_page(fig, headers, rows)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_vs_bp_bars(pdf, vs, page, total):
    fig = new_page("vs StepBP — mean AccΔ by mode × task", "FastProxy Acc-win on copy is the cleanest proxy>chain signal. Sparse Acc often loses.")
    focus = ["FastProxy", "Sparse", "HeadProxy", "Linear", "TweenSplit", "Tween", "LinearCache"]
    tasks = [t for t in ("xor", "sine", "copy") if t in set(vs["task"])]
    x = np.arange(len(focus))
    w = 0.25
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])
    for i, task in enumerate(tasks):
        means = []
        for m in focus:
            s = vs[(vs["task"] == task) & (vs["short"] == m)]
            means.append(s["d_acc"].mean() if len(s) else 0)
        ax.bar(x + (i - 1) * w, means, w, label=task)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(focus, rotation=20)
    ax.set_ylabel("mean AccΔ vs StepBP")
    ax.legend()
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_dtype_rank(pdf, df, page, total):
    fig = new_page("Dtype ranking — mean Acc and mean Score", "Weight storage only. Low-bit integer may collapse Acc while Score stays (Avail).")
    g = df.groupby("dtype", observed=True)[["acc", "score", "soft", "avail", "ram_kib"]].mean()
    # order by bits then name
    g["bits"] = [DTYPE_BITS.get(i, 32) for i in g.index]
    g = g.sort_values(["bits", "acc"], ascending=[True, False])
    ax1 = fig.add_axes([0.08, 0.10, 0.40, 0.78])
    ax1.barh(g.index.astype(str), g["acc"], color=TEAL)
    ax1.set_title("mean Acc")
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis="y", labelsize=6)
    ax2 = fig.add_axes([0.56, 0.10, 0.40, 0.78])
    ax2.barh(g.index.astype(str), g["score"], color=GOLD)
    ax2.set_title("mean Score")
    ax2.tick_params(axis="y", labelsize=6)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_arch_layer(pdf, df, page, total):
    fig = new_page("Arch and layer — mean Acc", "Cameral = extra merge, not automatically better Acc on 2s toys.")
    ax1 = fig.add_subplot(1, 2, 1)
    g = df.groupby(["task", "arch"])["acc"].mean().unstack()
    g.plot(kind="bar", ax=ax1, rot=0)
    ax1.set_title("mean Acc by arch")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=7)
    ax2 = fig.add_subplot(1, 2, 2)
    layer_acc = df.groupby("layer")["acc"].mean().sort_values()
    ax2.barh(layer_acc.index.astype(str), layer_acc.values, color=TEAL)
    ax2.set_title("mean Acc by layer")
    ax2.set_xlim(0, 100)
    ax2.tick_params(axis="y", labelsize=6)
    fig.tight_layout(rect=[0.03, 0.05, 0.99, 0.92])
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def _honesty_rows(df, group, delta=False):
    modes = [m for m in MODE_ORDER if m in set(df["short"])]
    g = df.groupby([group, "short"], observed=True)["acc"].mean().unstack()
    if delta:
        if "StepBP" not in g.columns:
            return [group], []
        bp = g["StepBP"]
        g = g.sub(bp, axis=0)
        modes = [m for m in modes if m != "StepBP" and m in g.columns]
    else:
        modes = [m for m in modes if m in g.columns]
    headers = [group] + [MODE_ABBR.get(m, m) for m in modes]
    keys = list(dict.fromkeys(df[group].tolist()))
    rows = []
    for k in keys:
        row = [str(k)]
        for m in modes:
            if k in g.index and m in g.columns and pd.notna(g.loc[k, m]):
                v = float(g.loc[k, m])
                row.append(f"{v:+.1f}" if delta else f"{v:.1f}")
            else:
                row.append("—")
        rows.append(row)
    return headers, rows


def page_honesty_matrix(pdf, df, group, title, n_fn, note="", delta=False):
    """Mean Acc (or AccΔ vs StepBP) of every training mode on every layer or dtype."""
    headers, rows = _honesty_rows(df, group, delta=delta)
    emit_tables(pdf, title, note or ABBR_NOTE, headers, rows, n_fn, fontsize=5.3)


def page_honesty_delta_heat(pdf, vs, n_fn):
    """AccΔ vs StepBP for every non-BP mode × every layer and every dtype, per task."""
    modes = [m for m in MODE_ORDER if m != "StepBP"]
    specs = [
        ("layer", "Honesty — AccΔ vs StepBP, every mode × every layer",
         "Matched cells. Green = beats backprop Acc. XOR is a 50/75/100 lottery."),
        ("dtype", "Honesty — AccΔ vs StepBP, every mode × every dtype",
         "Same matched cells, grouped by weight storage. Learning claim, not Score."),
    ]
    tasks = [t for t in ("xor", "sine", "copy") if t in set(vs["task"])]
    for grp, title, note in specs:
        page, total = n_fn()
        fig = new_page(title, note)
        for i, task in enumerate(tasks):
            ax = fig.add_subplot(1, len(tasks), i + 1)
            sub = vs[vs["task"] == task]
            pv = sub.pivot_table(index=grp, columns="short", values="d_acc", aggfunc="mean")
            cols = [m for m in modes if m in pv.columns]
            pv = pv.reindex(columns=cols)
            pv.columns = [MODE_ABBR.get(c, c) for c in pv.columns]
            heatmap(ax, pv, task, cmap="RdYlGn", vmin=-15, vmax=15, cbar=(i == len(tasks) - 1))
        fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.92])
        footer(fig, page, total)
        pdf.savefig(fig)
        plt.close(fig)


def page_honesty(pdf, df, vs, page, total):
    fig = new_page("Honesty checks", "If Score and Acc disagree, Acc is the learning claim.")
    sine = df[df["task"] == "sine"]
    lc = sine[sine["short"] == "LinearCache"]
    tw = sine[sine["short"] == "Tween"]
    sp = vs[vs["short"] == "Sparse"]
    fp = vs[vs["short"] == "FastProxy"]
    lines = [
        f"LinearCache on sine:  n={len(lc)}  mean Acc={lc['acc'].mean():.1f}  mean Adapt={lc['adapt'].mean():.1f}  (dead)",
        f"Tween on sine:        n={len(tw)}  mean Acc={tw['acc'].mean():.1f}  mean Soft={tw['soft'].mean():.1f}  (broadcast)",
        f"Collapsed jobs:       {int(df['collapsed'].sum()):,}  (Acc<16 + Cons≥70 + Stab≥90)",
        f"Copy floors cnn1/convt1: {int(df['floor'].sum()):,}  (Acc~47.7 Soft~50, including StepBP)",
        "",
        "Sparse vs StepBP AccΔ by task (mean):",
    ]
    for task in ("xor", "sine", "copy"):
        s = sp[sp["task"] == task]
        f = fp[fp["task"] == task]
        if s.empty:
            continue
        lines.append(
            f"  {task:5s}  Sparse AccΔ {s['d_acc'].mean():+.1f} (win {(s['d_acc']>0.5).mean()*100:.0f}%)  "
            f"ScoreΔ {s['d_score'].mean():+.0f}   |   FastProxy AccΔ {f['d_acc'].mean():+.1f} "
            f"(win {(f['d_acc']>0.5).mean()*100:.0f}%)  ScoreΔ {f['d_score'].mean():+.0f}"
        )
    lines += [
        "",
        "Family collapse (Step* vs non-Step, mean |AccΔ|):",
    ]
    pairs = [("StepBP", "NormalBP"), ("StepTween", "Tween"), ("StepTweenChain", "TweenChain"),
             ("StepTweenSplit", "TweenSplit"), ("StepTweenAlt", "TweenAlt")]
    for a, b in pairs:
        aa = df[df["short"] == a][["task", "dtype", "layer", "arch", "acc", "score"]]
        bb = df[df["short"] == b][["task", "dtype", "layer", "arch", "acc", "score"]]
        m = aa.merge(bb, on=["task", "dtype", "layer", "arch"], suffixes=("_a", "_b"))
        if m.empty:
            continue
        lines.append(f"  {a} vs {b}:  |AccΔ|={ (m['acc_a']-m['acc_b']).abs().mean():.2f}  "
                     f"|ScoreΔ|={(m['score_a']-m['score_b']).abs().mean():.0f}  n={len(m)}")
    lines += [
        "",
        "Allowed: FastProxy matched/beat StepBP Acc on copy and unsaturated sine layers.",
        "Forbidden: Sparse is better backprop. This 2s/shared-CPU file is the 10s board.",
    ]
    text_block(fig, lines, y=0.90, dy=0.030, size=8.5)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_winners_cross(pdf, df, page, total):
    fig = new_page("Cross-axis champions (tide dash style)", "Best cell per mode by Lucy Score — caption Acc or you are reading the clock.")
    rows = []
    headers = ["mode", "task", "dtype", "layer", "arch", "acc", "soft", "avail", "score", "n"]
    for mode in MODE_ORDER:
        sub = df[df["short"] == mode]
        if sub.empty:
            continue
        r = pick_best(sub, "score")
        if r is None:
            continue
        rows.append([
            mode, r["task"], r["dtype"], r["layer"], r["arch"],
            f"{r['acc']:.1f}", f"{r['soft']:.1f}", f"{r['avail']:.1f}", f"{r['score']:.0f}",
            str(len(sub)),
        ])
    table_page(fig, headers, rows)
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_best_mode_per_dtype(pdf, df, n_fn):
    rows = []
    headers = ["dtype", "mode", "task", "layer", "arch", "acc", "score"]
    for dt, sub in df.groupby("dtype", observed=True, sort=False):
        r = pick_best(sub, "score")
        if r is None:
            continue
        rows.append([dt, r["short"], r["task"], r["layer"], r["arch"], f"{r['acc']:.1f}", f"{r['score']:.0f}"])
    emit_tables(pdf, "Best mode per dtype (by Score)", "Usually Sparse. That is Avail, not Acc.",
                headers, rows, n_fn)


def page_best_acc_mode_per_dtype(pdf, df, n_fn):
    rows = []
    headers = ["dtype", "mode", "task", "layer", "arch", "acc", "soft", "score"]
    for dt, sub in df.groupby("dtype", observed=True, sort=False):
        r = pick_best(sub, "acc")
        if r is None:
            continue
        rows.append([dt, r["short"], r["task"], r["layer"], r["arch"], f"{r['acc']:.1f}", f"{r['soft']:.1f}", f"{r['score']:.0f}"])
    emit_tables(pdf, "Best mode per dtype (by hard Acc)",
                "This is the learning ranking. Compare to the Score page.",
                headers, rows, n_fn)


def page_scatter_headline(pdf, df, page, total):
    fig = new_page("Headline modes — Acc vs Avail", "Each point = one job. Sparse shifts right (Avail). FastProxy should sit with BP Acc, higher Avail.")
    hl = ["StepBP", "FastProxy", "Sparse", "HeadProxy", "Linear", "Tween", "LinearCache"]
    colors = {
        "StepBP": NAVY, "FastProxy": TEAL, "Sparse": GOLD, "HeadProxy": "#805ad5",
        "Linear": "#3182ce", "Tween": RED, "LinearCache": MUTED,
    }
    tasks = [t for t in ("xor", "sine", "copy") if t in set(df["task"])]
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(1, 3, i + 1)
        sub = df[(df["task"] == task) & (df["short"].isin(hl))]
        for m in hl:
            s = sub[sub["short"] == m]
            if s.empty:
                continue
            if len(s) > 2500:
                s = s.sample(2500, random_state=1)
            ax.scatter(s["avail"], s["acc"], s=8, alpha=0.35, c=colors[m], label=m, linewidths=0)
        ax.set_xlabel("Availability")
        ax.set_ylabel("Acc")
        ax.set_title(task)
        ax.set_ylim(0, 105)
        if i == 2:
            ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.92])
    footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def write_winners_json(df: pd.DataFrame, vs: pd.DataFrame, path: str):
    out = {"domains": {}, "vs_stepbp": {}, "census": {
        "n": int(len(df)), "ok": int(df["ok"].sum()), "errors": int((~df["ok"]).sum()),
        "collapsed": int(df["collapsed"].sum()), "floor": int(df["floor"].sum()),
        "tasks": sorted(df["task"].unique().tolist()),
        "dtypes": list(dict.fromkeys(df["dtype"].tolist())),
    }}
    specs = [
        ("score", "score", True),
        ("acc", "acc", True),
        ("soft", "soft", True),
        ("adapt", "adapt", True),
        ("avail", "avail", True),
        ("tput", "tput", True),
        ("zero_downtime", "zero_dt", True),
        ("mobile_score", "mobile_score", True),
        ("stability", "stab", True),
        ("acc_per_sec", "acc_per_sec", True),
        ("smallest_ram", "ram_kib", False),
    ]
    for name, col, higher in specs:
        r = pick_best(df, col, higher=higher)
        if r is None:
            continue
        out["domains"][name] = {k: (None if isinstance(v, float) and math.isnan(v) else v)
                                for k, v in r.to_dict().items()}
        top = df.dropna(subset=[col]).sort_values(col, ascending=not higher).head(10)
        out.setdefault("top10", {})[name] = [
            {"id": r2["id"], "acc": float(r2["acc"]), "score": float(r2["score"]),
             "val": float(r2[col])}
            for _, r2 in top.iterrows()
        ]
    focus = ["FastProxy", "Sparse", "HeadProxy", "Linear"]
    for task in sorted(vs["task"].unique()):
        out["vs_stepbp"][task] = {}
        for m in focus:
            s = vs[(vs["task"] == task) & (vs["short"] == m)]
            if s.empty:
                continue
            out["vs_stepbp"][task][m] = {
                "n": int(len(s)),
                "mean_acc_delta": float(s["d_acc"].mean()),
                "acc_win_pct": float((s["d_acc"] > 0.5).mean() * 100),
                "mean_score_delta": float(s["d_score"].mean()),
                "score_win_pct": float((s["d_score"] > 1).mean() * 100),
            }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Test48 Lucy PDF report (tide axes)")
    ap.add_argument("--json", default="test48_results.json")
    ap.add_argument("--out", default="test48_report.pdf")
    ap.add_argument("--cache", default="test48_flat.pkl")
    ap.add_argument("--winners", default="test48_winners.json")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    json_path = args.json if os.path.isabs(args.json) else os.path.join(here, args.json)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    cache_path = args.cache if os.path.isabs(args.cache) else os.path.join(here, args.cache)
    win_path = args.winners if os.path.isabs(args.winners) else os.path.join(here, args.winners)

    if not os.path.exists(json_path) and not os.path.exists(cache_path):
        sys.exit(f"missing {json_path}")

    df, meta = load_df(json_path, cache_path, args.refresh)
    print(f"rows={len(df)} ok={int(df['ok'].sum())} tasks={sorted(df['task'].unique())}", flush=True)
    dfo = ok(df)
    vs = vs_bp(dfo)

    # quality slices
    learned = dfo[dfo["acc"] >= 70]
    sine = dfo[dfo["task"] == "sine"]

    planned = 51
    page = 0

    print(f"writing {out_path} …", flush=True)
    with PdfPages(out_path) as pdf:
        def n():
            nonlocal page
            page += 1
            return page, planned

        page_cover(pdf, df, meta, *n())
        page_bests(
            pdf, dfo, *n(),
            title="Best raw (tide Best) — one winner per Lucy axis",
            subtitle="Accuracy here is hard Acc. tide’s accuracy axis used SoftAcc; both are listed.",
            specs=[
                ("Score (Lucy clock)", "score", True),
                ("Hard Acc (learn)", "acc", True),
                ("SoftAcc (live fit)", "soft", True),
                ("AdaptPct (sine switches)", "adapt", True),
                ("Availability (duty)", "avail", True),
                ("Throughput", "tput", True),
                ("ZeroDowntime", "zero_dt", True),
                ("Stability", "stab", True),
                ("Consistency", "cons", True),
            ],
        )
        page_bests(
            pdf, dfo, *n(),
            title="Best mobile (metric / MiB) — tide BestMobile",
            subtitle="Tie-break is already inside Mobile*. Tiny KiB + high Score wins.",
            specs=[
                ("MobileScore", "mobile_score", True),
                ("MobileAccuracy", "mobile_acc", True),
                ("MobileThroughput", "mobile_tput", True),
                ("MobileAvailability", "mobile_avail", True),
                ("Mobile Acc/sec", "mobile_acc_per_sec", True),
                ("Smallest RAM", "ram_kib", False),
            ],
        )
        page_bests(
            pdf, dfo, *n(),
            title="Best learn — tide BestLearn + gated Acc",
            subtitle="Acc≥70 slice refuses Sparse-clock wins with chance Acc.",
            specs=[
                ("Acc / sec", "acc_per_sec", True),
                ("Time to Acc 25 (faster)", "t25", False),
                ("Time to Acc 50 (faster)", "t50", False),
            ],
        )
        if len(learned):
            page_bests(
                pdf, learned, *n(),
                title="Best among Acc ≥ 70 — learning-gated Score",
                subtitle="Score winner here still updated the net. If this is Sparse, Acc held.",
                specs=[
                    ("Score | Acc≥70", "score", True),
                    ("Avail | Acc≥70", "avail", True),
                    ("MobileScore | Acc≥70", "mobile_score", True),
                    ("SoftAcc | Acc≥70", "soft", True),
                ],
            )
        if len(sine):
            page_bests(
                pdf, sine, *n(),
                title="Sine only — adaptation domain",
                subtitle="AdaptPct is the mid-stream metric. Hard Acc can be 100% while Soft is 40%.",
                specs=[
                    ("AdaptPct", "adapt", True),
                    ("Hard Acc", "acc", True),
                    ("SoftAcc", "soft", True),
                    ("Score", "score", True),
                    ("Avail", "avail", True),
                ],
            )
        page_leaderboard(pdf, dfo, "score", "Leaderboard — Lucy Score (top 20)", *n(),
                         note="Score = duty clock. Read Acc. Sparse will dominate unless Acc collapsed.")
        page_leaderboard(pdf, dfo, "acc", "Leaderboard — hard Acc (top 20)", *n(),
                         note="Learning ranking. Ties at 100% are common on 2s Dense sine.")
        page_leaderboard(pdf, dfo, "soft", "Leaderboard — SoftAcc (top 20)", *n(),
                         note="Live fit during the window. Sine Soft can be 40% while hard Acc is 100%.")
        page_leaderboard(pdf, dfo, "adapt", "Leaderboard — AdaptPct (top 20)", *n(),
                         note="Sine freq switches only. xor/copy AdaptPct is 0.")
        page_leaderboard(pdf, dfo, "avail", "Leaderboard — Availability (top 20)", *n(),
                         note="Infer / (Infer+Train). Sparse and mamba will sit here.")
        page_leaderboard(pdf, dfo, "tput", "Leaderboard — Throughput (top 20)", *n())
        page_leaderboard(pdf, dfo, "zero_dt", "Leaderboard — ZeroDowntime (top 20)", *n(),
                         note="SoftAcc × Avail / 100")
        page_leaderboard(pdf, dfo, "mobile_score", "Leaderboard — MobileScore (top 20)", *n(),
                         note="Score / WeightMiB. Low-bit dtypes can win this without winning Acc.")
        page_task_topn(pdf, dfo, *n(), n=10)
        page_task_mode_top(pdf, dfo, *n())
        page_task_mode_top_score(pdf, dfo, *n())
        page_winners_cross(pdf, dfo, *n())
        dtype_order = list(dict.fromkeys(dfo["dtype"].tolist()))
        page_topn_by_group(pdf, dfo, "dtype", "acc", "Top 3 Acc per dtype (all modes × layers × tasks)",
                           n, top=3, group_order=dtype_order,
                           note="Every numerical type. #1/#2/#3 can be different modes.")
        page_topn_by_group(pdf, dfo, "dtype", "score", "Top 3 Score per dtype (all modes × layers × tasks)",
                           n, top=3, group_order=dtype_order,
                           note="Duty clock per storage type. Usually Sparse.")
        page_topn_by_group(pdf, dfo, "short", "acc", "Top 3 Acc per training mode (all dtypes × layers × tasks)",
                           n, top=3, group_order=MODE_ORDER,
                           note="Every train mode. Winner cell includes dtype.")
        page_topn_by_group(pdf, dfo, "short", "score", "Top 3 Score per training mode (all dtypes × layers × tasks)",
                           n, top=3, group_order=MODE_ORDER)
        page_topn_by_group(pdf, dfo, "layer", "acc", "Top 3 Acc per layer kind",
                           n, top=3, note="Best dtype/mode for each mid Op.")
        page_topn_by_group(pdf, dfo, "layer", "score", "Top 3 Score per layer kind",
                           n, top=3)
        page_best_mode_per_dtype(pdf, dfo, n)
        page_best_acc_mode_per_dtype(pdf, dfo, n)
        page_mode_bars(pdf, dfo, *n())
        page_heat_dtype_mode(pdf, dfo, *n())
        page_heat_layer_mode(pdf, dfo, *n())
        page_scatter_headline(pdf, dfo, *n())
        page_pareto(pdf, dfo, *n())
        page_vs_bp(pdf, vs, *n())
        page_vs_bp_bars(pdf, vs, *n())
        page_dtype_rank(pdf, dfo, *n())
        page_arch_layer(pdf, dfo, *n())
        page_honesty_matrix(
            pdf, dfo, "layer",
            "Honesty — mean Acc of every mode on every layer (all tasks)",
            n, note="All dtypes pooled. " + ABBR_NOTE,
        )
        page_honesty_matrix(
            pdf, dfo, "dtype",
            "Honesty — mean Acc of every mode on every dtype (all tasks)",
            n, note="All layers pooled. " + ABBR_NOTE,
        )
        if len(sine):
            page_honesty_matrix(
                pdf, sine, "layer",
                "Honesty — sine Acc, every mode × every layer",
                n, note="Learning board. XOR lottery stripped. " + ABBR_NOTE,
            )
            page_honesty_matrix(
                pdf, sine, "dtype",
                "Honesty — sine Acc, every mode × every dtype",
                n, note="Learning board. Same weight storage, every train mode. " + ABBR_NOTE,
            )
            page_honesty_matrix(
                pdf, sine, "layer",
                "Honesty — sine AccΔ vs StepBP, every mode × every layer",
                n, delta=True,
                note="Green-ish positive = beats backprop Acc on that mid Op. " + ABBR_NOTE,
            )
            page_honesty_matrix(
                pdf, sine, "dtype",
                "Honesty — sine AccΔ vs StepBP, every mode × every dtype",
                n, delta=True,
                note="Positive = beats StepBP Acc on that storage type. " + ABBR_NOTE,
            )
        page_honesty_delta_heat(pdf, vs, n)
        page_honesty(pdf, dfo, vs, *n())

        d = pdf.infodict()
        d["Title"] = "Welvet test48 Lucy report"
        d["Author"] = "welvet test48"
        d["Subject"] = "Credit modes × layers × dtypes × xor/sine/copy"

    write_winners_json(dfo, vs, win_path)
    print(f"PDF  {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB, {page} pages)")
    print(f"winners  {win_path}")


if __name__ == "__main__":
    main()
