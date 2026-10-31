"""AR-Bench's own methods, by interaction round, on our frozen Qwen2.5-7B-Instruct.

The shape follows AR-Bench's Tables 8-9 figure: one panel a task, one line a method, the name
set at the right-hand end of its own line rather than in a legend box, so six lines stay
readable. We have one model, so one row instead of two.

    SLIDES=1 python meta/tools/plot_baseline_rounds.py curves.json figures/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLIDES = os.environ.get("SLIDES") == "1"
if SLIDES:
    INK, MUTE = "#1D1D1D", "#555555"
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    INK, MUTE = "#1c1f26", "#6b7382"
    FAM, FAMKEY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif"

plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM,
    "font.size": 11 if SLIDES else 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3c424c", "xtick.color": "#3c424c", "ytick.color": "#3c424c",
    "figure.dpi": 220, "savefig.dpi": 220,
})

# AR-Bench's own palette order, with our harness held out in the accent
STYLE = {
    "zero_shot":     ("Zero-shot",      "#1D1D1D", "-",  1.6),
    "few_shot":      ("Few-shot",       "#2E5E86", "-",  1.6),
    "few_shot_inst": ("Few-shot inst.", "#8FB4D4", "-",  1.6),
    "proactive_cot": ("Proactive CoT",  "#2E8B57", "-",  1.6),
    "tot":           ("ToT",            "#B03030", (0, (4, 2)), 1.6),
    "tot_fixed":     ("ToT",            "#B03030", (0, (4, 2)), 1.6),
    "ours":          ("ours",           "#990000", "-",  2.6),
}
TASKS = [("gn", "Guess Number", "posterior mass on the answer"),
         ("dc", "Detective Cases", "suspects interrogated, of five"),
         ("sp", "Situation Puzzles", "decisive answers, share of budget")]

data = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
for ax, (task, title, ylab) in zip(axes, TASKS):
    rows = data.get(task, {})
    ends = []
    for key, rec in rows.items():
        if key not in STYLE:
            continue
        name, col, ls, lw = STYLE[key]
        c = rec["curve"]
        xs = list(range(1, len(c) + 1))
        ax.plot(xs, c, color=col, ls=ls, lw=lw, zorder=3)
        ax.plot([5, 10, 15, 20, 25], [c[i - 1] for i in (5, 10, 15, 20, 25)],
                "o", color=col, ms=3.4, zorder=4)
        ends.append((c[-1], name, col))
    # names at the end of their own line, nudged apart where two lines finish together
    ends.sort(reverse=True)
    last = None
    for y, name, col in ends:
        y = min(y, last - 4.5) if last is not None and y > last - 4.5 else y
        ax.text(25.6, y, name, color=col, fontsize=10 if SLIDES else 8.2,
                va="center", ha="left")
        last = y
    ax.set_title(title, fontsize=12 if SLIDES else 10.2, color=INK, pad=8)
    ax.set_xlim(1, 25); ax.set_ylim(0, 104)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xlabel("interaction round", fontsize=11 if SLIDES else 9)
    ax.set_ylabel(ylab, fontsize=10 if SLIDES else 8.4)
    ax.grid(axis="y", color="#ECECEC", lw=0.6); ax.set_axisbelow(True)

fig.tight_layout(pad=0.8, w_pad=3.6)
name = "fig_baseline_rounds%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
