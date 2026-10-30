"""The main table as a frontier: accuracy against what one episode costs.

The table's two columns are the two axes, so plotting it says the thing the table only implies --
nothing published sits on the staircase. Cost is input plus output tokens per episode on a log
axis, because the arms span four orders of magnitude, from 215 to 286,701.

Everything is the held-out test split, and the numbers are read from
notes/arbench_mh/published_baselines_test.json so the figure cannot drift from Table 1. Only the
five methods AR-Bench provides on all three tasks are drawn; with five published arms per panel
there is no room for inline labels, so the methods are told apart by marker and named once in the
legend, and only our systems and the anchor are labelled in place.

    SLIDES=1 python meta/tools/plot_frontier.py figures/
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
    ACC, INK, MUTE, PUB = "#990000", "#1D1D1D", "#777777", "#5A5A5A"
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    ACC, INK, MUTE, PUB = "#b4530f", "#1c1f26", "#8b93a1", "#5d6470"
    FAM, FAMKEY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif"
plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM,
    "font.size": 11 if SLIDES else 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3c424c", "xtick.color": "#3c424c", "ytick.color": "#3c424c",
    "figure.dpi": 220, "savefig.dpi": 220,
})

# The five methods AR-Bench provides on all three tasks, in the table's order. Read from the
# scored test-split file rather than restated, so Table 1 and this figure cannot disagree.
PUBS = [("zero_shot", "Zero-shot", "o"), ("few_shot", "Few-shot", "^"),
        ("few_shot_inst", "Few-shot w/ inst.", "v"), ("proactive_cot", "Proactive CoT", "P"),
        ("tot", "Tree of Thoughts", "X")]
TOT = {"gn": "tot_fixed", "dc": "tot_eval", "sp": "tot_eval"}   # AR-Bench ships ToT per task
TASKS = [("gn", "Guess Number"), ("dc", "Detective Cases"), ("sp", "Situation Puzzles")]
OURS = {   # finalisation 2026-10-27, logs/finalize_run1b.out
    "gn": [("ask nothing", 0.0, 226, "anchor"), ("ours", 100.0, 7, "ours")],
    "dc": [("ask nothing", 36.0, 847, "anchor"),
           ("ours, cheapest", 57.0, 18308, "ours"), ("ours, best", 63.0, 25715, "ours")],
    "sp": [("ask nothing", 10.5, 215, "anchor"),
           ("ours, cheapest", 31.5, 10913, "ours"), ("ours, best", 32.5, 16142, "ours")],
}

ROWS = json.load(open(Path(__file__).resolve().parents[2]
                      / "notes/arbench_mh/published_baselines_test.json"))["rows"]
DATA = {}
for tid, title in TASKS:
    pts = []
    for key, label, mk in PUBS:
        r = ROWS["%s/%s" % (tid, TOT[tid] if key == "tot" else key)]
        pts.append((label, r["accuracy"], r["tokens"], "pub", mk))
    pts += [(n, a, t, k, None) for n, a, t, k in OURS[tid]]
    DATA[title] = pts

MARK = {"anchor": ("s", MUTE, 46), "ours": ("D", ACC, 56)}
# Only our systems and the anchor are labelled in place, so the collisions worth resolving by hand
# are few. The published arms are named once, in the legend.
OFFSET = {
    ("Guess Number", "ask nothing"): (0, 8, "center"),
    ("Guess Number", "ours"): (10, -3, "left"),
    ("Detective Cases", "ask nothing"): (0, 8, "center"),
    ("Detective Cases", "ours, cheapest"): (-8, -14, "right"),
    ("Detective Cases", "ours, best"): (0, 9, "center"),
    ("Situation Puzzles", "ask nothing"): (0, 8, "center"),
    ("Situation Puzzles", "ours, cheapest"): (-7, -15, "right"),
    ("Situation Puzzles", "ours, best"): (0, 9, "center"),
}
out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
for ax, (task, rows) in zip(axes, DATA.items()):
    # the staircase: nothing above and to the left of it
    front, best = [], -1
    for name, acc, tok, kind, mk in sorted(rows, key=lambda r: r[2]):
        if acc > best:
            front.append((tok, acc)); best = acc
    sx, sy = [front[0][0]], [front[0][1]]
    for x, y in front[1:]:
        sx += [x, x]; sy += [sy[-1], y]
    sx.append(max(r[2] for r in rows) * 2.2); sy.append(sy[-1])
    ax.plot(sx, sy, color=ACC, lw=1.2, ls=(0, (3, 2)), zorder=1, alpha=0.65)

    for name, acc, tok, kind, mk in rows:
        if kind == "pub":
            ax.scatter(tok, acc, marker=mk, s=46, color=PUB, edgecolors=INK,
                       linewidths=0.5, zorder=4)
            continue
        m, c, sz = MARK[kind]
        ax.scatter(tok, acc, marker=m, s=sz, color=c, edgecolors=INK, linewidths=0.6, zorder=5)
        dx, dy, ha = OFFSET.get((task, name), (0, 8, "center"))
        ax.annotate(name, (tok, acc), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=9.5 if SLIDES else 7.6,
                    color=ACC if kind == "ours" else INK)
    ax.set_xscale("log")
    top = max(r[1] for r in rows)
    ax.set_ylim(-top * 0.13, top * 1.30)
    ax.set_xlim(4, 10 ** 6)
    ax.set_title(task, fontsize=12.5 if SLIDES else 10.4, color=INK, pad=8)
    ax.set_xlabel("tokens per episode  (input + output)", fontsize=11 if SLIDES else 8.8)
    ax.grid(color="#ECECEC", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("accuracy %", fontsize=11.5 if SLIDES else 9.2)

h = [plt.Line2D([], [], ls="", marker=mk, color=PUB, mec=INK, mew=0.5, ms=6.5)
     for _, _, mk in PUBS]
h += [plt.Line2D([], [], ls="", marker=MARK[k][0], color=MARK[k][1], mec=INK, mew=0.6, ms=7)
      for k in ("anchor", "ours")]
fig.legend(h, [lbl for _, lbl, _ in PUBS] + ["ask nothing", "evolved harness"],
           frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.03),
           fontsize=10 if SLIDES else 8.2, columnspacing=1.4, handletextpad=0.4)
fig.tight_layout(pad=0.8, w_pad=2.2, rect=(0, 0.06, 1, 1))
name = "fig_frontier%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
