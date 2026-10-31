"""Accuracy against AR-Bench's own methods, on one frozen model and one split.

Everything here is the selection split, 100 instances, Qwen2.5-7B-Instruct, the same protocol and
metric for every row -- published_baselines.json for AR-Bench's arms and the run-1 snapshot for
ours -- so the bars are comparable. Cost is input plus output tokens per episode, printed beside
each bar because an accuracy that costs 287k tokens an episode is not the same result as one that
costs 863.

    SLIDES=1 python meta/tools/plot_accuracy.py figures/
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
    ACC, INK, MUTE, PUB, ANCH = "#990000", "#1D1D1D", "#555555", "#8A8A8A", "#C9C9C9"
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    ACC, INK, MUTE, PUB, ANCH = "#b4530f", "#1c1f26", "#6b7382", "#848b95", "#c8ccd2"
    FAM, FAMKEY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif"
plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM,
    "font.size": 11 if SLIDES else 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#3c424c", "xtick.color": "#3c424c", "ytick.color": "#3c424c",
    "figure.dpi": 220, "savefig.dpi": 220,
})

ROOT = Path(__file__).resolve().parents[2] / "notes" / "arbench_mh"
pub = json.loads((ROOT / "published_baselines.json").read_text())["rows"]
snap = json.loads((ROOT / "evolution_run1_snapshot.json").read_text())

TASKS = [("gn", "Guess Number", "exact match"),
         ("dc", "Detective Cases", "right suspect of five"),
         ("sp", "Situation Puzzles", "judged by jev-1.14.0")]
out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
for ax, (task, title, metric) in zip(axes, TASKS):
    rows = [(r["system"], r["accuracy"], r["tokens"], "pub") for r in pub if r["task_id"] == task]
    ours = [r for r in snap if r["task_id"] == task and r["frontier"] == "yes"]
    for r in ours:
        kind = "anchor" if r["kind"] == "seed" else "ours"
        rows.append((r["system"], r["accuracy"], r["tokens"], kind))
    rows.sort(key=lambda r: r[1])
    y = range(len(rows))
    col = {"pub": PUB, "ours": ACC, "anchor": ANCH}
    ax.barh(list(y), [r[1] for r in rows], 0.68,
            color=[col[r[3]] for r in rows], edgecolor=INK, lw=0.6, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0].replace("_", "​_") for r in rows],
                       fontsize=10 if SLIDES else 8)
    top = max(r[1] for r in rows)
    for i, r in enumerate(rows):
        ax.text(r[1] + top * 0.02, i, "%.1f" % r[1], va="center",
                fontsize=10 if SLIDES else 8.2,
                color=ACC if r[3] == "ours" else INK,
                fontweight="bold" if r[3] == "ours" else "normal")
        ax.text(top * 1.30, i, "%s tok" % format(int(round(r[2])), ","), va="center",
                fontsize=9 if SLIDES else 7.4, color=MUTE)
    ax.set_xlim(0, top * 1.62)
    ax.set_title("%s\n%s" % (title, metric), fontsize=12 if SLIDES else 10.2, color=INK, pad=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_xlabel("accuracy %", fontsize=11 if SLIDES else 9)

h = [plt.Rectangle((0, 0), 1, 1, fc=PUB, ec=INK, lw=0.6),
     plt.Rectangle((0, 0), 1, 1, fc=ANCH, ec=INK, lw=0.6),
     plt.Rectangle((0, 0), 1, 1, fc=ACC, ec=INK, lw=0.6)]
fig.legend(h, ["AR-Bench's published methods", "ask nothing", "evolved harness"],
           frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           fontsize=11 if SLIDES else 8.8, handlelength=1.4)
fig.tight_layout(pad=0.8, w_pad=2.4, rect=(0, 0.06, 1, 1))
name = "fig_accuracy%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
