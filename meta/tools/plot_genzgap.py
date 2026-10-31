"""The generalisation gap, drawn so the gap itself is the object on the page.

Prior art is GEPA (ICLR 2026) Fig. 16: one group per system, two bars -- hatched best-validation
against solid final-test -- and the signed difference annotated on a connector. Paired bars make
the reader subtract two heights by eye, and with four systems a panel the subtraction gets done
four times. So the two bars are stacked instead: the solid part is what survives on the sealed
split, the hatched part on top is what did not, and the whole column is still the validation
score. The gap is a block you can see, not a difference you compute.

Two systems a task, not four: the one the search picked, and the ask-nothing anchor that is the
control for it. The anchor is what makes the picked system's drop mean anything.

    SLIDES=1 python meta/tools/plot_genzgap.py figures/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLIDES = os.environ.get("SLIDES") == "1"
if SLIDES:
    ACC, INK, MUTE, KEPT, LOST = "#990000", "#1D1D1D", "#555555", "#595959", "#C9C9C9"
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    ACC, INK, MUTE, KEPT, LOST = "#b4530f", "#1c1f26", "#6b7382", "#5d6470", "#c8ccd2"
    FAM, FAMKEY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif"

plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM,
    "font.size": 12 if SLIDES else 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3c424c", "xtick.color": "#3c424c", "ytick.color": "#3c424c",
    "figure.dpi": 220, "savefig.dpi": 220,
})

# task -> [(system, validation, test)]; the picked system last so it sits on the right
DATA = {
    "Guess Number":      [("ask nothing", 0.0, 0.0), ("packed_group_echo", 100.0, 100.0)],
    "Detective Cases":   [("ask nothing", 37.0, 36.0), ("lean_verify", 80.0, 63.0)],
    "Situation Puzzles": [("ask nothing", 12.0, 10.5), ("atomic_audit", 41.0, 32.5)],
}


out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3))

W = 0.46
for ax, (task, rows) in zip(axes, DATA.items()):
    top = max(max(v, t) for _, v, t in rows)
    for i, (name, val, test) in enumerate(rows):
        kept, lost = min(val, test), abs(val - test)
        ax.bar(i, kept, W, color=KEPT, edgecolor=INK, lw=0.7, zorder=2)
        if lost >= 0.25:
            ax.bar(i, lost, W, bottom=kept, color=ACC, edgecolor=ACC, lw=0.8, zorder=2)
        ax.annotate("%+.1f" % (test - val), xy=(i, max(val, test)),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=12 if SLIDES else 9.2,
                    color=ACC if lost >= 0.25 else MUTE,
                    fontweight="bold" if lost >= 0.25 else "normal")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=11.5 if SLIDES else 9)
    ax.set_ylim(0, top * 1.30 + 6)
    ax.set_title(task, fontsize=13.5 if SLIDES else 10.6, color=INK, pad=10)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("score", fontsize=12 if SLIDES else 9.5)

h = [plt.Rectangle((0, 0), 1, 1, fc=KEPT, ec=INK, lw=0.7),
     plt.Rectangle((0, 0), 1, 1, fc=ACC, ec=ACC, lw=0.8)]
# The column height is the validation score and the grey part is the test score, so the two
# blocks are named for the two scores rather than for the difference between them.
fig.legend(h, ["test score", "validation score"],
           frameon=False, fontsize=12 if SLIDES else 9, ncol=2,
           loc="lower center", bbox_to_anchor=(0.5, -0.015), handlelength=1.5)

fig.tight_layout(pad=0.8, w_pad=2.0, rect=(0, 0.065, 1, 1))
name = "fig_genzgap%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
