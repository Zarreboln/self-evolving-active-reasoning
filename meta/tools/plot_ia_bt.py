"""The two abilities as process quantities: one panel per benchmark per ability, before and after.

Prior art. AREW (ICML 2026) draws these two ways. Its Fig. 2 is the dynamics -- per-turn AS,
per-turn BT and the outcome as curves against training steps. Its Fig. 4 is the headline -- one
small panel per benchmark, two horizontal bar groups, no numeric x axis at all, because the two
abilities are not in the same units.

This keeps Fig. 2's shape and changes its x axis. There is no gradient here, so training steps do
not exist; the axis the two quantities are defined over is the TURN inside an episode. Read a
column and the question is the one AREW asks of a training curve: is the ability accumulating, or
is it flat?

Top row, information acquisition: what the questions had bought by turn t. Bottom row, belief
tracking: what the system actually believed by turn t, read by ending the episode there.

Only belief tracking is the same measurement on all three tasks. Information acquisition is a
task-specific proxy -- posterior mass on gn, suspect coverage on dc, decisive answers on sp -- as
T3 (ICLR 2026) also does, instantiating one criterion per task from whatever the transcript makes
observable. The three are bounded 0-100 and monotone and that is all they share, so each panel
carries its own label and they are never averaged together.

    SLIDES=1 python meta/tools/plot_ia_bt.py curves.json figures/
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
    ACC, INK, MUTE, FAINT = "#990000", "#1D1D1D", "#555555", "#AAAAAA"
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    ACC, INK, MUTE, FAINT = "#b4530f", "#1c1f26", "#6b7382", "#b8bec7"
    FAM, FAMKEY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif"

plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM,
    "font.size": 11 if SLIDES else 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3c424c", "xtick.color": "#3c424c", "ytick.color": "#3c424c",
    "figure.dpi": 220, "savefig.dpi": 220,
})

data = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")

TASKS = [("Guess Number", "gn_template_zero_shot", "gn_packed_group_echo",
          "posterior mass on the answer"),
         ("Detective Cases", "dc_template_zero_shot", "dc_lean_verify",
          "suspects interrogated, of five"),
         ("Situation Puzzles", "sp_baseline_zero_shot", "sp_atomic_audit",
          "decisive answers, share of budget")]
TASKS = [t for t in TASKS if t[1] in data and t[2] in data]

fig, axes = plt.subplots(2, len(TASKS), figsize=(4.1 * len(TASKS), 6.2), squeeze=False)


def draw(ax, Ts, before, after, ylab, show_legend):
    for series, col, lw, ls, ms, lab in (
            (before, MUTE, 1.6, (0, (4, 2)), 3.4, "before evolution"),
            (after, ACC, 2.2, "-", 3.8, "after evolution")):
        if series is None:
            continue
        xs = [0] + [t for t, v in zip(Ts, series) if v is not None]
        ys = [0] + [v for v in series if v is not None]
        ax.plot(xs, ys, color=col, lw=lw, ls=ls, marker="o", ms=ms,
                mfc="white" if col is MUTE else col, label=lab, zorder=3)
    ax.set_xlim(0, Ts[-1])
    ax.set_ylim(-3, 107)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylab, fontsize=10 if SLIDES else 8.4)
    if show_legend:
        ax.legend(frameon=False, fontsize=9.6 if SLIDES else 7.8, loc="lower right",
                  handlelength=1.8, borderaxespad=0.4)


for col, (title, before, after, ialab) in enumerate(TASKS):
    b, a = data[before], data[after]
    Ts = b["turns"]
    top, bot = axes[0][col], axes[1][col]

    draw(top, Ts, b.get("ia"), a.get("ia"), ialab, col == 0)
    top.set_title(title, fontsize=12.5 if SLIDES else 10.4, color=INK, pad=9)
    if a.get("ia") is None and b.get("ia") is None:
        top.text(0.5, 0.5, "no proxy on this task", transform=top.transAxes, ha="center",
                 va="center", fontsize=10 if SLIDES else 8, color=FAINT)

    draw(bot, Ts, b["bt"], a["bt"], "belief in the right answer", False)
    bot.set_xlabel("turn the episode was cut at", fontsize=10.5 if SLIDES else 8.8)

axes[0][0].annotate("information\nacquisition", xy=(-0.42, 0.5), xycoords="axes fraction",
                    ha="center", va="center", rotation=90, fontsize=12 if SLIDES else 10,
                    color=INK)
axes[1][0].annotate("belief\ntracking", xy=(-0.42, 0.5), xycoords="axes fraction",
                    ha="center", va="center", rotation=90, fontsize=12 if SLIDES else 10,
                    color=INK)

fig.tight_layout(pad=0.8, h_pad=1.4, w_pad=2.2)
name = "fig_ia_bt%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
