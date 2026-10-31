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
# One hue per ability, so a glance at the colour says which row you are in. Within a row, the
# before-evolution line is the same hue lightened and dashed, which keeps the pair legible in
# greyscale as well.
if SLIDES:
    INK, FAINT = "#1D1D1D", "#AAAAAA"
    HUE = {"ia": ("#9A9A9A", "#2E5E86"), "bt": ("#9A9A9A", "#990000")}
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    INK, FAINT = "#1c1f26", "#b8bec7"
    HUE = {"ia": ("#8b93a1", "#2f6286"), "bt": ("#8b93a1", "#b4530f")}
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

fig, axes = plt.subplots(2, len(TASKS), figsize=(4.1 * len(TASKS) + 1.0, 6.2), squeeze=False)


def draw(ax, Ts, before, after, ylab, show_legend, hue):
    pale, full = HUE[hue]
    for series, col, lw, ls, ms, mfc, lab in (
            (before, pale, 1.7, (0, (4, 2)), 3.6, "white", "before evolution"),
            (after, full, 2.4, "-", 4.0, full, "after evolution")):
        if series is None:
            continue
        xs = [0] + [t for t, v in zip(Ts, series) if v is not None]
        ys = [0] + [v for v in series if v is not None]
        ax.plot(xs, ys, color=col, lw=lw, ls=ls, marker="o", ms=ms, mec=col,
                mfc=mfc, label=lab, zorder=3)
    ax.set_xlim(0, Ts[-1])
    ax.set_ylim(-3, 107)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylab, fontsize=10.5 if SLIDES else 8.6)
    if show_legend:
        # The two rows have their empty corner in different places: IA climbs to a ceiling and
        # leaves the bottom right, BT stays low before evolution and leaves the top left.
        ax.legend(frameon=False, fontsize=9.6 if SLIDES else 7.8,
                  loc="lower right" if hue == "ia" else "center right",
                  handlelength=1.8, borderaxespad=0.4)


for col, (title, before, after, ialab) in enumerate(TASKS):
    b, a = data[before], data[after]
    Ts = b["turns"]
    top, bot = axes[0][col], axes[1][col]

    draw(top, Ts, b.get("ia"), a.get("ia"), ialab, col == 0, "ia")
    top.set_title(title, fontsize=12.5 if SLIDES else 10.4, color=INK, pad=9)
    if a.get("ia") is None and b.get("ia") is None:
        top.text(0.5, 0.5, "no proxy on this task", transform=top.transAxes, ha="center",
                 va="center", fontsize=10 if SLIDES else 8, color=FAINT)

    draw(bot, Ts, b["bt"], a["bt"], "belief in the right answer", col == 0, "bt")
    bot.set_xlabel("turn the episode was cut at", fontsize=10.5 if SLIDES else 8.8)

# The row names read horizontally in a left gutter, in that row's own colour, so they are read
# once rather than tilted into.
GUT = 1.0 / (4.1 * len(TASKS) + 1.0)
fig.tight_layout(pad=0.8, h_pad=1.4, w_pad=2.2, rect=(GUT, 0, 1, 1))
for r, (name, hue) in enumerate((("information\nacquisition", "ia"), ("belief\ntracking", "bt"))):
    bb = axes[r][0].get_position()
    fig.text(GUT * 0.52, (bb.y0 + bb.y1) / 2, name, ha="center", va="center",
             fontsize=15 if SLIDES else 12.5, color=HUE[hue][1], linespacing=1.35)
name = "fig_ia_bt%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
