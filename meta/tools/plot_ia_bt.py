"""The two abilities as process quantities: what the questions bought, and what the system took.

Prior art. AREW (ICML 2026) draws these two ways. Its Fig. 2 is the dynamics: per-turn AS, per-turn
BT and the outcome as curves against training steps, outcome on a twin axis. Its Fig. 4 is the
headline: one small panel per benchmark, two horizontal groups (AS on top, BT below), one bar per
condition, the value printed on each bar, and no numeric x axis at all because the two abilities
are not in the same units.

We keep Fig. 2's shape and change its x axis. There is no gradient here, so training steps do not
exist; the axis IA and BT are actually defined over is the TURN inside an episode. Read down a
column and the question is the same one AREW asks of a training curve: is the ability accumulating,
or is it flat?

Top row, per benchmark: the information the questions bought (the oracle's belief, filled grey)
against what the system actually believed (the line), before and after evolution. The gap between
the fill and the line is what belief tracking is losing, turn by turn. Only Guess Number has the
fill -- the oracle belief needs an enumerable consistent set, which Detective Cases (free text, and
the murderer lies) and Situation Puzzles (unbounded) do not have.

Bottom row: the same thing as increments, which is what the two quantities ARE -- AREW's I_AS
telescopes a sum of one-step oracle gains and its C_BT sums the agent's own positive one-step
gains. A bar pair per interval: what arrived, and what was taken.

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
    ACC, INK, MUTE, FILL, FAINT = "#990000", "#1D1D1D", "#555555", "#E4E4E4", "#AAAAAA"
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    ACC, INK, MUTE, FILL, FAINT = "#b4530f", "#1c1f26", "#6b7382", "#e6e3dc", "#b8bec7"
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

TASKS = [("Guess Number",      "gn_template_zero_shot", "gn_packed_group_echo"),
         ("Detective Cases",   "dc_template_zero_shot", "dc_lean_verify"),
         ("Situation Puzzles", "sp_baseline_zero_shot", "sp_atomic_audit")]
TASKS = [t for t in TASKS if t[1] in data and t[2] in data]

fig, axes = plt.subplots(2, len(TASKS), figsize=(4.3 * len(TASKS), 6.0),
                         gridspec_kw={"height_ratios": [1.25, 1]}, squeeze=False)

for col, (title, before, after) in enumerate(TASKS):
    b, a = data[before], data[after]
    Ts = b["turns"]
    top, bot = axes[0][col], axes[1][col]

    # --- what was available, and what was believed ------------------------------------------
    if a.get("oracle"):
        top.fill_between([0] + Ts, [0] + a["oracle"], color=FILL, zorder=0,
                         label="information the questions bought")
        # The two conditions do not ask the same questions, so they do not buy the same
        # information. Drawing only one fill would let the reader assume they did; on this task
        # the two are nearly the same curve, and that IS the finding.
        top.plot([0] + Ts, [0] + b["oracle"], color=FAINT, lw=1.2, ls=(0, (1, 1.6)), zorder=1,
                 label="    same, before evolution")
    top.plot([0] + Ts, [0] + b["system"], color=MUTE, lw=1.6, ls=(0, (4, 2)),
             marker="o", ms=3.4, mfc="white", zorder=3, label="before evolution")
    top.plot([0] + Ts, [0] + a["system"], color=ACC, lw=2.2,
             marker="o", ms=3.8, zorder=4, label="after evolution")
    top.set_title(title, fontsize=12 if SLIDES else 10.2, color=INK, pad=8)
    top.set_ylim(0, 108)
    top.set_xlim(0, Ts[-1])
    top.grid(axis="y", color="#ECECEC", lw=0.6)
    top.set_axisbelow(True)
    if col == 0:
        top.set_ylabel("belief in the right answer", fontsize=11 if SLIDES else 9)
    # Where the panel is empty differs: with the fill the curves climb into the top left, without
    # it they climb into the bottom right.
    top.legend(frameon=False, fontsize=9.6 if SLIDES else 7.8,
               loc="lower right" if a.get("oracle") else "upper left",
               handlelength=1.6, borderaxespad=0.4)

    # --- the same thing as per-turn gains ----------------------------------------------------
    def gains(series):
        return [series[0]] + [series[i] - series[i - 1] for i in range(1, len(series))]

    x = range(len(Ts))
    w = 0.38
    if a.get("oracle"):
        bot.bar([i - w / 2 for i in x], gains(a["oracle"]), w, color=FAINT, edgecolor=INK,
                lw=0.5, label="IA  ·  arrived", zorder=2)
        bot.bar([i + w / 2 for i in x], gains(a["system"]), w, color=ACC, edgecolor=INK,
                lw=0.5, label="BT  ·  taken (after)", zorder=2)
        bot.bar([i + w / 2 for i in x], gains(b["system"]), w, color="white", edgecolor=MUTE,
                lw=1.1, label="BT  ·  taken (before)", zorder=1)
    else:
        bot.bar([i - w / 2 for i in x], gains(b["system"]), w, color="white", edgecolor=MUTE,
                lw=1.1, label="BT gain, before", zorder=2)
        bot.bar([i + w / 2 for i in x], gains(a["system"]), w, color=ACC, edgecolor=INK,
                lw=0.5, label="BT gain, after", zorder=2)
        bot.text(0.015, 0.03, "no oracle belief on this task", transform=bot.transAxes,
                 ha="left", va="bottom", fontsize=9 if SLIDES else 7.6, color=MUTE)
    bot.axhline(0, color=INK, lw=0.8)
    bot.set_xticks(list(x))
    bot.set_xticklabels([str(t) for t in Ts], fontsize=10 if SLIDES else 8)
    bot.set_xlabel("turn the episode was cut at", fontsize=11 if SLIDES else 9)
    if col == 0:
        bot.set_ylabel("gain over the previous cut", fontsize=11 if SLIDES else 9)
    bot.grid(axis="y", color="#ECECEC", lw=0.6)
    bot.set_axisbelow(True)
    bot.legend(frameon=False, fontsize=9.6 if SLIDES else 7.6, loc="upper right",
               handlelength=1.6, borderaxespad=0.2)

fig.tight_layout(pad=0.7, h_pad=1.6, w_pad=1.6)
name = "fig_ia_bt%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
