"""The generalisation-gap figure, drawn the way the field draws it.

Prior art, in the three forms it actually takes:

  GEPA (ICLR 2026) Fig. 16, "Generalization Gap (Testset vs. Best Validation Score) on <bench>
    for <model>": one group per optimizer, two bars per group -- solid = final test score,
    hatched = best validation score, same hue -- with a connector between the two tops carrying
    the signed gap. Colour means method; hatch means split. One panel per (benchmark, model).

  AFlow (ICLR 2025) Fig. 5A: running-best curves over search iterations, solid = test,
    dashed = validation, colour = condition. Needs the test set scored at every round, which
    would cost us the one-open protocol, so we cannot draw it.

  ADAS (ICLR 2025) Fig. 3a: the running best on the held-out test set with a shaded 95% bootstrap
    interval and dashed lines for the hand-designed baselines. The interval is the part worth
    taking: a gap only means something next to the width of the interval around it.

This is GEPA's layout with ADAS's intervals. The interval is Wilson 95% for a Bernoulli with the
observed mean, which upper-bounds the interval of any score bounded in [0,1] -- exact for Guess
Number and Detective Cases (accuracy is 0/1), conservative for Situation Puzzles (F1 is
continuous, and its per-episode scores are not stored yet).

    SLIDES=1 python meta/tools/plot_genzgap.py figures/
"""
from __future__ import annotations
import math, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLIDES = os.environ.get("SLIDES") == "1"
if SLIDES:
    ACC, PICK, PLAIN, ANCH, INK, MUTE = "#990000", "#990000", "#6E6E6E", "#B9B9B9", "#1D1D1D", "#555555"
    FAM, FAMKEY = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif"
else:
    ACC, PICK, PLAIN, ANCH, INK, MUTE = "#b4530f", "#b4530f", "#6b7382", "#c3c8d0", "#1c1f26", "#7b8290"
    FAM, FAMKEY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif"

plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM,
    "font.size": 11 if SLIDES else 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3c424c", "xtick.color": "#3c424c", "ytick.color": "#3c424c",
    "figure.dpi": 220, "savefig.dpi": 220,
})

N = 100  # episodes in each of the validation and test splits

# task -> [(system, validation, test, picked-as-best-on-validation)]
DATA = {
    "Guess Number": [
        ("ask nothing",         0.0,   0.0, False),
        ("packed_group_echo", 100.0, 100.0, True),
    ],
    "Detective Cases": [
        ("ask nothing",       37.0, 36.0, False),
        ("prior_verify_stop", 57.0, 57.0, False),
        ("lean_verify",       80.0, 63.0, True),
    ],
    "Situation Puzzles": [
        ("ask nothing",   12.0, 10.5, False),
        ("family_answer", 36.5, 31.5, False),
        ("coarse_first",  39.0, 30.5, False),
        ("atomic_audit",  41.0, 32.5, True),
    ],
}


def wilson(p, n=N, z=1.96):
    """Half-width of the Wilson interval, in points, for a proportion p given as a percentage."""
    p = p / 100.0
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100.0 * half


def se(p, n=N):
    """Standard error in points. Bounded-in-[0,1] scores have at most Bernoulli variance."""
    p = p / 100.0
    return 100.0 * math.sqrt(p * (1 - p) / n)


def significant(val, test):
    """The two splits are disjoint episode sets, so the difference has se = sqrt(se_v^2+se_t^2).
    Comparing the two intervals by eye -- the mistake the bars invite -- is about 1.4x too strict."""
    return abs(test - val) > 1.96 * math.sqrt(se(val) ** 2 + se(test) ** 2)


out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4),
                         gridspec_kw={"width_ratios": [len(v) for v in DATA.values()]})

W, OFF = 0.33, 0.185
for ax, (task, rows) in zip(axes, DATA.items()):
    top = max(max(v, t) for _, v, t, _ in rows)
    labels = []
    for i, (name, val, test, picked) in enumerate(rows):
        col = ANCH if name == "ask nothing" else (PICK if picked else PLAIN)
        # Hatched = validation, solid = test, one hue per system. GEPA's encoding exactly.
        ax.bar(i - OFF, val, W, color=col, edgecolor=INK, lw=0.7, hatch="///", zorder=2)
        ax.bar(i + OFF, test, W, color=col, edgecolor=INK, lw=0.7, zorder=2)
        for x, y in ((i - OFF, val), (i + OFF, test)):
            ax.errorbar(x, y, yerr=wilson(y), fmt="none", ecolor=INK, elinewidth=0.9,
                        capsize=3, capthick=0.9, zorder=3)

        gap = test - val
        hi, lo = max(val, test), min(val, test)
        if abs(gap) >= 0.25:
            # The connector sits in the channel between the two bars, as in GEPA Fig. 16.
            ax.plot([i, i], [lo, hi], color=INK, lw=0.9, zorder=4)
            for y in (lo, hi):
                ax.plot([i - 0.05, i + 0.05], [y, y], color=INK, lw=0.9, zorder=4)
        # A gap no wider than its own interval is not a drop; say so rather than let the reader
        # guess from bar heights.
        sig = significant(val, test)
        labels.append((i, max(val + wilson(val), test + wilson(test)),
                       ("%+.1f" % gap) + ("" if sig else " n.s."), sig and gap < 0))

    # One label height for the whole panel, so the row reads across instead of stair-stepping
    # through the bars.
    ytxt = max(y for _, y, _, _ in labels) + top * 0.10
    for i, _, txt, bad in labels:
        ax.text(i, ytxt, txt, ha="center", va="bottom",
                fontsize=10 if SLIDES else 8.6, color=ACC if bad else MUTE,
                fontweight="bold" if bad else "normal")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=10 if SLIDES else 8.4)
    ax.set_ylim(0, ytxt + top * 0.17)
    ax.set_title("Generalization gap (test vs. best validation)\non %s" % task,
                 fontsize=11.5 if SLIDES else 9.6, color=INK, pad=10)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("score")

h = [plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, lw=0.7, hatch="///"),
     plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, lw=0.7)]
# Below the row of gap labels, which owns the top of every panel.
axes[0].legend(h, ["best validation score", "final test score"], frameon=False,
               loc="upper left", bbox_to_anchor=(0.0, 0.86),
               fontsize=10 if SLIDES else 8.4, handlelength=1.5)

fig.tight_layout(pad=0.8)
name = "fig_genzgap%s.png" % ("_slides" if SLIDES else "")
fig.savefig(out / name, bbox_inches="tight", facecolor="white")
print("  " + name)
