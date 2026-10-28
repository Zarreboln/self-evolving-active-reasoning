"""Two views of the generalisation gap.

Left: every system we scored on both splits, as a line from its validation score to its test
score. The systems that were *picked* on validation fall; the anchor and the frontier member that
was never first do not. That contrast is the evidence, not the drop on its own.

Right: how much of the drop the act of picking can account for. Splitting the validation episodes
in half, choosing the best system on one half and reading it off the other, gives the penalty of
argmax alone -- and it grows with the number of candidates you choose among. On Detective Cases
it reaches about 3 points against an observed gap of 17, so argmax is not the explanation: what
is left is the proposer having read the development episodes for five rounds while writing the
code.

    SLIDES=1 python meta/tools/plot_overfitting.py figures/
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLIDES = os.environ.get("SLIDES") == "1"
if SLIDES:
    ACC, DOM, RULE, MUTE, INK = "#990000", "#8A8A8A", "#CFCFCF", "#555555", "#1D1D1D"
    FAM = ["Georgia", "Times New Roman", "DejaVu Serif"]; FAMKEY = "serif"
else:
    ACC, DOM, RULE, MUTE, INK = "#b4530f", "#8b93a1", "#c9ced6", "#7b8290", "#1c1f26"
    FAM = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]; FAMKEY = "sans-serif"

plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM,
    "font.size": 12 if SLIDES else 9.5,
    "axes.labelsize": 12 if SLIDES else 9.5,
    "xtick.labelsize": 11 if SLIDES else 9, "ytick.labelsize": 11 if SLIDES else 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3c424c", "xtick.color": "#3c424c", "ytick.color": "#3c424c",
    "figure.dpi": 220, "savefig.dpi": 220,
})

# (task, system, validation, test, was it the system selected as best on validation)
ROWS = [
    ("GN", "packed_group_echo", 100.0, 100.0, True),
    ("GN", "ask nothing",         0.0,   0.0, False),
    ("DC", "lean_verify",        80.0,  63.0, True),
    ("DC", "prior_verify_stop",  57.0,  57.0, False),
    ("DC", "ask nothing",        37.0,  36.0, False),
    ("SP", "atomic_audit",       41.0,  32.5, True),
    ("SP", "coarse_first",       39.0,  30.5, False),
    ("SP", "family_answer",      36.5,  31.5, False),
    ("SP", "ask nothing",        12.0,  10.5, False),
]
# half-split argmax penalty on Detective Cases, by how many candidates are chosen among
POOL = [(2, 1.9), (4, 1.5), (8, 2.3), (16, 2.9)]
OBSERVED_DC_GAP = 17.0

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.8),
                             gridspec_kw={"width_ratios": [1.45, 1]})

for task, name, v, t, picked in ROWS:
    c = ACC if picked else DOM
    a1.plot([0, 1], [v, t], color=c, lw=2.4 if picked else 1.4,
            marker="o", ms=7 if picked else 5,
            mfc=c, mec="white", mew=.8, zorder=3 if picked else 2)

# The right-hand labels are what the reader actually reads, and four of these systems land within
# five points of each other, so they are spread by hand: sorted by their value, then pushed apart
# to a minimum spacing, with a leader back to the endpoint.
lab = sorted(ROWS, key=lambda r: -r[3])
span = max(r[2] for r in ROWS) - min(r[3] for r in ROWS)
gap = span * 0.072
ys = []
for _, _, _, t, _ in lab:
    y = t if not ys else min(t, ys[-1] - gap)
    ys.append(y)
for (task, name, v, t, picked), y in zip(lab, ys):
    c = ACC if picked else DOM
    a1.annotate("%s  %s  %s%.1f" % (task, name, "−" if t < v else "+", abs(v - t)),
                (1, t), xytext=(1.06, y), textcoords="data",
                va="center", fontsize=11 if SLIDES else 8.6, color=c,
                fontweight="bold" if picked else "normal",
                arrowprops=dict(arrowstyle="-", lw=.6, color=RULE,
                                shrinkA=2, shrinkB=4) if abs(y - t) > gap * .35 else None)

a1.set_xlim(-0.1, 2.05); a1.set_xticks([0, 1])
a1.set_xticklabels(["validation", "test"])
a1.set_ylabel("accuracy %")
a1.set_title("Every system scored on both splits", loc="left", pad=30,
             fontsize=14 if SLIDES else 11, color=INK)
a1.grid(axis="y", color=RULE, lw=.6); a1.set_axisbelow(True)
a1.text(0.0, 1.015, "bold = the system that was chosen as best on validation",
        transform=a1.transAxes, fontsize=10.5 if SLIDES else 8.4, color=MUTE, va="bottom")

ks = [k for k, _ in POOL]; ps = [p for _, p in POOL]
a2.plot(ks, ps, color=ACC, lw=2.2, marker="o", ms=7, mfc=ACC, mec="white", mew=.8, zorder=3)
a2.axhline(OBSERVED_DC_GAP, color=INK, lw=1.4, ls=(0, (4, 2.5)), zorder=2)
a2.text(16, OBSERVED_DC_GAP - 0.9, "observed gap on Detective Cases, 17.0",
        ha="right", va="top", fontsize=11 if SLIDES else 8.6, color=INK)
a2.fill_between([min(ks), max(ks)], 0, ps[-1], color=ACC, alpha=.10, zorder=1)
a2.text(max(ks), ps[-1] + 0.7, "what choosing a winner can explain: ~3",
        ha="right", fontsize=11 if SLIDES else 8.6, color=ACC)
a2.set_xscale("log", base=2); a2.set_xticks(ks); a2.set_xticklabels([str(k) for k in ks])
a2.set_xlabel("candidates chosen among")
a2.set_ylabel("penalty of choosing, points")
a2.set_ylim(0, OBSERVED_DC_GAP * 1.18)
a2.set_title("Choosing a winner explains little of it", loc="left", pad=12,
             fontsize=14 if SLIDES else 11, color=INK)
a2.grid(axis="y", color=RULE, lw=.6); a2.set_axisbelow(True)

fig.tight_layout(pad=1.0)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
tag = "_bw" if SLIDES else ""
for ext in ("png", "pdf"):
    fig.savefig(out / f"overfitting{tag}.{ext}", bbox_inches="tight", facecolor="white")
print("  overfitting%s.png / .pdf" % tag)
