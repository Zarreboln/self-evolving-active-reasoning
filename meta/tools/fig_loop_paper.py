"""The active-reasoning turn, drawn for the paper rather than for the deck.

The deck's version labels the three arrows in plain English because it has to be read from the
back of a room. The paper has already defined the POMDP a few lines above the figure, so here each
arrow carries the symbol it stands for and the picture and the Problem Statement say the same
thing twice --- once in notation, once in a shape.

    python meta/tools/fig_loop_paper.py figures/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MID = "#1D1D1D", "#555555"
IA_C, BT_C = "#2E5E86", "#990000"   # one hue per ability, the same two the IA/BT curves use
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",     # Times-like math, so the symbols match the body text
    "figure.dpi": 240, "savefig.dpi": 240,
})

# The canvas is sized so that, scaled to 0.82\textwidth, the labels come out at about the
# body text size. A wider canvas at the same font sizes would land them near 7pt.
fig, ax = plt.subplots(figsize=(6.4, 2.06))
ax.set_xlim(-0.35, 10.35)   # the boxes end at 0 and 10; without the margin tight bbox clips their borders
ax.set_ylim(-0.30, 3.05)
ax.axis("off")

for x, w, title, body, fill in (
        # The boxes widen outwards, not inwards: the 3.5--6.5 gap belongs to the arrows and
        # their labels, and narrowing it is what makes the formulas collide.
        (0.0, 3.5, "Agent", r"maintains a belief  $b_t$", "#FFFFFF"),
        (6.5, 3.5, "Environment", r"holds the hidden state  $s^{\star}$", "#F0F0F0")):
    ax.add_patch(FancyBboxPatch((x, 0.95), w, 1.45, boxstyle="round,pad=0.08",
                                fc=fill, ec=INK, lw=1.1))
    ax.text(x + w / 2, 2.03, title, ha="center", fontsize=19, color=INK)
    ax.text(x + w / 2, 1.33, body, ha="center", fontsize=14, color=MID)

# IA: the agent spends a turn on a question
ax.add_patch(FancyArrowPatch((3.5, 2.08), (6.5, 2.08), arrowstyle="-|>",
                             mutation_scale=17, lw=1.7, color=IA_C))
ax.text(5.0, 2.68, r"IA  $\cdot$  ask a question", ha="center", fontsize=14, color=IA_C)
ax.text(5.0, 2.26, r"$a_t \sim \pi^{\mathrm{ia}}(\cdot \mid b_t)$",
        ha="center", fontsize=14, color=IA_C)

# the environment answers and nothing more
ax.add_patch(FancyArrowPatch((6.5, 1.32), (3.5, 1.32), arrowstyle="-|>",
                             mutation_scale=17, lw=1.5, color=MID))
ax.text(5.0, 1.04, "answer", ha="center", fontsize=14, color=MID)
ax.text(5.0, 0.62, r"$o_t \sim O(\cdot \mid s^{\star}, a_t)$",
        ha="center", fontsize=14, color=MID)

# BT: whether that answer is allowed to move the belief
ax.add_patch(FancyArrowPatch((1.05, 0.92), (2.75, 0.92), arrowstyle="-|>",
                             mutation_scale=15, lw=1.5, color=BT_C,
                             connectionstyle="arc3,rad=0.55"))
ax.text(1.70, 0.22, r"BT  $\cdot$  update the belief", ha="center", fontsize=14, color=BT_C)
ax.text(1.70, -0.18, r"$b_{t+1} \sim \pi^{\mathrm{bt}}(\cdot \mid b_t, a_t, o_t)$",
        ha="center", fontsize=14, color=BT_C)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
fig.tight_layout(pad=0.2)
fig.savefig(out / "fig_loop_paper.png", bbox_inches="tight", facecolor="white")
print("  fig_loop_paper.png")
