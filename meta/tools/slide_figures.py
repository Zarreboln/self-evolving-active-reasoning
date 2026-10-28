"""The two figures the midterm deck needs beyond the evolution plots, in the deck's own look:
Georgia, greyscale, white ground. Shape and fill carry what colour would have.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MID, FAINT = "#1D1D1D", "#555555", "#AAAAAA"
ACCENT = "#990000"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.dpi": 220, "savefig.dpi": 220,
})

data = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])


def loop():
    """The cycle the task is: the agent asks, the environment answers, the belief updates."""
    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    for x, w, title, body, fill in ((0.1, 3.5, "Agent", "holds a belief over\nwhat the answer is", "#FFFFFF"),
                                    (6.4, 3.5, "Environment", "holds the hidden state,\nanswers and nothing more", "#F0F0F0")):
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 2.0, boxstyle="round,pad=0.08",
                                    fc=fill, ec=INK, lw=1.1))
        ax.text(x + w / 2, 2.25, title, ha="center", fontsize=15, color=INK)
        ax.text(x + w / 2, 1.45, body, ha="center", fontsize=11, color=MID, linespacing=1.5)
    ax.add_patch(FancyArrowPatch((3.8, 2.25), (6.2, 2.25), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.3, color=ACCENT))
    ax.text(5.0, 2.45, "IA  ·  chooses the question", ha="center", fontsize=11.5, color=ACCENT)
    ax.add_patch(FancyArrowPatch((6.2, 1.25), (3.8, 1.25), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.3, color=MID))
    ax.text(5.0, 0.95, "answer  ·  one turn spent", ha="center", fontsize=11.5, color=MID)
    ax.add_patch(FancyArrowPatch((1.0, 0.72), (2.9, 0.72), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.1, color=ACCENT,
                                 connectionstyle="arc3,rad=0.55"))
    ax.text(1.95, 0.08, "BT  ·  updates the belief", ha="center", fontsize=11.5, color=ACCENT)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "fig_loop_bw.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig_loop_bw.png")


def failures():
    """Left: the candidate set over a Guess Number episode -- belief tracking failing while the
    information is already in hand. Right: how long the vision model goes without learning
    anything -- information acquisition failing."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.5))

    # Most episodes pin the answer down early and then keep playing. That -- not a plateau at
    # some large number -- is what the flat part of these curves is, and it is the same fact as
    # "the feedback already determines the answer in 48.5% of episodes, and it still answers wrong".
    for t in data["gn_traces"]:
        a1.plot(range(1, len(t["sizes"]) + 1), t["sizes"], color=FAINT, lw=0.8, zorder=1)
    det = [t for t in data["gn_traces"] if 1 in t["sizes"] and len(t["sizes"]) >= 20]
    pick = min(det, key=lambda t: t["sizes"].index(1)) if det else data["gn_traces"][0]
    k = pick["sizes"].index(1) + 1
    a1.plot(range(1, len(pick["sizes"]) + 1), pick["sizes"], color=ACCENT, lw=2.2, zorder=3)
    a1.axvspan(k, len(pick["sizes"]), color="#F3E9E9", zorder=0)
    a1.set_yscale("log")
    a1.set_xlabel("turn", fontsize=11); a1.set_ylabel("candidates still possible", fontsize=11)
    a1.set_title("Guess Number: the answer is pinned down, then %d more turns are spent"
                 % (len(pick["sizes"]) - k), fontsize=12.5, color=INK, loc="left", pad=10)
    a1.tick_params(labelsize=10)
    a1.annotate("one candidate left\nby turn %d" % k, xy=(k, 1.0), xytext=(k - 2.8, 26),
                fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MID))
    a1.text((k + len(pick["sizes"])) / 2, 2.6,
            "nothing left to learn —\nand it still answers wrong\nin 48.5% of episodes",
            fontsize=10.5, color=MID, ha="center", linespacing=1.5)

    runs = data["amigo_runs"]
    bins = list(range(0, 21))
    a2.hist(runs, bins=bins, color="#D8C7C7", edgecolor=ACCENT, lw=0.7)
    a2.axvline(3, color=ACCENT, lw=1.4, ls=(0, (3, 2)))
    a2.text(0.98, 0.94, "3 or more consecutive dead turns: 80–93% of episodes\n"
            "10 or more: 21–33%", transform=a2.transAxes, ha="right", va="top",
            fontsize=10.5, color=INK, linespacing=1.6)
    a2.set_xlabel("longest run of consecutive turns that taught nothing", fontsize=11)
    a2.set_ylabel("episodes", fontsize=11)
    a2.set_title("AMIGO (vision): 500 episodes across five difficulty tiers",
                 fontsize=12.5, color=INK, loc="left", pad=10)
    a2.tick_params(labelsize=10)

    for ax in (a1, a2):
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    fig.savefig(out / "fig_failures_bw.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig_failures_bw.png")


loop(); failures()
