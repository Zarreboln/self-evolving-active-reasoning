"""Which Bayesian factor survived, read along the frontier of one evolution run.

Section 7.1's skeleton decomposes the Bayesian agent into six factors, of which perception is not
used on the three text tasks. Those five are the rows. The columns are the iterations, and each
column is that iteration's frontier harness -- on Guess Number accuracy is 100% for every measured
candidate, so the frontier is a single point and the column is unambiguous.

Reading a row left to right answers the question the attribution needs: did the search keep this
factor, lose it, or never have it. A filled dot is present, a hollow ring absent, and consecutive
present cells are joined -- so a factor that drops out and returns shows a visible break rather
than being averaged into a percentage.

Two rows are drawn as hatched bands instead of dots, because on this task "absent" would be the
wrong word. Guess Number fixes its own state space (every permutation of four digits), so
hypothesis generation has nothing to propose; and env.guess() sets solved on a hit, so every guess
is also an answer and there is no decision to stop asking. The census judge says Yes on 4 and 6 of
the 15 candidates respectively, against its own stated criteria; those cells are the judge reading
a given state space as a proposed one, and a forced-move shortcut as a stopping rule.

Colour: the accent is the blue the other figures in this paper already use, and the absent ring is
ink, not a second series. validate_palette.js passes that pair on lightness, CVD separation
(dE 17.2 protan), normal vision (19.3) and contrast, and fails only its chroma floor -- which is
the intended reading, since the ring is meant to recede. State is carried by fill and texture
before colour, so the figure survives greyscale and print.

    SLIDES=1 python meta/tools/plot_factors.py notes/arbench_mh/factor_census_gn_neutral1.json \
        meta/arbench_mh/runs/gn_fixed/logs/neutral1 figures/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from lineage import load

SLIDES = os.environ.get("SLIDES") == "1"
if SLIDES:
    INK, MUTE, ACCENT = "#1D1D1D", "#8b93a1", "#2E5E86"
    FAM, FAMKEY, BASE = ["Georgia", "Times New Roman", "DejaVu Serif"], "serif", 11.5
else:
    INK, MUTE, ACCENT = "#1c1f26", "#8b93a1", "#2f6286"
    FAM, FAMKEY, BASE = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif", 9.5
FAINT = "#dfe3e8"

plt.rcParams.update({
    "font.family": FAMKEY, ("font." + FAMKEY): FAM, "font.size": BASE,
    "figure.dpi": 220, "savefig.dpi": 220,
})

census = {r["harness"]: r for r in json.loads(Path(sys.argv[1]).read_text())}
rows = [r for r in load(Path(sys.argv[2]), "gn") if r["measured"]]
out = Path(sys.argv[3] if len(sys.argv) > 3 else "figures")

by_it = {}
for r in rows:
    by_it.setdefault(r["iteration"], []).append(r)
its = sorted(by_it)
chain = [min(by_it[it], key=lambda r: r["tokens"]) for it in its]

# Ordered as the story reads: kept throughout, lost and regained, arrived late, then the two the
# task leaves nothing to do.
LIVE = [("belief_update", "belief update", "kept throughout"),
        ("question_selection", "question selection", "lost for two rounds, regained"),
        ("observation_model", "observation model", "arrived late, kept")]
VOID = [("hypothesis_generation", "hypothesis generation", "state space is given by the rules"),
        ("stopping", "stopping", "a correct guess ends the episode")]

nr = len(LIVE) + len(VOID)
fig, ax = plt.subplots(figsize=(8.4 if not SLIDES else 9.8, 0.42 * nr + 1.75))
ax.set_xlim(0.45, len(its) + 0.55)
ax.set_ylim(-0.80, nr + 0.78)
ax.invert_yaxis()
ax.axis("off")

for x in range(1, len(its) + 1):                      # recessive column guides
    ax.plot([x, x], [-0.42, nr - 0.55], color=FAINT, lw=0.8, zorder=0)

for y, (key, label, verdict) in enumerate(LIVE):
    present = [bool(census.get(w["system"], {}).get(key)) for w in chain]
    for i in range(len(present) - 1):                  # join consecutive present cells
        if present[i] and present[i + 1]:
            ax.plot([i + 1, i + 2], [y, y], color=ACCENT, lw=2.0, zorder=1,
                    solid_capstyle="round")
    xs = [i + 1 for i in range(len(present))]
    ax.scatter([x for x, p in zip(xs, present) if p], [y] * sum(present),
               s=(BASE * 1.45) ** 2, c=ACCENT, edgecolors=ACCENT, lw=1.6, zorder=2)
    ax.scatter([x for x, p in zip(xs, present) if not p], [y] * (len(present) - sum(present)),
               s=(BASE * 1.15) ** 2, facecolors="white", edgecolors=MUTE, lw=1.6, zorder=2)
    ax.text(0.30, y, label, ha="right", va="center", fontsize=BASE + 1.3, color=INK)
    ax.text(len(its) + 0.72, y, verdict, ha="left", va="center",
            fontsize=BASE - 0.6, color=ACCENT if sum(present) else MUTE)

y0 = len(LIVE) - 0.5                                   # rule separating the two kinds of row
ax.plot([0.45, len(its) + 0.55], [y0, y0], color=MUTE, lw=0.7, ls=(0, (4, 3)), zorder=0)

for j, (key, label, why) in enumerate(VOID):
    y = len(LIVE) + j
    ax.add_patch(Rectangle((0.80, y - 0.15), len(its) + 0.40 - 0.80, 0.30, facecolor="white",
                           edgecolor=FAINT, lw=0.6, hatch="///", zorder=1))
    ax.text(0.30, y, label, ha="right", va="center", fontsize=BASE + 1.3, color=MUTE)
    ax.text(len(its) + 0.72, y, "nothing to do here \u2014 " + why, ha="left", va="center",
            fontsize=BASE - 0.6, color=MUTE)

for i, w in enumerate(chain):                          # the column IS a harness, and its cost
    ax.text(i + 1, -0.60, "%d" % its[i], ha="center", va="center",
            fontsize=BASE + 0.8, color=INK)
    ax.text(i + 1, nr - 0.42, w["system"].replace("_", "\n"), ha="center", va="top",
            fontsize=BASE - 2.0, color=INK, linespacing=1.3)
    ax.text(i + 1, nr + 0.58, "%.1f" % w["tokens"], ha="center", va="center",
            fontsize=BASE - 0.4, color=MUTE)
ax.text(0.30, -0.60, "iteration", ha="right", va="center", fontsize=BASE, color=MUTE)
ax.text(0.30, nr + 0.58, "tokens/episode", ha="right", va="center", fontsize=BASE - 0.4,
        color=MUTE)

tag = "_slides" if SLIDES else ""
for ext in ("png", "pdf"):
    fig.savefig(out / ("fig_factors%s.%s" % (tag, ext)), bbox_inches="tight", facecolor="white")
    print("  fig_factors%s.%s" % (tag, ext))
