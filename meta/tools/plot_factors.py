"""Which Bayesian factor survived, read along one evolution run's accepted chain.

Section 7.1's skeleton decomposes the Bayesian agent into six factors, of which perception is not
used on a text task. Those five are the rows. The columns are the steps of the run, each column
being the harness that step accepted, and reading a row left to right answers what the attribution
needs: did the search keep this factor, lose it, or never have it. A filled dot is present, a
hollow ring absent, and consecutive present cells are joined -- so a factor that drops out and
returns shows a visible break rather than being averaged into a percentage.

Two run shapes are read, because the two lines differ in how a step is decided:

  * meta/arbench_mh (Guess Number): a Pareto frontier over accuracy and cost, re-evaluated each
    iteration. On this task accuracy is 100% for every measured candidate, so the frontier is a
    single point and the column is unambiguous; the figure's footer is cost, which is what
    selection actually moved on.
  * meta/cd_tthe (Circuit Decoding, Situation Puzzles): three batches, six candidates each, with
    one committed per batch in opt_log.jsonl. The footer is the committed harness's own score on
    its batch.

A row is drawn as a hatched band, not as empty cells, where "absent" would be the wrong word
because the task leaves that factor nothing to do. Guess Number and Circuit Decoding both fix
their state space -- every permutation of four digits, and the public circuit space the inventory
defines -- so hypothesis generation has nothing to propose on either, and the census criterion
says so itself. On Guess Number env.guess() also sets solved on a hit, so every guess is an answer
and there is no decision to stop asking; Circuit Decoding and Situation Puzzles both end with one
final answer after a budget of queries, so there stopping is a real decision. Situation Puzzles
leaves all five live, which is why its empty cells mean what they say.

Colour: the accent is the blue the other figures in this paper already use, and the absent ring is
ink, not a second series. validate_palette.js passes that pair on lightness, CVD separation
(dE 17.2 protan), normal vision (19.3) and contrast, and fails only its chroma floor -- which is
the intended reading, since the ring is meant to recede. State is carried by fill and texture
before colour, so the figure survives greyscale and print.

    SLIDES=1 python meta/tools/plot_factors.py <task> figures/
    task in: gn, sp, cd_strict, cd_labeled, all
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

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "notes/arbench_mh"
MH = ROOT / "meta/arbench_mh/runs"
TT = ROOT / "meta/cd_tthe/logs"

# Per task: the census file, how its steps are read, the footer under each column, and the rows the
# task leaves nothing to do. The void set is an argument from the task's own rules, not a judgement
# about the harnesses -- see the module docstring.
TASKS = {
    "gn": dict(title="Guess Number", kind="frontier",
               census=CENSUS / "factor_census_gn_neutral1.json",
               run=MH / "gn_fixed/logs/neutral1", step="iteration",
               foot="tokens/episode", fmt="%.1f",
               void={"hypothesis_generation": "state space is given by the rules",
                     "stopping": "a correct guess ends the episode"}),
    "cd_strict": dict(title="Circuit Decoding, label-free", kind="tthe",
                      census=CENSUS / "factor_census_cd_tthe.json",
                      run=TT / "strict6_b10g3r2", step="batch",
                      foot="circuit-wise accuracy", key="circuit_wise_accuracy", fmt="%.3f",
                      void={"hypothesis_generation": "circuit space is public"}),
    "cd_labeled": dict(title="Circuit Decoding, labelled", kind="tthe",
                       census=CENSUS / "factor_census_cd_tthe.json",
                       run=TT / "labeled6_b10g3r2", step="batch",
                       foot="circuit-wise accuracy", key="circuit_wise_accuracy", fmt="%.3f",
                       void={"hypothesis_generation": "circuit space is public"}),
    "sp": dict(title="Situation Puzzles, label-free", kind="tthe",
               census=CENSUS / "factor_census_sp_tthe.json",
               run=TT / "sp_strict1_b10g3r2", step="batch",
               foot="character-level F1", key="f1_char", fmt="%.3f",
               void={}),
}

ORDER = [("belief_update", "belief update"),
         ("question_selection", "question selection"),
         ("observation_model", "observation model"),
         ("hypothesis_generation", "hypothesis generation"),
         ("stopping", "stopping")]


def columns(spec):
    """(step label, harness name, footer value) per accepted step, in order."""
    if spec["kind"] == "frontier":
        rows = [r for r in load(spec["run"], "gn") if r["measured"]]
        by_it = {}
        for r in rows:
            by_it.setdefault(r["iteration"], []).append(r)
        return [(str(it), w["system"], w["tokens"])
                for it in sorted(by_it)
                for w in [min(by_it[it], key=lambda r: r["tokens"])]]
    opt = [json.loads(l) for l in (spec["run"] / "opt_log.jsonl").read_text().splitlines() if l.strip()]
    return [(str(o["batch"]), o["committed"], (o.get("transductive") or {}).get(spec["key"]))
            for o in opt]


def verdict(present):
    """How the row reads left to right, in the words the attribution uses."""
    if not any(present):
        return "never appeared"
    if all(present):
        return "kept throughout"
    first, last = present.index(True), len(present) - 1 - present[::-1].index(True)
    if not all(present[first:last + 1]):
        return "lost, then regained"
    if last < len(present) - 1:
        return "appeared, then lost"
    return "arrived late, kept" if first else "kept from the start"


def panel(ax, spec, census, cols, show_foot=True):
    live = [(k, lab) for k, lab in ORDER if k not in spec["void"]]
    void = [(k, lab) for k, lab in ORDER if k in spec["void"]]
    n, nr = len(cols), len(ORDER)
    ax.set_xlim(0.45, n + 0.55)
    ax.set_ylim(-0.80, nr + (0.78 if show_foot else 0.10))
    ax.invert_yaxis()
    ax.axis("off")
    for x in range(1, n + 1):                              # recessive column guides
        ax.plot([x, x], [-0.42, nr - 0.55], color=FAINT, lw=0.8, zorder=0)

    for y, (key, label) in enumerate(live):
        present = [bool(census.get(h, {}).get(key)) for _s, h, _v in cols]
        for i in range(n - 1):                             # join consecutive present cells
            if present[i] and present[i + 1]:
                ax.plot([i + 1, i + 2], [y, y], color=ACCENT, lw=2.0, zorder=1,
                        solid_capstyle="round")
        xs = [i + 1 for i in range(n)]
        ax.scatter([x for x, p in zip(xs, present) if p], [y] * sum(present),
                   s=(BASE * 1.45) ** 2, c=ACCENT, edgecolors=ACCENT, lw=1.6, zorder=2)
        ax.scatter([x for x, p in zip(xs, present) if not p], [y] * (n - sum(present)),
                   s=(BASE * 1.15) ** 2, facecolors="white", edgecolors=MUTE, lw=1.6, zorder=2)
        ax.text(0.30, y, label, ha="right", va="center", fontsize=BASE + 1.3, color=INK)
        ax.text(n + 0.72, y, verdict(present), ha="left", va="center", fontsize=BASE - 0.6,
                color=ACCENT if any(present) else MUTE)

    if void:                                               # rule between the two kinds of row
        y0 = len(live) - 0.5
        ax.plot([0.45, n + 0.55], [y0, y0], color=MUTE, lw=0.7, ls=(0, (4, 3)), zorder=0)
    for j, (key, label) in enumerate(void):
        y = len(live) + j
        ax.add_patch(Rectangle((0.80, y - 0.15), n + 0.40 - 0.80, 0.30, facecolor="white",
                               edgecolor=FAINT, lw=0.6, hatch="///", zorder=1))
        ax.text(0.30, y, label, ha="right", va="center", fontsize=BASE + 1.3, color=MUTE)
        ax.text(n + 0.72, y, "nothing to do here \u2014 " + spec["void"][key], ha="left",
                va="center", fontsize=BASE - 0.6, color=MUTE)

    for i, (step, harness, val) in enumerate(cols):        # the column IS a harness, and its score
        ax.text(i + 1, -0.60, step, ha="center", va="center", fontsize=BASE + 0.8, color=INK)
        short = harness.replace("cand_", "").replace(spec["run"].name + "_", "")
        # a step that commits the previous step's harness found no improvement on its batch
        if i and harness == cols[i - 1][1]:
            short += "\n(unchanged)"
        ax.text(i + 1, nr - 0.42, short.replace("_", "\n"), ha="center", va="top",
                fontsize=BASE - 2.0, color=INK, linespacing=1.3)
        if show_foot and val is not None:
            ax.text(i + 1, nr + 0.58, spec["fmt"] % val, ha="center", va="center",
                    fontsize=BASE - 0.4, color=MUTE)
    ax.text(0.30, -0.60, spec["step"], ha="right", va="center", fontsize=BASE, color=MUTE)
    if show_foot:
        ax.text(0.30, nr + 0.58, spec["foot"], ha="right", va="center", fontsize=BASE - 0.4,
                color=MUTE)


def draw(name):
    spec = TASKS[name]
    census = {r["harness"]: r for r in json.loads(spec["census"].read_text())}
    cols = columns(spec)
    fig, ax = plt.subplots(figsize=((8.4 if not SLIDES else 9.8), 0.42 * len(ORDER) + 1.75))
    panel(ax, spec, census, cols)
    ax.set_title(spec["title"], fontsize=BASE + 2.5, color=INK, pad=14, loc="left", x=-0.055)
    tag = "_slides" if SLIDES else ""
    for ext in ("png", "pdf"):
        f = "fig_factors_%s%s.%s" % (name, tag, ext)
        fig.savefig(out / f, bbox_inches="tight", facecolor="white")
        print("  " + f)
    plt.close(fig)


which = sys.argv[1] if len(sys.argv) > 1 else "all"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
for nm in (list(TASKS) if which == "all" else [which]):
    draw(nm)
