"""Final-report figures (2026-11-08). Static PNG + SVG in report/figures/.
figA  answer-step memory: story score change vs. a plain resample of the same final answer (5 seeds, both judges)
figB  where the gain comes from: answer-step gain by how close the nearest TRAIN story is to the TEST story
figC  judge gap: change in question recall vs. the empty playbook under the lenient and the strict judge
figD  evolution rounds: validation recall under the judge that selected the playbook vs. strict test result
figE  asking vs. using: final-story score when a stronger model asks or writes (TEST seed 0)
Paired over the 100 official SP test puzzles, seeds averaged within puzzle, 95% bootstrap CI over puzzles."""
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402
import probe_ex  # noqa: E402
import sp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
US = ROOT / "runs" / "use_step"
EVO = ROOT / "runs" / "evolve"
PROBE = ROOT / "runs" / "probe_use"
FIG = mf.FIG
INK, INK2, MUTED, GRID, AXIS, SURF = mf.INK, mf.INK2, mf.MUTED, mf.GRID, mf.AXIS, mf.SURF
S1, S2, S3, S4 = mf.S1, mf.S2, mf.S3, mf.S4
# Match the team's midterm style (Georgia, white, dark red + grey + muted blue)
INK, INK2, MUTED, GRID, AXIS, SURF = "#1d1d1d", "#555555", "#8a8a8a", "#e6e6e6", "#bdbdbd", "#ffffff"
RED, GREY, BLUE, GOLD = "#990000", "#9a9a9a", "#2f5d8c", "#c08a2b"
S1, S2, S3, S4 = BLUE, RED, GREY, GOLD
plt.rcParams.update({"font.family": ["Georgia", "DejaVu Serif"], "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
JUDGES = (("haiku", "Lenient judge (Haiku)", GREY), ("gemini", "Strict judge (Gemini)", RED))


def boot(d, b=10_000):
    r = random.Random(0)
    n = len(d)
    ms = sorted(mean(d[r.randrange(n)] for _ in range(n)) for _ in range(b))
    return mean(d), ms[int(.025 * b)], ms[int(.975 * b)]


SLIDE = "--slide" in sys.argv          # slide versions: no figure title (the slide has one), file suffix _slide


def save(fig, name):
    if SLIDE:
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        name += "_slide"
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def load_use(cond, arm="a0"):
    out = {}
    for s in range(5):
        f = US / f"{arm}_s{s}_{cond}.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                x = json.loads(line)
                out[(x["id"], s)] = x
    return out


def use_diffs(cond, judge, base):
    a, per = load_use(cond), {}
    for k, x in a.items():
        if k in base and x[judge] is not None and base[k][judge] is not None:
            per.setdefault(k[0], []).append(x[judge] - base[k][judge])
    return {pid: mean(v) for pid, v in per.items()}


def dot_rows(ax, rows, xlabel, fmt="{:+.2f}"):
    """rows: [(label, {judge: (m, lo, hi)})] top to bottom."""
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    for i, (label, cells) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, (judge, jl, col) in enumerate(JUDGES):
            if judge not in cells:
                continue
            m, lo, hi = cells[judge]
            y = y0 + (0.16 if j == 0 else -0.16)
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([m], [y], s=40, color=col, edgecolor=SURF, linewidth=1.8, zorder=3,
                       label=jl if i == 0 else None)
            ax.text(hi + 0.01, y, fmt.format(m + 0.0).replace("-0.00", "+0.00"), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9)


def figA():
    base = load_use("plain7b")
    conds = [("ex7b", "2 similar past puzzles + true story"),
             ("tex7b", "2 similar puzzles, strong model asked + story"),
             ("ex7b_rand", "2 random past puzzles + true story"),
             ("ex7b_far", "2 past puzzles, near variants excluded"),
             ("facts7b", "List confirmed facts first, then write"),
             ("ex7b_nostory", "2 similar puzzles, Q&A only (no story)")]
    rows = [(label, {j: boot(list(use_diffs(c, j, base).values())) for j, _, _ in JUDGES}) for c, label in conds]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    dot_rows(ax, rows, "Change in story score (key points, 0-6) vs. resampling the same answer\n"
                       "(same frozen 7B, same transcript; 100 test puzzles x 5 seeds; 95% CI)")
    fig.suptitle("Showing worked examples just before the final answer helps under both judges;\n"
                 "the true story in the example is what matters", x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "figA_answer_step")
    return rows


# --- figB: closeness of the nearest TRAIN story --------------------------------------------------------------
def _tok(s):
    return [w for w in re.findall(r"[a-z']+", s.lower()) if len(w) > 2]


def story_closeness():
    """For each TEST puzzle: cosine (TF-IDF) between its true story and the true story of the TRAIN puzzle whose
    SURFACE is most similar. Diagnostic only: the method never sees TEST stories."""
    bank = probe_ex.BANK
    docs = [Counter(_tok(b["bottom"])) for b in bank]
    df = Counter(w for d in docs for w in d)
    idf = {w: math.log((1 + len(docs)) / (1 + c)) + 1 for w, c in df.items()}

    def vec(c):
        v = {w: tf * idf.get(w, math.log(1 + len(docs)) + 1) for w, tf in c.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1
        return {w: x / n for w, x in v.items()}
    bvec = [vec(d) for d in docs]
    out = {}
    for p in sp.load("test"):
        q = probe_ex._vec(Counter(probe_ex._toks(p["surface"])))
        top = sorted(range(len(bank)), key=lambda i: -sum(q.get(w, 0) * x for w, x in probe_ex._VECS[i].items()))[:2]
        t = vec(Counter(_tok(p["bottom"])))
        out[p["_id"]] = max(sum(t.get(w, 0) * x for w, x in bvec[i].items()) for i in top)
    return out


def figB():
    clos = story_closeness()
    base = load_use("plain7b")
    order = sorted(clos, key=clos.get)
    thirds = [set(order[i * len(order) // 3:(i + 1) * len(order) // 3]) for i in range(3)]
    conds = [("ex7b", "similar examples", RED), ("ex7b_rand", "random examples", BLUE), ("ex7b_far", "near variants excluded", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, (judge, jl, _) in zip(axes, JUDGES):
        for k, (c, label, col) in enumerate(conds):
            d = use_diffs(c, judge, base)
            for t, ids in enumerate(thirds):
                m, lo, hi = boot([v for pid, v in d.items() if pid in ids])
                x = t + (k - 1) * 0.22
                ax.plot([x, x], [lo, hi], color=col, lw=2, solid_capstyle="round")
                ax.scatter([x], [m], s=36, color=col, edgecolor=SURF, linewidth=1.5, zorder=3,
                           label=label if t == 0 else None)
        ax.axhline(0, color=AXIS, lw=1)
        means = [mean(clos[i] for i in ids) for ids in thirds]
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{n}\n(closeness {m:.2f})" for n, m in zip(("least similar", "middle", "most similar"), means)])
        ax.set_title(jl, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Change in story score vs. resample")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Similar examples help most on puzzles that have a close variant in TRAIN;\n"
                 "random examples give a smaller gain that does not depend on closeness",
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "figB_where_gain_comes_from")


# --- figC: judge gap on question recall ------------------------------------------------------------------------
GAP_ARMS = [("a2", "Strong-model playbook, no experience"), ("prior", "Evolved: hidden-story prior"),
            ("cce", "Evolved: step-level credit"), ("cce_shuf", "Evolved: step credit, shuffled"),
            ("traj", "Evolved: success vs. failure contrast"), ("traj_gem", "Evolved, selected by strict judge"),
            ("traj_cons", "Evolved, selected by both judges"), ("ace", "ACE (open-source)"),
            ("dc", "Dynamic Cheatsheet (open-source)"), ("bo8", "Best-of-8 written into rules"),
            ("tex", "Strong-model examples in prompt"), ("r1", "Best-of-8 at test time (~5x calls)")]


SLIDE_GAP = ("traj", "traj_gem", "traj_cons", "ace", "dc", "bo8", "r1")


def figC():
    rows = []
    for a, label in GAP_ARMS:
        if SLIDE and a not in SLIDE_GAP:
            continue
        cells = {}
        for j, _, _ in JUDGES:
            r = mf.paired(a, "a0", j)
            if r:
                cells[j] = r
        if len(cells) == 2:
            rows.append((label, cells))
    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(rows) + 1.6))
    dot_rows(ax, rows, "Change in question recall vs. empty playbook (share of hidden key points uncovered)\n"
                       "(100 test puzzles, 3-5 seeds, 95% CI)", fmt="{:+.3f}")
    fig.suptitle("Gains that show up under the lenient judge mostly disappear under the strict one;\n"
                 "ACE, which learns from judge feedback, gets worse", x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "figC_judge_gap")
    return rows


# --- figD: rounds -----------------------------------------------------------------------------------------------
def figD():
    mechs = [("traj", "selected by lenient judge", GREY), ("traj_cons", "selected by both judges", BLUE),
             ("traj_gem", "selected by strict judge", RED)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for m, label, col in mechs:
        curves = []
        for s in range(3):
            h = json.loads((EVO / f"{m}_s{s}" / "history.json").read_text(encoding="utf-8"))
            c = [h[0]["val_recall"]] + [r["val_recall_incumbent"] for r in h[1:]]
            curves.append([v - c[0] for v in c])
        y = [mean(c[i] for c in curves) for i in range(len(curves[0]))]
        ax.plot(range(len(y)), y, color=col, lw=2, marker="o", ms=5, markeredgecolor=SURF, markeredgewidth=1.5,
                label=label)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["start", "1", "2", "3", "4"])
    ax.set_xlabel("evolution round (answer -> evolve -> answer)")
    ax.set_ylabel("Validation recall gain since start\n(under the selecting judge, mean of 3 seeds)")
    ax.set_title("Validation: always goes up", loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5)
    for i, (m, label, col) in enumerate(reversed(mechs)):
        r = mf.paired(m, "a0", "gemini")
        bx.plot([r[1], r[2]], [i, i], color=col, lw=2, solid_capstyle="round")
        bx.scatter([r[0]], [i], s=40, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
        bx.text(r[2] + 0.004, i, f"{r[0]:+.3f}", va="center", fontsize=8.5, color=INK2)
    bx.axvline(0, color=AXIS, lw=1)
    bx.set_yticks(range(len(mechs)))
    bx.set_yticklabels([m[1] for m in reversed(mechs)], color=INK)
    bx.set_xlabel("Test recall vs. empty playbook\n(strict judge, 95% CI)")
    bx.set_title("Test: no gain", loc="left", color=INK, fontsize=10)
    bx.grid(axis="x", color=GRID, lw=0.6)
    fig.suptitle("Evolution keeps improving the score it is selected on, but the improvement does not transfer to test",
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "figD_rounds_vs_test")


# --- figE: asking vs using --------------------------------------------------------------------------------------
def _mean_field(f, key):
    vals = [json.loads(x).get(key) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    vals = [v for v in vals if v is not None]
    return (mean(vals), len(vals)) if vals else (None, 0)


def figE(cells):
    """cells: [(label, haiku, gemini)] computed by the caller from runs/probe_use."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    w = 0.36
    for i, (label, h, g) in enumerate(cells):
        for j, (v, col) in enumerate(((h, GREY), (g, RED))):
            if v is None:
                continue
            x = i + (j - 0.5) * w
            ax.bar(x, v, width=w - 0.04, color=col, label=JUDGES[j][1] if i == 0 else None)
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells], color=INK)
    ax.set_ylabel("Final-story score (key points, 0-6)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Both asking and using are bottlenecks: a stronger writer or a stronger asker\n"
                 "each more than doubles the 7B's story score (test, seed 0)", x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "figE_ask_vs_use")


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a != "--slide"] or ["A", "B", "C", "D"]
    if "A" in which:
        for label, c in figA():
            print("A", label, {j: tuple(round(x, 3) for x in v) for j, v in c.items()})
    if "B" in which:
        figB()
    if "C" in which:
        for label, c in figC():
            print("C", label, {j: tuple(round(x, 3) for x in v) for j, v in c.items()})
    if "D" in which:
        figD()
    print("written to", FIG)


def ask_use_cells():
    def m(f, key, sub=None):
        vals = []
        for line in (PROBE / f).read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            v = (x[sub] if sub else x).get(key)
            if v is not None:
                vals.append(v)
        return mean(vals), len(vals)
    spec = [("7B asks\n7B writes", "a0_orig.jsonl", None), ("7B asks\nQwen3-235B writes", "a0_qwen235.jsonl", None),
            ("7B asks\nOpus writes", "a0_opus.jsonl", None), ("Qwen3-235B asks\n7B writes", "strong_asker.jsonl", "strong_7b"),
            ("Qwen3-235B asks\nQwen3-235B writes", "strong_asker.jsonl", "strong_strong")]
    cells = []
    for label, f, sub in spec:
        (h, nh), (g, ng) = m(f, "haiku", sub), m(f, "gemini", sub)
        print("E", label.replace("\n", " / "), round(h, 3), nh, round(g, 3), ng)
        cells.append((label, h, g))
    return cells


if __name__ == "__main__" and "E" in sys.argv[1:]:
    figE(ask_use_cells())
