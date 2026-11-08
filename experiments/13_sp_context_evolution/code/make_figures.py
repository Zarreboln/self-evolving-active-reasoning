"""Figures for the context-evolution results (static PNG + SVG for the report and slides).
fig1: gain in key-point recall over the empty playbook (a0) per method, under each judge, paired over the 100 official
      SP test puzzles (seed-averaged within puzzle), 95% bootstrap CI.
fig2: key-point recall vs LLM calls per test episode (deployment cost, log x), one panel per judge.
fig3: validation recall of the incumbent playbook per evolution round, mean of 3 seeds, main judge (VAL is also used
      for selection, so these values are optimistic; the test set is the unbiased check).
Judges: Haiku = claude-haiku-5.5 + reasoning, per question (main); Gemini = gemini-3.7-flash, batched (robustness)."""
import json
import re
import sys
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "runs" / "test"
GEM = TEST / "rescore_google_gemini-3.7-flash_batched"
EVO = ROOT / "runs" / "evolve"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

NAMES = {"a0": "Empty playbook", "a2": "Claude playbook, no experience", "prior": "Evolved: hidden-story prior",
         "cce": "Evolved: step-level credit (CCE)", "cce_shuf": "Evolved: CCE, shuffled credit",
         "traj": "Evolved: trajectory contrast", "traj_gem": "Evolved: trajectory contrast, Gemini-judged",
         "r1": "Test-time best-of-8 (self-select)", "balar": "BALAR (Bayesian EIG, 2026)"}
CALLS = {"r1": 25 * 9 + 25 + 1, "balar": 296.6}          # others: 25 actor + 25 referee + 1 answer = 51
# Titles state the takeaway; set them only from finished numbers (see results notes).
TITLE1 = ("No evolved playbook beats the empty one under the stricter judge;\n"
          "test-time best-of-8 leads under both judges")
TITLE2 = ("Best-of-8 buys recall with ~5x the calls; BALAR (1 seed) uses ~6x and scores below empty\n"
          "(other playbooks sit at 51 calls: Fig. 1)")
FIG2_ARMS = ("a0", "traj", "r1", "balar")
JUDGES = (("haiku", "Haiku judge (main)", S1), ("gemini", "Gemini judge (stricter)", S2))


def _files(d, arm, ext):
    return [f for f in d.glob(f"{arm}_s*.{ext}") if re.fullmatch(rf"{re.escape(arm)}_s\d+\.{ext}", f.name)]


def per_puzzle(arm, judge):
    """{puzzle id: recall averaged over the arm's seeds}; Gemini uses only seed files rescored completely."""
    per = {}
    if judge == "haiku":
        for f in _files(TEST, arm, "jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                s = json.loads(line)["score"]
                per.setdefault(s["id"], []).append(s["key_recall"])
    else:
        for f in _files(GEM, arm, "json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            if all(v is not None for v in d.values()):
                for pid, v in d.items():
                    per.setdefault(pid, []).append(v)
    return {k: mean(v) for k, v in per.items()}


def n_seeds(arm, judge):
    return len(_files(TEST, arm, "jsonl")) if judge == "haiku" else len(_files(GEM, arm, "json"))


def paired(arm, base, judge):
    a, b = per_puzzle(arm, judge), per_puzzle(base, judge)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    return evaluate.boot([a[p] - b[p] for p in common])


def fig1(arms):
    rows = [a for a in arms if paired(a, "a0", "haiku") and paired(a, "a0", "gemini")]
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(rows) + 1.3))
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    for i, a in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, (judge, label, col) in enumerate(JUDGES):
            m, lo, hi = paired(a, "a0", judge)
            y = y0 + (0.16 if j == 0 else -0.16)
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([m], [y], s=40, color=col, edgecolor=SURF, linewidth=1.8, zorder=3,
                       label=label if i == 0 else None)
            ax.text(hi + 0.004, y, f"{m:+.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAMES[a] for a in rows][::-1], color=INK)
    ax.set_xlabel("Change in key-point recall vs. empty playbook\n"
                  "(official SP test, 100 puzzles, seeds averaged, paired 95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title(TITLE1, loc="left", color=INK, fontsize=11, pad=24)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig1_gain_vs_empty.{ext}", dpi=200)
    plt.close(fig)


def fig2(arms):
    arms = [a for a in arms if a in FIG2_ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True)
    for ax, (judge, label, col) in zip(axes, JUDGES):
        pts = []
        for a in arms:
            d = per_puzzle(a, judge)
            if not d:
                continue
            pts.append((a, CALLS.get(a, 51), mean(d.values())))
        for a, c, m in pts:
            x = c * {"a0": 0.95, "traj": 1.05}.get(a, 1.0)   # a0 and traj share x = 51 calls
            ax.scatter([x], [m], s=44, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
            ax.text(x * (1.08 if a != "balar" else 0.93), m, NAMES[a].replace("Evolved: ", ""), va="center",
                    ha="left" if a != "balar" else "right", fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([50, 100, 200, 300])
        ax.set_xticklabels(["50", "100", "200", "300"])
        ax.set_xlim(38, 420)
        ax.set_title(label, loc="left", color=INK, fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("LLM calls per test episode (log)")
    axes[0].set_ylabel("Key-point recall (mean over puzzles)")
    fig.suptitle(TITLE2, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig2_recall_vs_cost.{ext}", dpi=200)
    plt.close(fig)


def fig3():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ends = []
    for mech, col in (("cce", S1), ("cce_shuf", S2), ("traj", S3), ("prior", S4)):
        curves = []
        for s in (0, 1, 2):
            h = json.loads((EVO / f"{mech}_s{s}" / "history.json").read_text(encoding="utf-8"))
            curves.append([h[0]["val_recall"]] + [r.get("val_recall_incumbent", 0) for r in h[1:]])
        m = [mean(c[i] for c in curves) for i in range(len(curves[0]))]
        ax.plot(range(len(m)), m, color=col, lw=2, marker="o", ms=5, markeredgecolor=SURF, markeredgewidth=1.5)
        ends.append([m[-1], NAMES[mech].replace("Evolved: ", ""), len(m) - 1])
    ends.sort(key=lambda e: -e[0])                     # spread end labels so they never collide
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.008)
    for y, label, x in ends:
        ax.text(x + 0.12, y, label, va="center", fontsize=9, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["start", "round 1", "round 2", "round 3", "round 4"])
    ax.set_xlim(-0.2, 5.6)
    ax.set_ylabel("Validation key-point recall\n(incumbent playbook, mean of 3 seeds)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Validation recall across evolution rounds (main judge, used for selection)", loc="left", color=INK,
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig3_evolution_rounds.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    arms = [a for a in ("a2", "prior", "cce", "cce_shuf", "traj", "traj_gem", "r1", "balar")
            if _files(TEST, a, "jsonl")]
    fig1(arms)
    fig2(["a0"] + arms)
    fig3()
    for a in ["a0"] + arms:
        print(a, "seeds haiku/gemini:", n_seeds(a, "haiku"), n_seeds(a, "gemini"))
    print("figures written to", FIG)
