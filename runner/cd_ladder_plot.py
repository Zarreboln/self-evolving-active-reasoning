"""Actionability-ladder figure: uniq@end and ex-ante efficiency per harness (our diagnostics; the
benchmark's own metric, circuit-wise accuracy, is in the score JSON)
with 95% bootstrap CIs, from cd_score.py --out JSON files.

    python runner/cd_score.py results/cd_ladder/*.jsonl --split all --out results/cd_ladder/scores.json
    python runner/cd_ladder_plot.py results/cd_ladder/scores.json --out results/cd_ladder/ladder.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["bare", "facts", "symbolic", "enum15", "t3_gate", "splits", "oracle_hint"]
LABEL = {"bare": "bare", "facts": "(a) facts", "symbolic": "symbolic", "enum15": "enum15\n(hand-written)",
         "t3_gate": "T3-gated", "splits": "(c) split counts", "oracle_hint": "(d) oracle hint"}
REF = {"random fresh": 0.629, "sequential": 0.540}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores")
    p.add_argument("--out", default="ladder.png")
    p.add_argument("--title", default="Circuit Decoding, Qwen2.5-7B-Instruct, 300 instances x 2 seeds")
    a = p.parse_args()
    S = json.load(open(a.scores))
    names = [n for n in ORDER if n in S] + [n for n in S if n not in ORDER]
    x = range(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    u = [100 * S[n]["uniq"] for n in names]
    lo = [100 * (S[n]["uniq"] - S[n]["uniq_ci"][0]) for n in names]
    hi = [100 * (S[n]["uniq_ci"][1] - S[n]["uniq"]) for n in names]
    ax[0].bar(x, u, yerr=[lo, hi], capsize=3, color="#4c72b0")
    ax[0].set_ylabel("uniq@end (%)")
    ax[0].set_ylim(0, 100)
    ex = [100 * S[n]["joint_truth_table_correct"] for n in names]
    for i, v in enumerate(ex):
        ax[0].text(i, u[i] + hi[i] + 2, f"joint table {v:.1f}", ha="center", fontsize=7)
    e = [S[n]["eff"] for n in names]
    ax[1].bar(x, e, color="#dd8452")
    for k, v in REF.items():
        ax[1].axhline(v, ls="--", lw=0.8, color="gray")
        ax[1].text(len(names) - 0.5, v + 0.01, k, ha="right", fontsize=7, color="gray")
    ax[1].set_ylabel("ex-ante efficiency")
    ax[1].set_ylim(0, 1.0)
    for axx in ax:
        axx.set_xticks(list(x))
        axx.set_xticklabels([LABEL.get(n, n) for n in names], fontsize=8, rotation=20, ha="right")
    fig.suptitle(a.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("->", a.out)


if __name__ == "__main__":
    main()
