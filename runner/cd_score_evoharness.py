"""Score EvoHarness-RL Circuit Decoding trajectories with ar_harness's metrics.

Their records hold won/score and the env-action sequence; replaying the queries
through the public hypothesis space gives uniq@end, ex-ante efficiency,
zero-info%, oracle-agreement%, repeat% and illegal% exactly as cd_score.py
defines them, so the two stacks are compared on the same numbers.

    python runner/cd_score_evoharness.py <out_dir>[ <out_dir> ...]
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stage0"))
from cd_oracle import CDGame, load_instances  # noqa: E402

LABELS = "ABCDEFGH"
QUERY_RE = re.compile(r"query\s*([a-h])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)", re.I)
INSTS = load_instances(ROOT / "mt_puzzles/cd_test.jsonl")
GAMES = {}


def game(inst):
    key = tuple(sorted(inst["components"].items()))
    if key not in GAMES:
        GAMES[key] = CDGame(dict(key))
    return GAMES[key]


def replay(rec):
    k = int(rec["task_id"][2:])
    inst = INSTS[k]; g = game(inst); truth = inst["truth"]
    alive = np.arange(len(g.H))
    eff, zero, orcl, rep, ill, seen = [], [], [], [], 0, set()
    for t in rec["trace"]:
        if t["kind"] != "env":
            continue
        m = QUERY_RE.search(t["action"] or "")
        if not m:
            if not (t["action"] or "").lower().startswith("guess"):
                ill += 1
            continue
        q = LABELS.index(m.group(1).upper()) * 8 + int(m.group(2) + m.group(3) + m.group(4), 2)
        if len(alive) > 1:
            avail, best = g.available_bits(alive)
            chosen = g.expected_bits(q, alive)
            if avail > 0:
                eff.append(chosen / avail)
            zero.append(chosen < 1e-9)
            orcl.append(best is not None and q == best)
            rep.append(q in seen)
        seen.add(q)
        alive = alive[g.H[alive, q] == int(truth[q])]
    return {"uniq": len(alive) == 1, "joint_truth_table_correct": bool(rec["won"]), "circuit_wise_accuracy": float(rec["score"]),
            "eff": st.mean(eff) if eff else float("nan"), "zero": st.mean(zero) if zero else float("nan"),
            "orcl": st.mean(orcl) if orcl else float("nan"), "rep": st.mean(rep) if rep else float("nan"),
            "ill": ill, "meta": sum(rec["meta_actions"]["executed"].values()),
            "turns": rec["turns"], "n_alive_end": int(len(alive))}


def boot_ci(vals, n=2000, seed=0):
    a = np.asarray(vals, float); rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(dirs):
    # the benchmark's own metric first (circuit-wise accuracy), then joint truth table correct, then our diagnostics; see experiments/OFFICIAL_DEFINITIONS.md
    print(f"{'run':<34}{'n':>5}{'circuit-wise accuracy':>23}{'joint truth table correct':>27}{'uniq@end':>10}{'  95% CI':>14}{'eff':>7}{'zero%':>7}{'orcl%':>7}{'rep%':>6}{'ill/ep':>7}{'meta/ep':>8}{'turns':>7}")
    for d in dirs:
        recs = [json.loads(p.read_text()) for p in sorted(Path(d, "trajectories").glob("*.json"))]
        if not recs:
            print(f"{d:<34} (no trajectories)"); continue
        ms = [replay(r) for r in recs]
        mean = lambda k: st.mean(x[k] for x in ms if x[k] == x[k])
        lo, hi = boot_ci([m["uniq"] for m in ms])
        print(f"{Path(d).name:<34}{len(ms):>5}{100*mean('circuit_wise_accuracy'):>23.1f}{100*mean('joint_truth_table_correct'):>27.1f}"
              f"{100*mean('uniq'):>10.1f}{'[' + f'{100*lo:.1f},{100*hi:.1f}' + ']':>14}{mean('eff'):>7.3f}{100*mean('zero'):>7.1f}"
              f"{100*mean('orcl'):>7.1f}{100*mean('rep'):>6.1f}{mean('ill'):>7.2f}{mean('meta'):>8.2f}{mean('turns'):>7.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
