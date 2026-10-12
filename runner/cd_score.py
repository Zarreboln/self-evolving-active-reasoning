"""Score CircuitDecoding episode logs written by runner/cd_harness_run.py.

    python runner/cd_score.py results/cd/*/search.jsonl --split search
    python runner/cd_score.py results/cd/enum15/search.jsonl --vs results/cd/bare/search.jsonl --out score.json

Episode-level metrics (one row per harness):
    uniq@end   feasible set collapsed to one hypothesis after the budget (ours, diagnostic;
               measures information gathering, immune to final-guess luck)
    joint truth table correct   the whole 24-bit joint truth table is right (our literal
               reading of a successful episode; the paper defines none for this task)
    circuit-wise accuracy   fraction of the 3 circuits whose full truth table is right --
               the benchmark's own metric (MTP Table 1; its normalization is undefined)
    bit        bit-wise accuracy of the final guess (ours)
    parsed     final guess was in the required format
    alive_end  mean |feasible set| after the budget
Turn-level metrics (over legal turns while >1 hypothesis remained):
    eff        ex-ante efficiency = expected bits of the chosen query / best available
    zero%      queries whose answer was already determined (0 bits)
    oracle%    queries equal to the max-entropy query
    repeat%    queries identical to an earlier query
    illegal%   turns with no parsable query (over all turns)
95% bootstrap CIs on uniq@end and joint truth table correct; paired bootstrap on the delta vs --vs.
"""

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(path, split=None, split_file=ROOT / "splits/cd_split.json"):
    latest = {}
    for line in Path(path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        latest[(r["condition"], r["key"], r["seed"])] = r  # last write wins (retries)
    if split and split != "all":
        keep = set(json.loads(Path(split_file).read_text())[split])
        latest = {k: v for k, v in latest.items() if k[1] in keep}
    return latest


def episode_metrics(r):
    res = r.get("result") or {}
    g, t = res.get("guess_bits"), res.get("truth_bits")
    m = {"err": bool(r.get("error")), "uniq": bool(res.get("unique_at_end")),
         "joint_truth_table_correct": bool(res.get("correct")), "parsed": bool(res.get("parsed_guess")),
         "alive_end": res.get("n_alive_end"), "tok_out": r.get("out_tokens", 0),
         "tok_in": r.get("in_tokens", 0)}
    if g and t and len(g) == len(t):
        k = len(t) // 3
        m["bit"] = sum(x == y for x, y in zip(g, t)) / len(t)
        m["circuit_wise_accuracy"] = sum(g[i * k:(i + 1) * k] == t[i * k:(i + 1) * k] for i in range(3)) / 3
    return m


def turn_metrics(rows):
    out = defaultdict(list)
    for q in rows:
        out["illegal"].append(bool(q.get("illegal")))
        if q.get("illegal") or q.get("n_before", 0) <= 1:
            continue
        if q.get("efficiency") is not None:
            out["eff"].append(q["efficiency"])
        out["zero"].append(bool(q.get("zero_info")))
        out["repeat"].append(bool(q.get("repeat")))
        if q.get("best_query") is not None:
            out["oracle"].append(q["query"] == q["best_query"])
    return out


def boot_ci(vals, n_boot=2000, seed=0):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    means = a[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_delta(a_by_key, b_by_key, field, n_boot=2000, seed=0):
    keys = sorted(set(a_by_key) & set(b_by_key))
    if not keys:
        return None
    d = np.array([float(a_by_key[k][field]) - float(b_by_key[k][field]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"n_pairs": len(keys), "delta": float(d.mean()),
            "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def summarize(latest):
    eps = {k[1:]: episode_metrics(v) for k, v in latest.items()}
    turns = defaultdict(list)
    for v in latest.values():
        for kk, vv in turn_metrics(v.get("rows") or []).items():
            turns[kk] += vv
    n = len(eps)
    mean = lambda f: (st.mean(x[f] for x in eps.values() if x.get(f) is not None)
                      if any(x.get(f) is not None for x in eps.values()) else float("nan"))
    tmean = lambda f: st.mean(turns[f]) if turns[f] else float("nan")
    s = {"n": n, "err": mean("err"), "uniq": mean("uniq"), "joint_truth_table_correct": mean("joint_truth_table_correct"),
         "circuit_wise_accuracy": mean("circuit_wise_accuracy"), "bit": mean("bit"), "parsed": mean("parsed"),
         "alive_end": mean("alive_end"), "eff": tmean("eff"), "zero": tmean("zero"),
         "oracle": tmean("oracle"), "repeat": tmean("repeat"), "illegal": tmean("illegal"),
         "tok_out": mean("tok_out"), "tok_in": mean("tok_in"),
         "uniq_ci": boot_ci([x["uniq"] for x in eps.values()]),
         "joint_truth_table_correct_ci": boot_ci([x["joint_truth_table_correct"] for x in eps.values()])}
    return s, eps


# Column order: the benchmark's own metric first ("circuit-wise accuracy", MTP Table 1), then
# "joint truth table correct" (the whole 24-bit joint truth table right -- our literal reading of a
# successful episode; the paper defines no success event for this task), then our diagnostics
# (uniq@end, bit, eff, ...). See experiments/OFFICIAL_DEFINITIONS.md.
HDR = (f"{'harness':<14}{'n':>5}{'err%':>6}{'circuit-wise accuracy':>23}{'joint truth table correct':>27}{'uniq@end':>10}{'  95% CI':>14}{'bit':>7}"
       f"{'eff':>7}{'zero%':>7}{'orcl%':>7}{'rep%':>6}{'ill%':>6}{'prs%':>6}{'alive':>7}{'tok':>7}")
# Errored episodes (context overflow, API failures) count as NOT unique / NOT correct:
# they are real failures of that harness at this context budget. Turn-level metrics
# and `alive` are computed over the completed episodes only. err% makes this visible.


def fmt_row(name, s):
    pc = lambda v: "  n/a" if v != v else f"{100 * v:.1f}"
    return (f"{name:<14}{s['n']:>5}{pc(s['err']):>6}{pc(s['circuit_wise_accuracy']):>23}{pc(s['joint_truth_table_correct']):>27}{pc(s['uniq']):>10}"
            f"{'[' + pc(s['uniq_ci'][0]) + ',' + pc(s['uniq_ci'][1]) + ']':>14}"
            f"{pc(s['bit']):>7}"
            f"{s['eff']:>7.3f}{pc(s['zero']):>7}{pc(s['oracle']):>7}{pc(s['repeat']):>6}"
            f"{pc(s['illegal']):>6}{pc(s['parsed']):>6}{s['alive_end']:>7.2f}{s['tok_out']:>7.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--split", default="all", choices=["all", "search", "heldout"])
    p.add_argument("--split-file", default=str(ROOT / "splits/cd_split.json"))
    p.add_argument("--vs", default=None, help="baseline jsonl for a paired delta")
    p.add_argument("--out", default=None, help="write the (single-file) summary JSON here")
    a = p.parse_args()

    base = None
    if a.vs:
        _, base = summarize(load(a.vs, a.split, a.split_file))
    print(HDR)
    print("-" * len(HDR))
    summaries = {}
    for f in a.files:
        latest = load(f, a.split, a.split_file)
        if not latest:
            print(f"{Path(f).parent.name + '/' + Path(f).name:<14} (no episodes in split {a.split})")
            continue
        conds = sorted({k[0] for k in latest})
        for c in conds:
            sub = {k: v for k, v in latest.items() if k[0] == c}
            s, eps = summarize(sub)
            if base:
                s["vs"] = {fld: paired_delta(eps, base, fld) for fld in ("uniq", "joint_truth_table_correct")}
            summaries[c] = s
            print(fmt_row(c, s))
            if base and s["vs"]["uniq"]:
                d = s["vs"]
                vs_name = Path(a.vs).stem if Path(a.vs).stem not in ("search", "heldout") else Path(a.vs).parent.name
                print(f"{'':<14}  vs {vs_name}: uniq {100*d['uniq']['delta']:+.1f} "
                      f"[{100*d['uniq']['ci'][0]:+.1f},{100*d['uniq']['ci'][1]:+.1f}]  "
                      f"joint truth table correct {100*d['joint_truth_table_correct']['delta']:+.1f} "
                      f"[{100*d['joint_truth_table_correct']['ci'][0]:+.1f},{100*d['joint_truth_table_correct']['ci'][1]:+.1f}]  "
                      f"(n={d['uniq']['n_pairs']} paired)")
    if a.out:
        Path(a.out).write_text(json.dumps(summaries, indent=1))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
