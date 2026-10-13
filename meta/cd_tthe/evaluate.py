"""Frozen evaluation of finished cd_tthe harnesses: no proposer, no judge. Each named harness in
agents/ plays every instance of the split once; the controller scores with the gold records.

    python evaluate.py --out logs/eval_<name> --harnesses bare cand_x cand_y [--split heldout] [--workers 48]
        [--exclude-stream-seed 0 --exclude-stream-limit 30]   # also report the instances the stream never used

Writes <out>/results/<harness>.jsonl (full records incl. gold, controller-side) and <out>/scores.json,
and prints a table: circuit-wise accuracy, joint truth table correct (with bootstrap CIs), uniq@end,
queries/episode, model calls, tokens, crashes.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bridge          # noqa: E402
import optimize as O   # noqa: E402


def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed); k = len(vals)
    ms = sorted(sum(rng.choice(vals) for _ in range(k)) / k for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def summarize(rs):
    g = [r.get("gold") or {} for r in rs]; s = [r.get("signals") or {} for r in rs]
    joint = [float(bool(x.get("joint_truth_table_correct"))) for x in g]
    cwa = [float(x.get("circuit_wise_accuracy") or 0) for x in g]
    return {"n": len(rs), "circuit_wise_accuracy": st.mean(cwa), "circuit_wise_accuracy_ci": boot_ci(cwa),
            "joint_truth_table_correct": st.mean(joint), "joint_truth_table_correct_ci": boot_ci(joint),
            "uniq_end": st.mean(float(bool(x.get("uniq_end"))) for x in s),
            "queries": st.mean(x.get("n_queries") or 0 for x in s),
            "calls": st.mean(r.get("solver_calls") or 0 for r in rs),
            "tokens_out": st.mean((r.get("tokens") or {}).get("out", 0) for r in rs),
            "crashes": sum(1 for r in rs if r.get("status") != "ok")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--harnesses", nargs="+", required=True)
    ap.add_argument("--arm", default="scratch"); ap.add_argument("--split", default="heldout"); ap.add_argument("--split-file", default=None)
    ap.add_argument("--budget", type=int, default=18); ap.add_argument("--solver-base-url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--solver-model", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--solver-mock", default=None)
    ap.add_argument("--solver-seed", type=int, default=0); ap.add_argument("--solve-timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude-stream-seed", type=int, default=None); ap.add_argument("--exclude-stream-limit", type=int, default=30)
    args = ap.parse_args()
    O.BUDGET_OVERRIDE = args.budget
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); (out / "results").mkdir(exist_ok=True)
    instances = bridge.make_stream(args.split, args.split_file, 0, None)
    instances = sorted(instances)[: args.limit] if args.limit else sorted(instances)
    seen = set(bridge.make_stream(args.split, args.split_file, args.exclude_stream_seed, args.exclude_stream_limit)) if args.exclude_stream_seed is not None else set()
    solver_cfg = {"base_url": args.solver_base_url, "model": args.solver_model, "mock": args.solver_mock,
                  "seed": args.solver_seed, "cache_dir": str(out / "solver_cache")}
    (out / "config.json").write_text(json.dumps({**vars(args), "instances": instances, "excluded_seen": sorted(seen)}, indent=1))
    scores = {}
    for name in args.harnesses:
        if not O.loadable(name, args.arm):
            print(f"{name}: not loadable under arm {args.arm}, skipped", flush=True); continue
        t0 = time.time()
        rs = O.observe(out, name, instances, solver_cfg, args.arm, args.solve_timeout, args.workers, "eval")
        with (out / "results" / f"{name}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        scores[name] = {"all": summarize(rs), "unseen": summarize([r for r in rs if r["index"] not in seen]) if seen else None,
                        "seconds": round(time.time() - t0)}
        a = scores[name]["all"]; u = scores[name]["unseen"]
        print(f"{name:<36} n={a['n']:<4} circuit-wise acc={100*a['circuit_wise_accuracy']:5.1f} [{100*a['circuit_wise_accuracy_ci'][0]:.1f},{100*a['circuit_wise_accuracy_ci'][1]:.1f}]  "
              f"joint={100*a['joint_truth_table_correct']:5.1f} [{100*a['joint_truth_table_correct_ci'][0]:.1f},{100*a['joint_truth_table_correct_ci'][1]:.1f}]  "
              f"uniq@end={100*a['uniq_end']:5.1f}  q/ep={a['queries']:4.1f}  calls={a['calls']:4.1f}  tok={a['tokens_out']:5.0f}  crash={a['crashes']}  "
              + (f"| unseen n={u['n']} joint={100*u['joint_truth_table_correct']:5.1f} cwa={100*u['circuit_wise_accuracy']:5.1f}" if u else "") + f"  ({scores[name]['seconds']}s)", flush=True)
        (out / "scores.json").write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
