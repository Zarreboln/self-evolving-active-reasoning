"""Test-time harness evolution (TTHE, arXiv 2607.08124) for Circuit Decoding.

One committed harness H walks an unlabeled stream in batches. Per batch:
  OBSERVE  run every active harness on the batch (isolated workers, hard timeouts), write traces
           and label-free signals;
  PROPOSE  G fixed branches x R rounds: branch g's proposer edits branch g's harness into a child,
           reading all branches' traces; an invalid child falls back to its parent;
  JUDGE    one agentic judge picks among every harness observed this batch, incumbent included
           (rollback gate); its pick becomes H for the next batch.
Scores kept for the record: transductive (the committed harness on this batch, from its cached
run) and prequential (the incoming harness on this batch, before adaptation) -- the latter is
the deployed-performance curve TTHE's limitations section asks for. Gold (exact match) is only
written into the logs; the loop never reads it.

    python optimize.py --run-name pilot --arm scratch --batch-size 10 --group 3 --rounds 2 \
        --split heldout --limit 60 --solver-base-url http://127.0.0.1:18000/v1 \
        --solver-model Qwen/Qwen2.5-7B-Instruct --proposer-backend claude
    python optimize.py --run-name dry --proposer-backend stub --solver-mock sequential --limit 10 \
        --batch-size 5 --group 2 --rounds 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bridge          # noqa: E402
import proposer as P   # noqa: E402
from audit import audit  # noqa: E402

AGENTS = HERE / "agents"
BUDGET_OVERRIDE = 18


def loadable(name, arm):
    p = AGENTS / f"{name}.py"
    return p.exists() and not audit(p.read_text(), arm)


def controller_dir(run_dir):
    """Private store of the controller (gold records, oracle-derived signals, full logs): outside the
    repository, so a proposer or judge session that reads the run directory never sees labels. Copied
    into <run>/controller/ once the run is over."""
    d = Path.home() / ".ar_harness_controller" / Path(run_dir).name
    d.mkdir(parents=True, exist_ok=True)
    return d


def public_agg(a, labeled):
    keys = ("n", "health", "tokens_out") + (("circuit_wise_accuracy", "joint_truth_table_correct") if labeled else ())
    return {k: a[k] for k in keys if k in a}


def solve_one(run_dir, name, index, solver_cfg, arm, timeout, tag):
    jobs = controller_dir(run_dir) / "solve_jobs"; jobs.mkdir(parents=True, exist_ok=True)
    task, out = jobs / f"{tag}.task.json", jobs / f"{tag}.result.json"
    out.unlink(missing_ok=True)
    task.write_text(json.dumps({"harness": name, "index": index, "output": str(out), "solver": solver_cfg, "arm": arm,
                                "budget": BUDGET_OVERRIDE, "reveal": "controller"}))
    proc = subprocess.Popen([sys.executable, "-u", str(HERE / "solve_worker.py"), "--task", str(task)],
                            cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        _, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return {"harness": name, "index": index, "status": "timeout", "error": f"wall timeout {timeout}s",
                "signals": {"health": False, "crashed": True, "n_queries": None, "uniq_end": False, "n_alive_end": None,
                            "zero_info": None, "repeat": None, "oracle_agree": None, "eff": None},
                "gold": {"joint_truth_table_correct": False, "circuit_wise_accuracy": 0.0, "parsed_guess": False}, "history": [], "trace": [], "tokens": {}}
    if not out.exists():
        return {"harness": name, "index": index, "status": "worker_error", "error": (err or "")[-800:],
                "signals": {"health": False, "crashed": True, "n_queries": None, "uniq_end": False, "n_alive_end": None,
                            "zero_info": None, "repeat": None, "oracle_agree": None, "eff": None},
                "gold": {"joint_truth_table_correct": False, "circuit_wise_accuracy": 0.0, "parsed_guess": False}, "history": [], "trace": [], "tokens": {}}
    return json.loads(out.read_text())


def observe(run_dir, name, batch, solver_cfg, arm, timeout, workers, tag):
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(lambda i: solve_one(run_dir, name, i, solver_cfg, arm, timeout, f"{tag}_{name}_i{i}"), batch))
    return results


def agg(results):
    sig = [r.get("signals") or {} for r in results]
    n = max(1, len(sig))
    return {"n": len(sig),
            "uniq_end": sum(1 for s in sig if s.get("uniq_end")) / n,
            "health": sum(1 for s in sig if s.get("health")) / n,
            "circuit_wise_accuracy": sum(float((r.get("gold") or {}).get("circuit_wise_accuracy") or 0) for r in results) / n,   # record only
            "joint_truth_table_correct": sum(1 for r in results if (r.get("gold") or {}).get("joint_truth_table_correct")) / n,   # record only
            "tokens_out": sum((r.get("tokens") or {}).get("out", 0) for r in results) / n}


def log_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--arm", choices=["scratch", "lib"], default="scratch")
    ap.add_argument("--initial-harness", default="bare")
    ap.add_argument("--split", default="heldout"); ap.add_argument("--split-file", default=None)
    ap.add_argument("--stream-seed", type=int, default=0); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--budget", type=int, default=18, help="queries per episode (the benchmark's 18; 12 keeps headroom)")
    ap.add_argument("--group", type=int, default=3, help="G fixed branches")
    ap.add_argument("--rounds", type=int, default=2, help="R proposer rounds per batch")
    ap.add_argument("--solver-base-url", default=None); ap.add_argument("--solver-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--solver-mock", default=None, help="CPU test: scripted task model")
    ap.add_argument("--solver-seed", type=int, default=0)
    ap.add_argument("--proposer-backend", choices=["claude", "solver", "stub"], default="claude")
    ap.add_argument("--proposer-model", default="opus")
    ap.add_argument("--propose-timeout", type=int, default=900); ap.add_argument("--judge-timeout", type=int, default=900)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--signals", choices=["strict", "labeled"], default="strict",
                    help="strict: raw traces + run status only; labeled: strict + each episode's final outcome")
    ap.add_argument("--solve-timeout", type=int, default=600, help="wall-clock cap per episode")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--start-batch", type=int, default=0)
    args = ap.parse_args()

    global BUDGET_OVERRIDE
    BUDGET_OVERRIDE = args.budget
    P.BUDGET = args.budget
    P.STRICT = True                       # the oracle-derived signal arm was dropped (2026-10-12)
    P.LABELED = args.signals == "labeled"
    run_dir = HERE / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stream = bridge.make_stream(args.split, args.split_file, args.stream_seed, args.limit)
    batches = [stream[i:i + args.batch_size] for i in range(0, len(stream), args.batch_size)]
    solver_cfg = {"base_url": args.solver_base_url, "model": args.solver_model, "mock": args.solver_mock,
                  "seed": args.solver_seed, "cache_dir": str(run_dir / "solver_cache")}
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "stream": stream, "n_batches": len(batches)}, indent=1))
    opt_log, lineage_log = run_dir / "opt_log.jsonl", run_dir / "branch_lineage.jsonl"
    ctrl = controller_dir(run_dir)

    H = args.initial_harness
    if args.start_batch > 0:   # resume: the committed harness of the previous batch
        rows = [json.loads(l) for l in opt_log.open()] if opt_log.exists() else []
        prev = [r for r in rows if r.get("batch") == args.start_batch - 1 and "committed" in r]
        if prev:
            H = prev[-1]["committed"]
    if not loadable(H, args.arm):
        sys.exit(f"initial harness {H} not loadable under arm {args.arm}")
    print(f"[tthe] run={args.run_name} arm={args.arm} stream={len(stream)} batches={len(batches)} B={args.batch_size} "
          f"G={args.group} R={args.rounds} backend={args.proposer_backend} H0={H}", flush=True)

    for bi in range(args.start_batch, len(batches)):
        batch = batches[bi]
        t_batch = time.time()
        trace_dir = run_dir / "traces" / f"b{bi}"
        results = {}           # harness name -> results on this batch (observed once)

        def ensure(name):
            if name not in results and loadable(name, args.arm):
                results[name] = observe(run_dir, name, batch, solver_cfg, args.arm, args.solve_timeout, args.workers, f"b{bi}")
                P.write_traces(results[name], trace_dir, labeled=P.LABELED)

        ensure(H)
        prequential = agg(results[H])          # the incoming harness, before adaptation
        print(f"\n===== batch {bi}/{len(batches)} ({len(batch)} inst) H={H}  prequential circuit-wise acc={100*prequential['circuit_wise_accuracy']:.0f}% "
              f"joint table correct={100*prequential['joint_truth_table_correct']:.0f}% (diag uniq_end={100*prequential['uniq_end']:.0f}%) =====", flush=True)
        branches = [H] * args.group
        for rnd in range(args.rounds):
            tag = f"b{bi}r{rnd}"
            active = list(dict.fromkeys(branches))
            children = []
            with ThreadPoolExecutor(max_workers=args.group) as ex:
                futs = []
                for gi, parent in enumerate(branches):
                    child = f"cand_{args.run_name}_b{bi}r{rnd}g{gi}"
                    futs.append((gi, parent, child, ex.submit(
                        P.propose, parent, child, gi, trace_dir, active,
                        {n: results[n] for n in active if n in results}, run_dir, tag, batch,
                        solver_cfg, args.arm, args.proposer_backend, args.proposer_model,
                        args.propose_timeout, args.solve_timeout, args.effort)))
                for gi, parent, child, fut in futs:
                    try:
                        ok = fut.result()
                    except Exception as e:
                        ok = False; print(f"   proposer g{gi} failed: {e}", flush=True)
                    children.append((gi, parent, child, ok))
            next_branches = []
            for gi, parent, child, ok in children:
                accepted = child if ok and loadable(child, args.arm) else parent
                if accepted == child:
                    ensure(child)
                next_branches.append(accepted)
                lrow = {"batch": bi, "round": rnd, "branch": gi, "parent": parent, "child": child,
                        "accepted": accepted, "fell_back": accepted == parent,
                        "child_agg": agg(results[child]) if child in results else None}
                log_line(ctrl / "branch_lineage_full.jsonl", lrow)
                log_line(lineage_log, {**lrow, "child_agg": public_agg(lrow["child_agg"], P.LABELED) if lrow["child_agg"] else None})
            branches = next_branches
            print(f"   round {rnd}: " + ", ".join(f"g{gi}:{b}" for gi, b in enumerate(branches)), flush=True)

        pool = list(results.keys())    # everything observed this batch, incumbent first (rollback gate)
        chosen = P.judge(pool, H, trace_dir, results, run_dir, f"b{bi}", batch, solver_cfg, args.arm,
                         args.proposer_backend, args.proposer_model, args.judge_timeout, args.solve_timeout, args.effort)
        H_next = chosen if chosen in results else H
        transductive = agg(results[H_next])
        row = {"batch": bi, "instances": batch, "incoming": H, "committed": H_next,
               "prequential": prequential, "transductive": transductive,
               "pool": {n: agg(results[n]) for n in pool}, "seconds": round(time.time() - t_batch)}
        log_line(ctrl / "opt_log_full.jsonl", row)
        log_line(opt_log, {**row, "prequential": public_agg(row["prequential"], P.LABELED),
                           "transductive": public_agg(row["transductive"], P.LABELED),
                           "pool": {n: public_agg(a, P.LABELED) for n, a in row["pool"].items()}})
        print(f"   JUDGE -> {H_next}   transductive circuit-wise acc={100*transductive['circuit_wise_accuracy']:.0f}% joint table correct={100*transductive['joint_truth_table_correct']:.0f}% "
              f"(diag uniq_end={100*transductive['uniq_end']:.0f}%)   ({row['seconds']}s)", flush=True)
        H = H_next

    # the run is over: the private store becomes part of the record (full logs replace the public ones)
    shutil.copy(ctrl / "opt_log_full.jsonl", opt_log)
    if (ctrl / "branch_lineage_full.jsonl").exists():
        shutil.copy(ctrl / "branch_lineage_full.jsonl", lineage_log)
    if (ctrl / "solve_jobs").exists():
        shutil.copytree(ctrl / "solve_jobs", run_dir / "controller" / "solve_jobs", dirs_exist_ok=True)
    rows = [json.loads(l) for l in opt_log.open()]
    def curve(key):
        n = sum(len(r["instances"]) for r in rows)
        return {m: sum(r[key][m] * len(r["instances"]) for r in rows) / n for m in ("uniq_end", "circuit_wise_accuracy", "joint_truth_table_correct")}
    pq, tr = curve("prequential"), curve("transductive")
    summary = {"final_harness": H, "n_instances": sum(len(r["instances"]) for r in rows),
               "prequential": pq, "transductive": tr,
               "per_batch": [{"batch": r["batch"], "incoming": r["incoming"], "committed": r["committed"],
                              "preq_uniq": r["prequential"]["uniq_end"], "trans_uniq": r["transductive"]["uniq_end"],
                              "preq_circuit_wise_accuracy": r["prequential"]["circuit_wise_accuracy"], "trans_circuit_wise_accuracy": r["transductive"]["circuit_wise_accuracy"],
                              "preq_joint_truth_table_correct": r["prequential"]["joint_truth_table_correct"], "trans_joint_truth_table_correct": r["transductive"]["joint_truth_table_correct"]} for r in rows]}
    (run_dir / "result.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[tthe] final H={H}  prequential circuit-wise acc={100*pq['circuit_wise_accuracy']:.1f}% joint table correct={100*pq['joint_truth_table_correct']:.1f}% (uniq_end {100*pq['uniq_end']:.1f}%)  "
          f"transductive circuit-wise acc={100*tr['circuit_wise_accuracy']:.1f}% joint table correct={100*tr['joint_truth_table_correct']:.1f}%  -> {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
