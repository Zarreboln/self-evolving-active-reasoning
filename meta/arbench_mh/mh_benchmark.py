"""The benchmark.py Meta-Harness's loop shells out to, for one AR-Bench task.

The loop never imports this; it runs it with the flags below and reads the files it writes. The
frontier mathematics is theirs, copied rather than reimplemented, so that "we ran their loop" stays
a checkable claim:

    load_results            walks <dir>/<dataset>/<system>/<model>/val.json
    compute_pareto_frontier a point survives if no other is at least as accurate and no more
                            expensive, with at least one strict improvement
    print_frontier          best per dataset, then the aggregate Pareto, then frontier_val.json

Two things are ours. The cost axis is `total_tokens`, input plus output, where theirs is
`memory_context_chars`: a method can read modestly and generate enormously, and generation is
the side that takes the wall-clock, so input alone can understate a run by a factor of four. And the evaluation path runs a harness over a split of AR-Bench
instances instead of a memory system over text classification.

    python mh_benchmark.py --task dc --memory baseline_tot      # ships to the cluster, scores, pulls back
    python mh_benchmark.py --task dc --frontier --model Qwen2.5-7B-Instruct
    python mh_benchmark.py --task dc --results
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from collections import defaultdict

MODEL = os.environ.get("POLICY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SHORT = MODEL.split("/")[-1]
SEEDS = {
    "gn": ["baseline_zero_shot", "baseline_tot_fixed", "baseline_proactive_cot"],
    "dc": ["baseline_few_shot_inst", "baseline_proactive_stop", "baseline_tot"],
    "sp": ["baseline_zero_shot", "baseline_few_shot_inst", "baseline_tot_eval"],
}


def run_dir(base, dataset, system, model, seed=None):
    d = os.path.join(base, dataset, system, model if seed is None else "%s_seed%d" % (model, seed))
    return d


def load_results(base, filename):
    """<base>/<dataset>/<system>/<model>/<filename> -> {(model, dataset, system): data}"""
    out = {}
    if not os.path.isdir(base):
        return out
    for dataset in sorted(os.listdir(base)):
        dd = os.path.join(base, dataset)
        if not os.path.isdir(dd):
            continue
        for system in sorted(os.listdir(dd)):
            sd = os.path.join(dd, system)
            if not os.path.isdir(sd):
                continue
            for model in sorted(os.listdir(sd)):
                f = os.path.join(sd, model, filename)
                if os.path.exists(f):
                    try:
                        out[(model, dataset, system)] = json.load(open(f))
                    except Exception:
                        pass
    return out


def compute_pareto_frontier(points):
    """Theirs, unchanged: (name, accuracy, cost) -> the non-dominated points, best first."""
    pareto = [p for p in points
              if not any(o_acc >= p[1] and o_cost <= p[2] and (o_acc > p[1] or o_cost < p[2])
                         for _, o_acc, o_cost in points)]
    return sorted(pareto, key=lambda x: (-x[1], x[2]))


def print_frontier(logs_dir, results_dir, task, model_filter, metric):
    base = results_dir if metric == "test" else logs_dir
    results = load_results(base, "test.json" if metric == "test" else "val.json")
    if model_filter:
        results = {k: v for k, v in results.items() if k[0] == model_filter}
    if not results:
        print("No results found")
        return

    by_dataset = defaultdict(list)
    by_system = defaultdict(lambda: {"acc": [], "cost": []})
    for (model, dataset, system), d in results.items():
        acc = (d.get("accuracy") or 0) * 100
        cost = d.get("total_tokens", 0)
        by_dataset[dataset].append({"system": system, "accuracy": acc, "cost": cost})
        by_system[system]["acc"].append(acc)
        by_system[system]["cost"].append(cost)

    frontier = {}
    for dataset, rows in by_dataset.items():
        best = max(rows, key=lambda x: (x["accuracy"], -x["cost"]))
        frontier[dataset] = {"best_system": best["system"], "accuracy": best["accuracy"],
                             "total_tokens": best["cost"]}

    print("\n" + "=" * 64)
    print("FRONTIER [%s] %s" % (metric, "(model: %s)" % model_filter if model_filter else ""))
    print("=" * 64)
    for dataset, info in frontier.items():
        print("  %s: %s (%.1f%%, %s tokens/ep)"
              % (dataset, info["best_system"], info["accuracy"], f"{int(info['total_tokens']):,}"))

    points = [(s, sum(v["acc"]) / len(v["acc"]), int(sum(v["cost"]) / len(v["cost"])))
              for s, v in by_system.items()]
    pareto = compute_pareto_frontier(points)
    print("\nPARETO FRONTIER [%s] (accuracy vs total tokens per episode):" % metric)
    print("  %-30s %8s %14s" % ("system", "acc", "tokens/ep"))
    print("  " + "-" * 54)
    for name, acc, cost in pareto:
        print("  %-30s %7.1f%% %14s" % (name, acc, f"{cost:,}"))

    frontier["_pareto"] = [{"system": n,
                            ("val_accuracy" if metric == "val" else "test_accuracy"): round(a, 1),
                            "total_tokens": c} for n, a, c in pareto]
    path = os.path.join(logs_dir, "frontier_val.json" if metric == "val" else "frontier.json")
    os.makedirs(logs_dir, exist_ok=True)
    json.dump(frontier, open(path, "w"), indent=2)
    print("\nWrote %s" % path)


def print_results(results, label):
    print("\n%-30s %-10s %10s %14s" % ("system", "dataset", label, "tokens/ep"))
    print("-" * 68)
    for (model, dataset, system), d in sorted(results.items(), key=lambda kv: -(kv[1].get("accuracy") or 0)):
        print("%-30s %-10s %9.1f%% %14s"
              % (system, dataset, 100 * (d.get("accuracy") or 0),
                 f"{int(d.get('total_tokens', 0)):,}"))


# Where the episodes run. Empty (or "local") means this machine is the cluster, which is the
# normal case now that uv and the claude CLI are installed there: the loop, the proposer and the
# episodes all sit beside AR-Bench and the served model, and nothing depends on a laptop staying
# awake for a forty-minute proposer session.
HOST = os.environ.get("MH_HOST", "")
REMOTE = "code/arbench_mh"


def _rename_for_test(d, split, rc):
    """score_val writes val.json whatever the split, and load_results looks for test.json on the
    test side. The remote path renames on the way back (see `evaluate`); the local path never had
    to, because finalization had not run on this machine before. Without this the episodes all
    run, the scores all land, and finalization reports every group missing.
    """
    if rc == 0 and split == "test":
        src, dst = os.path.join(d, "val.json"), os.path.join(d, "test.json")
        if os.path.exists(src):
            os.replace(src, dst)
    return rc


def _evaluate_here(task, system, remote_split, d, split, workers, here):
    """Everything on one machine: no shipping, no copying back."""
    py = os.environ.get("MH_PYTHON", os.path.expanduser("~/venv/bin/python"))
    episodes = os.path.join(d, "episodes.json")
    if task == "amigo":
        # amigo scores itself: accuracy is a plain exact match and the chance rate differs per
        # gallery, so there is nothing for score_val's judge machinery to do, and keeping it in
        # the package means a cleanup of a scratch directory cannot take the scorer with it.
        cmd = [py, "-I", os.path.join(here, "amigo", "benchmark.py"),
               "--tier", os.environ.get("AMIGO_TIER", "tau06"),
               "--harness", system, "--split", remote_split, "--out", episodes,
               "--workers", str(workers), "--val-out", os.path.join(d, "val.json")]
        return _rename_for_test(d, split,
                                subprocess.run(cmd, env=os.environ.copy()).returncode)
    cmd = [py, "-I", os.path.join(here, "benchmark.py"), "--task", task]
    if task == "sp":
        cmd = [py, "-I", os.path.join(here, "sp", "benchmark.py")]
    cmd += ["--harness", system, "--split", remote_split, "--out", episodes,
            "--workers", str(workers)]
    if subprocess.run(cmd, env=os.environ.copy()).returncode != 0:
        return 1
    return _rename_for_test(d, split, subprocess.run(
        [py, "-I", os.path.expanduser("~/code/score_val.py"), episodes,
         "--task", task, "--system", system,
         "--logs-dir", os.path.dirname(os.path.dirname(os.path.dirname(d)))],
        env=os.environ.copy()).returncode)


def evaluate(task, system, split, logs_dir, results_dir, workers):
    """Run one harness on the cluster, then bring its val.json back here.

    The loop and the proposer run on this machine because `uv` and the `claude` CLI are only here;
    the episodes have to run on the cluster because AR-Bench, the splits and the served model are
    only there. So a candidate is shipped over, evaluated and scored remotely, and the one file the
    loop actually reads is copied back -- `load_results` walks the local tree as it always has.
    """
    # The loop speaks val/test; our splits are named for their job. "selection" is what the
    # frontier is computed from, and "test" is AR-Bench's own test.json, sealed until finalization.
    remote_split = {"val": "selection", "test": "test"}[split]
    base = results_dir if split == "test" else logs_dir
    d = run_dir(base, task, system, SHORT)
    os.makedirs(d, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, task, "agents", system + ".py")
    if not os.path.exists(src):
        print("no such candidate: %s" % src)
        return 1

    if not HOST or HOST == "local":
        return _evaluate_here(task, system, remote_split, d, split, workers, here)

    if subprocess.run(["scp", "-q", "-o", "BatchMode=yes", src,
                       "%s:%s/%s/agents/" % (HOST, REMOTE, task)]).returncode != 0:
        return 1

    rbase = "~/results/mh_remote/%s/%s" % (task, split)
    script = (
        'set -e; EP=$(cat ~/results/serve_qwen25_7b_32k_v2.endpoint); '
        'export POLICY_BASE_URL="http://$EP/v1" POLICY_API_KEY=x '
        'RESPONSE_BASE_URL="http://127.0.0.1:8099/v1" RESPONSE_API_KEY=x; '
        'mkdir -p %(rb)s; '
        '~/venv/bin/python -I ~/%(rem)s/benchmark.py --task %(t)s --harness %(s)s '
        '--split %(sp)s --out %(rb)s/%(s)s.json --workers %(w)d; '
        '~/venv/bin/python -I ~/code/score_val.py %(rb)s/%(s)s.json --task %(t)s '
        '--system %(s)s --logs-dir %(rb)s/scored'
    ) % {"rb": rbase, "rem": REMOTE, "t": task, "s": system, "sp": remote_split, "w": workers}
    if subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, script]).returncode != 0:
        return 1

    name = "test.json" if split == "test" else "val.json"
    remote_val = "%s/scored/%s/%s/%s/val.json" % (rbase, task, system, SHORT)
    return subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                           "%s:%s" % (HOST, remote_val), os.path.join(d, name)]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("gn", "dc", "sp", "amigo"))
    ap.add_argument("--memory", help="evaluate this harness")
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--results", action="store_true")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--test", action="store_true", help="the held-out split; finalization only")
    ap.add_argument("--model", default=None)
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    root = os.path.expanduser("~/results/mh/%s" % a.task)
    logs_dir = a.logs_dir or os.path.join(root, "logs")
    results_dir = a.results_dir or os.path.join(root, "results")
    metric = "test" if a.test else "val"

    if a.memory:
        sys.exit(evaluate(a.task, a.memory, metric, logs_dir, results_dir, a.workers))
    if a.frontier:
        print_frontier(logs_dir, results_dir, a.task, a.model, metric)
        return
    if a.results:
        base = results_dir if a.test else logs_dir
        print_results(load_results(base, "test.json" if a.test else "val.json"), metric)
        return
    ap.error("one of --memory, --frontier, --results")


if __name__ == "__main__":
    main()
