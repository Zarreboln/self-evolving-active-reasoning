"""Run AR-Bench's OWN SP (Situation Puzzle / 'turtle soup') evaluator,
unmodified, against a local vLLM, sharded across subprocesses.

Same reasoning as run_arbench_gn.py: `_run_traditional_evaluation` in
sp_evaluator.py is a plain Python for-loop over the dataset, one instance at
a time -- no concurrency. The fix adds no code to their side: split the
dataset into shards, run N copies of the unmodified evaluator on disjoint
slices against one vLLM (which already serves many concurrent requests via
continuous batching), merge the per-shard logs back into dataset order.
Concurrency across independent episodes changes wall-clock only, not the
rollout or the scoring.

sp_evaluator.py's `main()` signature differs from gn_evaluator's: it takes
separate policy_model/response_model (both point at the same served model
here) and policy_*/response_* temperature/top_p pairs instead of one shared
pair, plus branch/simulate_depth (unused by zero_shot/few_shot/few_shot_inst,
required by tot/uot).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def shard(data, n):
    out = [[] for _ in range(n)]
    for i, x in enumerate(data):
        out[i % n].append((i, x))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arbench", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--methods", default="zero_shot")
    p.add_argument("--n-instances", type=int, default=100)
    p.add_argument("--shards", type=int, default=16)
    p.add_argument("--max-turn", type=int, default=25)
    p.add_argument("--branch", type=int, default=3)
    p.add_argument("--simulate-depth", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--workdir", required=True)
    a = p.parse_args()

    arbench = Path(a.arbench).resolve()
    work = Path(a.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    full = json.load(open(arbench / "data/sp/test.json"))[: a.n_instances]
    parts = shard(full, a.shards)

    env = dict(os.environ)
    env.update(POLICY_API_KEY="EMPTY", POLICY_BASE_URL=a.base_url,
               RESPONSE_API_KEY="EMPTY", RESPONSE_BASE_URL=a.base_url,
               PYTHONPATH=str(arbench))

    for method in a.methods.split(","):
        t0 = time.time()
        merged_path = work / f"sp_{method}.json"
        if merged_path.exists():
            print(f"[{method}] already done, skipping", flush=True)
            continue
        procs, shard_files = [], []
        for si, part in enumerate(parts):
            if not part:
                continue
            dpath = work / f"_data_{method}_{si}.json"
            opath = work / f"_log_{method}_{si}.json"
            json.dump([x for _, x in part], open(dpath, "w"))
            shard_files.append((opath, [i for i, _ in part]))
            cmd = [sys.executable, "-m", "arbench.reasoner.sp.sp_evaluator",
                   f"--method={method}",
                   f"--data_path={dpath}", f"--output_path={opath}",
                   f"--policy_model={a.model}", f"--response_model={a.model}",
                   f"--branch={a.branch}", f"--max_turn={a.max_turn}",
                   f"--policy_temperature={a.temperature}", f"--policy_top_p={a.top_p}",
                   f"--response_temperature={a.temperature}", f"--response_top_p={a.top_p}",
                   f"--simulate_depth={a.simulate_depth}"]
            procs.append(subprocess.Popen(
                cmd, cwd=arbench, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        bad = 0
        for pr in procs:
            _, err = pr.communicate()
            if pr.returncode != 0:
                bad += 1
                print(f"[{method}] shard rc={pr.returncode}: "
                      f"{(err or b'').decode()[-400:]}", flush=True)
        merged = []
        for opath, orig_idx in shard_files:
            if not opath.exists():
                continue
            for rec, gi in zip(json.load(open(opath)), orig_idx):
                rec["global_idx"] = gi
                merged.append(rec)
        merged.sort(key=lambda r: r["global_idx"])
        json.dump(merged, open(merged_path, "w"))
        if not merged:
            merged_path.unlink(missing_ok=True)
            sys.exit(f"ABORT: {method} produced no episodes ({bad} bad shards)")
        f1c = sum(r["f1_score_char"] for r in merged) / len(merged)
        f1w = sum(r["f1_score_word"] for r in merged) / len(merged)
        print(f"[{method}] {len(merged)} episodes, f1_char={f1c:.3f} "
              f"f1_word={f1w:.3f}, {bad} bad shards, {time.time()-t0:.0f}s "
              f"-> {merged_path}", flush=True)
        for f in work.glob(f"_*_{method}_*.json"):
            f.unlink()


if __name__ == "__main__":
    main()
