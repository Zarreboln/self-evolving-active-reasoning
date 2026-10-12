"""Run AR-Bench's OWN GuessNumbers evaluator, unmodified, against a local vLLM.

Why not my own loop: AR-Bench's `_run_traditional_evaluation` does not stop
when the agent guesses right. It always burns all max_turn turns, then asks
for a final guess, and scores `correctness = guess_list[-1][1] == 4` -- only
the last guess counts. My runner returned at the first correct guess and
counted that as solved. Those are different quantities, so my numbers would
not have been comparable to AR-Bench's or T3's. Their protocol wins.

Their loop is serial, which is the 15-hour problem again. The fix adds no code
to their side: split the dataset into shards, run N copies of their unmodified
evaluator on disjoint slices against one vLLM, merge the logs. Concurrency
across independent episodes changes throughput, not results.

Scoring with the Stage 0 instrument happens afterwards, offline, from the
logged (true_number, guess_list) -- see score_arbench_log.py. The instrument
never touches the rollout.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

METHODS = {           # AR-Bench's name -> what we have been calling it
    "zero_shot": "bare",
    "few_shot_inst": "taught_incremental",
    "proactive_cot": "cot",
}


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
    p.add_argument("--methods", default="zero_shot,few_shot_inst,proactive_cot")
    p.add_argument("--n-instances", type=int, default=100)
    p.add_argument("--shards", type=int, default=16)
    p.add_argument("--max-turn", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--workdir", required=True)
    a = p.parse_args()

    arbench = Path(a.arbench).resolve()
    work = Path(a.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    full = json.load(open(arbench / "data/gn/test.json"))[: a.n_instances]
    parts = shard(full, a.shards)

    env = dict(os.environ)
    env.update(POLICY_API_KEY="EMPTY", POLICY_BASE_URL=a.base_url,
               RESPONSE_API_KEY="EMPTY", RESPONSE_BASE_URL=a.base_url,
               PYTHONPATH=str(arbench))

    for method in a.methods.split(","):
        t0 = time.time()
        merged_path = work / f"gn_{method}.json"
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
            cmd = [sys.executable, "-m", "arbench.reasoner.gn.gn_evaluator",
                   f"--model={a.model}", f"--method={method}",
                   f"--data_path={dpath}", f"--output_path={opath}",
                   f"--max_turn={a.max_turn}", f"--temperature={a.temperature}",
                   f"--top_p={a.top_p}", "--simulate_depth=1"]
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
        ok = sum(r.get("correctness") for r in merged)
        print(f"[{method} = {METHODS.get(method, method)}] "
              f"{len(merged)} episodes, correct {ok}/{len(merged)}, "
              f"{bad} bad shards, {time.time()-t0:.0f}s -> {merged_path}",
              flush=True)
        # Silent success on an empty result is the worst failure mode: the
        # scorer happily prints a table of n/a and the job exits 0.
        if not merged:
            merged_path.unlink(missing_ok=True)
            sys.exit(f"ABORT: {method} produced no episodes ({bad} bad shards)")
        for f in work.glob(f"_*_{method}_*.json"):
            f.unlink()


if __name__ == "__main__":
    main()
