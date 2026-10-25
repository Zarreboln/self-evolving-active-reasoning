"""Count silently-substituted patient turns in the runs made before retries existed.

The pre-retry code replaced a failed patient call with "The record does not say." and scored the
episode anyway, so a rate-limited run and a clean one look the same in the results. This counts
the placeholder, which is the only trace it left.
"""
import glob
import json

PLACEHOLDER = "the record does not say"
for d in sorted(glob.glob("/home/tmp_l/results/shards/md_*/out_*.json")):
    pass
runs = {}
for p in sorted(glob.glob("/home/tmp_l/results/shards/md_*/out_*.json")):
    run = p.split("/")[-2]
    try:
        rows = json.load(open(p))
    except Exception:
        continue
    hit, turns, eps = runs.get(run, (0, 0, 0))
    for r in rows:
        eps += 1
        for m in r.get("record") or []:
            if m.get("role") == "user" and m.get("content", "").lower().startswith("a: "):
                turns += 1
                if PLACEHOLDER in m["content"].lower():
                    hit += 1
    runs[run] = (hit, turns, eps)
for run, (hit, turns, eps) in runs.items():
    print("%-34s episodes %3d  patient turns %5d  placeholder %4d (%.2f%%)"
          % (run, eps, turns, hit, 100.0 * hit / max(turns, 1)))
