"""Run on the cluster with the test order as JSON on stdin: mean score per test task over its fixed problems,
matched by real instance id in the raw episode records (exports hash the ids). Used by official_test.py."""
import glob
import json
import os
import statistics
import sys

o = json.load(sys.stdin)
base = os.path.expanduser("~/results/mb/evo_b/official_test")
res = {}
for task, items in o.items():
    want = {(c, str(i)) for c, i in items}
    sc = []
    for f in glob.glob(base + "/" + task + "/*/out/**/*.json", recursive=True):
        if os.path.basename(f) in ("summary.json", "rejected.json"):
            continue
        e = json.load(open(f))
        if isinstance(e, dict) and "score" in e and "infra" not in e and \
                (e["bench"] + ":" + e["subtask"], str(e["instance"])) in want:
            sc.append(e["score"])
    if sc:
        res[task] = {"episodes": len(sc), "of": len(items), "score": round(statistics.mean(sc), 4)}
print(json.dumps(res))
