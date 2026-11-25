"""Merge the shard logs AR-Bench's evaluators wrote, and report each task's own metric.

GN and DC: the mean of AR-Bench's `correctness` field over all items, exactly as AR-Bench scores them.

    python -m mb.official.merge_arbench --out DIR --split test --tasks gn,dc --methods proactive_cot
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default="gn,dc")
    ap.add_argument("--methods", default="proactive_cot")
    a = ap.parse_args()
    rows = []
    for task in a.tasks.split(","):
        n_items = sum(len(json.load(open(f))) for f in glob.glob(f"{a.out}/data/{task}/{a.split}/shard*.json"))
        for m in a.methods.split(","):
            logs = []
            for f in sorted(glob.glob(f"{a.out}/logs/{task}/{m}/{a.split}/shard*.json")):
                try:
                    logs += json.load(open(f))
                except (json.JSONDecodeError, ValueError):
                    pass
            c = [bool(x.get("correctness")) for x in logs]
            rows.append({"task": task, "method": m, "split": a.split, "items": n_items, "done": len(logs),
                         "correctness": round(sum(c) / len(c), 4) if c else None,
                         "input_tokens": sum(x.get("input_token", 0) for x in logs),
                         "output_tokens": sum(x.get("output_token", 0) for x in logs)})
    json.dump(rows, open(os.path.join(a.out, f"summary_{a.split}.json"), "w"), indent=1)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
