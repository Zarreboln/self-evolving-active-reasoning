"""Shard our AR-Bench splits into AR-Bench's own data format, so its evaluators run unchanged.

AR-Bench's evaluators read a JSON list and play it one item at a time, resuming from the log they
write. Running K shards as K processes is the only change to how they are run; nothing in AR-Bench is
edited. Items are the same ones our adapters use (search/selection from ~/arbench_mh_splits, test from
~/sealed).

    python -m mb.official.prep_arbench --out DIR --split test --shards 10
"""
from __future__ import annotations

import argparse
import json
import os


def items(task, split):
    root = "~/sealed" if split == "test" else "~/arbench_mh_splits"
    return json.load(open(os.path.expanduser(f"{root}/{task}/{split}.json")))["instances"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default="gn,dc")
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--n", type=int, default=0, help="first n items only (0 = all)")
    a = ap.parse_args()
    for task in a.tasks.split(","):
        xs = items(task, a.split)
        xs = xs[:a.n] if a.n else xs
        d = os.path.join(a.out, task, a.split)
        os.makedirs(d, exist_ok=True)
        for k in range(a.shards):
            json.dump(xs[k::a.shards], open(os.path.join(d, f"shard{k}.json"), "w"))
        print(task, a.split, len(xs), "items ->", a.shards, "shards")


if __name__ == "__main__":
    main()
