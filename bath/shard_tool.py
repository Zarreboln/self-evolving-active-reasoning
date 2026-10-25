"""Split an AR-Bench test set into contiguous shards, and merge the shards' logs back.

AR-Bench's evaluators are a plain `for i in tqdm(range(len(logs), len(dataset)))` loop with no
concurrency anywhere in the package, so one evaluator keeps exactly one request in flight and a
7B model on an H100 sits nearly idle. Both `--data_path` and `--output_path` are flags, so the
benchmark can be run concurrently without touching its code: give each worker a contiguous slice
of the test set and its own log file, then concatenate.

Contiguous slices are what make the merge exact. Shard j holds instances [offset_j, offset_j+n_j),
each record's `idx` is local to the slice it was run on, so the merge rewrites it to
`offset_j + position`, and concatenating the shards in order reproduces the single-process file
record for record.

    python shard_tool.py split data/sp/test.json <workdir> 16
    python shard_tool.py merge <workdir> results/sp_few_shot.json
"""

import json
import os
import sys


def offsets(total, n):
    """Contiguous block boundaries: the first `total % n` blocks take one extra instance."""
    q, r = divmod(total, n)
    out, start = [], 0
    for j in range(n):
        size = q + (1 if j < r else 0)
        if size:
            out.append((start, size))
        start += size
    return out


def split(data_path, workdir, n):
    data = json.load(open(data_path))
    os.makedirs(workdir, exist_ok=True)
    blocks = offsets(len(data), n)
    for j, (start, size) in enumerate(blocks):
        with open(os.path.join(workdir, f"shard_{j:02d}.json"), "w") as f:
            json.dump(data[start:start + size], f)
    json.dump({"source": data_path, "total": len(data),
               "blocks": [{"shard": j, "offset": s, "size": k} for j, (s, k) in enumerate(blocks)]},
              open(os.path.join(workdir, "blocks.json"), "w"), indent=2)
    print(len(blocks))


def merge(workdir, out_path):
    """Write the merged log only if every shard is complete; report what is missing otherwise."""
    meta = json.load(open(os.path.join(workdir, "blocks.json")))
    merged, missing = [], []
    for b in meta["blocks"]:
        p = os.path.join(workdir, f"out_{b['shard']:02d}.json")
        try:
            rows = json.load(open(p))
        except Exception:
            rows = []
        if len(rows) < b["size"]:
            missing.append(f"shard {b['shard']:02d}: {len(rows)}/{b['size']}")
            continue
        for pos, rec in enumerate(rows[:b["size"]]):
            if isinstance(rec, dict):
                rec["idx"] = b["offset"] + pos
            merged.append(rec)
    if missing:
        print("INCOMPLETE " + "; ".join(missing))
        return 1
    if len(merged) != meta["total"]:
        print(f"INCOMPLETE merged {len(merged)} != {meta['total']}")
        return 1
    tmp = out_path + ".tmp"
    json.dump(merged, open(tmp, "w"), indent=4)
    os.replace(tmp, out_path)
    print(f"MERGED {len(merged)} -> {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "split":
        split(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "merge":
        sys.exit(merge(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
