"""Prove the shard merge reproduces a sequential run record for record.

Splits the real test set into blocks, fakes each worker's log by taking that block out of a known
sequential output (with `idx` made local, as a worker writing its own file would), merges, and
compares against the original. Also checks the merge refuses to write when a shard is short.
"""
import json, os, shutil, sys

sys.path.insert(0, "/home/tmp_l/code")
import shard_tool

W = "/tmp/shardcheck"
shutil.rmtree(W, ignore_errors=True)
shard_tool.split("data/gn/test.json", W, 16)

orig = json.load(open("/home/tmp_l/results/gn_few_shot.json"))
meta = json.load(open(os.path.join(W, "blocks.json")))
for b in meta["blocks"]:
    rows = [dict(r) for r in orig[b["offset"]:b["offset"] + b["size"]]]
    for pos, r in enumerate(rows):
        r["idx"] = pos
    with open(os.path.join(W, "out_%02d.json" % b["shard"]), "w") as f:
        json.dump(rows, f)

shard_tool.merge(W, os.path.join(W, "merged.json"))
merged = json.load(open(os.path.join(W, "merged.json")))
print("round-trip identical:", merged == orig, "  n =", len(merged))
print("block sizes:", [b["size"] for b in meta["blocks"]])

os.remove(os.path.join(W, "out_07.json"))
rc = shard_tool.merge(W, os.path.join(W, "merged2.json"))
print("refuses when a shard is short:", rc == 1,
      "  wrote merged2:", os.path.exists(os.path.join(W, "merged2.json")))
