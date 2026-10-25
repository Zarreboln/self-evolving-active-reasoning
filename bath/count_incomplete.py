"""Count baseline combos that do not yet hold their 100 episodes.

A missing file counts as incomplete, not as done: the sweep uses this to decide whether another
pass is needed, and a combo whose file was never created is exactly the case a pass must cover.
The method list is BALAR's -- Few-Shot and AR-Bench's second few-shot variant, Proactive CoT, UoT
and ToT. gn_zero_shot is kept as a floor from an earlier run and is not part of this list.
"""
import json
import os

METHODS = ("few_shot", "few_shot_inst", "proactive_cot", "uot", "tot")
n = 0
for t in ("gn", "sp", "dc"):
    for m in METHODS:
        p = f"/home/tmp_l/results/{t}_{m}.json"
        try:
            if len(json.load(open(p))) < 100:
                n += 1
        except Exception:
            n += 1
print(n)
