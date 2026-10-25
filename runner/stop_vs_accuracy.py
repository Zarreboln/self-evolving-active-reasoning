"""Accuracy against the turn the reasoner chose to commit on.

This is observational, not a controlled contrast: the model chose when to stop, so an early commit
may mean an easy instance as much as a confident one. It answers the narrower question of whether
the model's own sense of "I know enough" tracks being right.
"""
import json
import os
from collections import defaultdict

RUNS = [("md_zero_shot_balar", "iCraft-MD zero_shot (10 turns, stop allowed)"),
        ("md_few_shot_balar", "iCraft-MD few_shot (10 turns, stop allowed)"),
        ("md_few_shot_inst_balar", "iCraft-MD few_shot_inst (10 turns, stop allowed)")]

pool = defaultdict(lambda: [0, 0])
for stem, label in RUNS:
    p = f"/home/tmp_l/results/{stem}.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    by = defaultdict(lambda: [0, 0])
    for r in d:
        k = r.get("round", 0)
        by[k][1] += 1
        pool[k][1] += 1
        if r.get("correctness"):
            by[k][0] += 1
            pool[k][0] += 1
    print(f"\n{label}")
    for k in sorted(by):
        ok, n = by[k]
        print(f"   committed after {k:2} questions: {ok:3}/{n:3}  ({100*ok/n:5.1f}%)")

print("\nall three runs pooled (420 episodes)")
early = [0, 0]
late = [0, 0]
for k in sorted(pool):
    ok, n = pool[k]
    print(f"   committed after {k:2} questions: {ok:3}/{n:3}  ({100*ok/n:5.1f}%)")
    tgt = early if k <= 7 else late
    tgt[0] += ok
    tgt[1] += n
for name, (ok, n) in (("committed by turn 7 (genuinely early)", early),
                      ("committed at turn 8-10 (used the budget)", late)):
    if n:
        print(f"   {name:42} {ok:3}/{n:3}  ({100*ok/n:5.1f}%)")
