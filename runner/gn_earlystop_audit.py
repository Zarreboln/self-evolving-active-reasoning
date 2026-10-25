"""Measure what AR-Bench's Guess Number metric costs the few-shot methods.

The traditional methods (zero_shot / few_shot / few_shot_inst) never break on success: the model
keeps guessing for all 25 turns and `correctness` reads `guess_list[-1]`, the last guess alone.
UoT and Proactive CoT both `break` the moment the feedback is [4, 0], and UoT scores
`any(g[1] == 4 for g in guess_list)`. So the two families are not scored on the same event. This
prints, per finished run, the official number against the number of episodes that hit the answer
at any point, and when those hits happened.
"""
import json
import sys

print("%-16s %14s %14s %10s" % ("method", "official(last)", "any-hit", "median turn"))
for m in sys.argv[1:]:
    d = json.load(open(f"/home/tmp_l/results/gn_{m}.json"))
    last = sum(1 for r in d if (r.get("guess_list") or [[None, -1]])[-1][1] == 4)
    turns = []
    for r in d:
        gl = r.get("guess_list") or []
        idx = next((i for i, g in enumerate(gl) if g[1] == 4), None)
        if idx is not None:
            turns.append(idx + 1)
    turns.sort()
    med = turns[len(turns) // 2] if turns else None
    print("%-16s %12d/100 %12d/100 %10s" % (m, last, len(turns), med))
    if turns:
        print("%-16s hit turns: %s" % ("", turns))
