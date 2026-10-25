"""Score iCraft-MD runs: BALAR's exact match on the option text, plus parse health and chance."""
import json
import os
import sys

print("%-16s %10s %10s %10s %10s" % ("method", "accuracy", "parsed", "mean turns", "tok out"))
for m in sys.argv[1:]:
    p = f"/home/tmp_l/results/md_{m}.json"
    if not os.path.exists(p):
        print("%-16s %10s" % (m, "-"))
        continue
    d = json.load(open(p))
    acc = sum(1 for r in d if r.get("correctness"))
    parsed = sum(1 for r in d if r.get("parsed"))
    turns = sum(r.get("round", 0) for r in d) / max(len(d), 1)
    out = sum(r.get("output_token", 0) for r in d) / max(len(d), 1)
    print("%-16s %6d/%-3d %9d %10.1f %10.0f" % (m, acc, len(d), parsed, turns, out))
    # which option position the model favours, as a guard against a positional prior
    from collections import Counter
    pos = Counter()
    for r in d:
        opts = r.get("options") or {}
        inv = {v: k for k, v in opts.items()}
        pos[inv.get(r.get("pred"), "?")] += 1
    print("%-16s picked by letter: %s" % ("", dict(sorted(pos.items()))))
