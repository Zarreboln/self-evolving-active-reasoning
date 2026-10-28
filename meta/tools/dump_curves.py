"""Both abilities, per turn, for every task -- read off the budget sweep and written as one json.

Belief tracking is the same measurement everywhere: end the episode at turn T, take what the
system answers. Information acquisition is not, because only Guess Number has an enumerable
consistent set. T3 (ICLR 2026) handles the same problem the same way -- one task-agnostic
criterion, instantiated per task with whatever the transcript makes observable -- so:

  gn   the oracle's posterior mass on the truth, mean 1/|H_t|. Exact: the feedback is
       deterministic, so H_t follows from the transcript alone and the target is always in it.
  dc   distinct suspects interrogated, out of five. A suspect never heard cannot be ruled in or
       out, and "whom to ask, in what order" is exactly what lean_verify changed.
  sp   turns that got a decisive answer, as a share of the budget. This is T3's own SP proxy: a
       step whose referee says "unknown" taught nothing.

The three are bounded 0-100 and monotone, and that is all they share. They are three different
quantities and the figure labels each panel with its own.

Both columns for turn T come from the SAME run, the one whose budget was T, so a harness that
plans differently under a shorter budget is measured consistently on both axes.

    python meta/tools/dump_curves.py ~/results/belief > curves.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ia_bt import ALL, feedback  # noqa: E402

PREFIXES = ["gn_template_zero_shot", "gn_packed_group_echo",
            "dc_template_zero_shot", "dc_lean_verify",
            "sp_baseline_zero_shot", "sp_atomic_audit"]
BUDGET = 25
DECISIVE = ("yes", "no")


def ia(task, history):
    """The task's own proxy, on one episode's complete transcript. 0-100."""
    if task == "gn":
        H = ALL
        for g, a, b in (tuple(h) for h in history):
            H = [s for s in H if feedback(s, g) == (a, b)]
        return 100.0 / len(H) if H else 0.0
    if task == "dc":
        return 100.0 * len({h[0] for h in history}) / 5.0
    return 100.0 * sum(1 for h in history
                       if str(h[-1]).strip().strip(".").lower() in DECISIVE) / BUDGET


def main(root):
    out = {}
    for pref in PREFIXES:
        task = pref[:2]
        runs = {}
        for f in sorted(glob.glob(os.path.join(root, pref + "_b*.json"))):
            m = re.search(r"_b(\d+)\.json$", f)
            if not m or f.endswith(".scored.json"):
                continue
            rows = json.load(open(f))
            side = f[:-5] + ".scored.json"
            sc = ([float(p["score"]) for p in json.load(open(side))] if os.path.exists(side)
                  else [1.0 if r.get("correctness") else 0.0 for r in rows])
            runs[int(m.group(1))] = (rows, sc)
        if not runs:
            continue
        Ts = sorted(runs)
        n = min(len(v[1]) for v in runs.values())
        rec = {"turns": Ts, "n": n,
               "bt": [100.0 * sum(runs[T][1][:n]) / n for T in Ts]}
        # Per budget, not all-or-nothing: a run written before transcripts were recorded leaves
        # a gap in the curve rather than deleting it.
        rec["ia"] = [(sum(ia(task, r["history"]) for r in runs[T][0][:n]) / n
                      if all("history" in r for r in runs[T][0][:n]) else None) for T in Ts]
        if not any(v is not None for v in rec["ia"]):
            rec["ia"] = None
        out[pref] = rec
    return out


if __name__ == "__main__":
    print(json.dumps(main(os.path.expanduser(sys.argv[1]))))
