"""Information acquisition by interaction round, for AR-Bench's own methods on our frozen model.

AR-Bench's Tables 8 and 9 report a process score every five rounds for each method; this is the
same picture computed from the transcripts of the baseline runs we did on Qwen2.5-7B-Instruct, so
no run is repeated. The proxy is the one used throughout (meta/tools/dump_curves.py):

  gn  the oracle's posterior mass on the answer, 100/|H_t|, exact from the deterministic feedback
  dc  distinct suspects interrogated, out of five
  sp  turns that got a decisive answer, as a share of the 25-turn budget

The three are not the same quantity and are never averaged; each panel carries its own axis.

    python meta/tools/baseline_rounds.py <logdir> out.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from itertools import permutations

ALL = ["".join(p) for p in permutations("0123456789", 4)]
BUDGET = 25
DECISIVE = ("yes", "no")


def fb(target, guess):
    a = sum(1 for i in range(4) if target[i] == guess[i])
    b = sum(min(target.count(d), guess.count(d)) for d in set(guess)) - a
    return a, b


def gn_curve(ep):
    """|H_t| after each played guess. guess_list is [[guess, in-place, misplaced], ...]."""
    H, out = ALL, []
    for g, a, b in ep["guess_list"]:
        g = str(g).zfill(4)
        H = [s for s in H if fb(s, g) == (int(a), int(b))]
        out.append(100.0 / len(H) if H else 0.0)
    return out


def dc_curve(ep):
    """Two log shapes: the chat transcript, where the suspect is the reply to 'Now choose a
    suspect', and ToT / proactive-CoT, which store one dict a turn."""
    msgs = ep["record"]
    if msgs and isinstance(msgs[0], dict) and "suspect" in msgs[0]:
        seen, out = set(), []
        for t in msgs:
            seen.add(str(t["suspect"]).strip().strip("."))
            out.append(100.0 * len(seen) / 5.0)
        return out
    seen, out = set(), []
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and "choose a suspect" in m.get("content", ""):
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                seen.add(msgs[i + 1]["content"].strip().strip("."))
                out.append(100.0 * len(seen) / 5.0)
    return out


def sp_curve(ep):
    """A turn is decisive when the referee answers Yes or No rather than Unknown. Two log shapes,
    as on dc."""
    msgs = ep["record"]
    if msgs and isinstance(msgs[0], dict) and "answer" in msgs[0]:
        n, out = 0, []
        for t in msgs:
            if str(t["answer"]).strip().strip(".").lower() in DECISIVE:
                n += 1
            out.append(100.0 * n / BUDGET)
        return out
    n, out = 0, []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
            a = msgs[i + 1]["content"].strip().strip(".").lower()
            if a in DECISIVE:
                n += 1
            out.append(100.0 * n / BUDGET)
    return out


CURVE = {"gn": gn_curve, "dc": dc_curve, "sp": sp_curve}


def mean_curve(eps, task):
    """Mean over episodes at each round; an episode that stopped early holds its last value,
    because the set it had pinned down does not grow after it stops asking."""
    per = [CURVE[task](e) for e in eps]
    per = [c for c in per if c]
    if not per:
        return []
    out = []
    for t in range(BUDGET):
        vals = [c[t] if t < len(c) else c[-1] for c in per]
        out.append(sum(vals) / len(vals))
    return out


def main(logdir, dest):
    data = {}
    for f in sorted(glob.glob(os.path.join(logdir, "*.json"))):
        b = os.path.basename(f)[:-5]
        m = re.match(r"(gn|dc|sp)_(.+)$", b)
        if not m:
            continue
        task, method = m.groups()
        try:
            eps = json.load(open(f))
        except Exception:
            continue
        if not isinstance(eps, list) or not eps:
            continue
        c = mean_curve(eps, task)
        if c:
            data.setdefault(task, {})[method] = {"n": len(eps), "curve": c}
            print("  %-4s %-16s n=%-4d  round 5/15/25: %5.1f %5.1f %5.1f"
                  % (task, method, len(eps), c[4], c[14], c[24]))
    json.dump(data, open(dest, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
