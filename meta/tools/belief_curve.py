"""The belief trajectory of harness-plus-model, read without installing a readout.

AREW (ICML 2026) tracks BT as the per-turn progress of the agent's own belief, which it obtains by
instructing the agent to maintain and report a belief vector. We cannot do that: requiring a
harness to keep and report a belief installs the very scaffold the search is supposed to invent.
And probing the frozen model with the raw transcript is worse -- it reads the BARE MODEL, with the
harness's ledger, surviving set or grid stripped off, which is the opposite of the belief we want.

So the belief at turn t is read by ending the episode at turn t and taking the system's answer.
Whatever state the harness keeps is in that answer by construction. Sweeping the budget gives the
trajectory. Two things this costs, both stated rather than assumed away:

  * the harness sees a shorter budget, so it may plan differently. The control is budget = the
    full budget, which must reproduce the ordinary run.
  * the runs are independent samples at temperature 0.7, so an episode that is right at t and
    wrong at t+1 may be sampling noise rather than belief being destroyed. The control is the
    same budget run twice; its disagreement rate is the floor under the drift number.

Paired against it is the oracle trajectory 1/|H_t|, replayed from the transcripts for free. The
gap between the two curves is what belief tracking is losing, turn by turn.

    python meta/tools/belief_curve.py ~/results/belief/gn_template_zero_shot
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ia_bt import ALL, feedback  # noqa: E402


def curves(prefix):
    """prefix is e.g. .../gn_template_zero_shot; files are <prefix>_b<T>.json (+ _b25r2)."""
    runs = {}
    for f in sorted(glob.glob(prefix + "_b*.json")):
        m = re.search(r"_b(\d+)(r2)?\.json$", f)
        if not m:
            continue
        runs[(int(m.group(1)), bool(m.group(2)))] = json.load(open(f))
    if not runs:
        print("  no budget sweep under %s" % prefix)
        return

    Ts = sorted({t for t, r2 in runs if not r2})
    n = min(len(v) for v in runs.values())
    print("  %s   n=%d  budgets=%s" % (Path(prefix).name, n, Ts))

    # The system's belief: right or wrong when the episode is cut at T.
    sysP = {T: [bool(r.get("correctness")) for r in runs[(T, False)][:n]] for T in Ts}

    # The oracle's belief on the same episodes, replayed from the longest run's transcripts.
    full = runs[(max(Ts), False)]
    orc = {}
    for T in Ts:
        vals = []
        for r in full[:n]:
            H = ALL
            for g, a, b in [tuple(h) for h in r.get("history", [])][:T]:
                H = [s for s in H if feedback(s, g) == (a, b)]
            vals.append(1.0 / len(H) if H else 0.0)
        orc[T] = sum(vals) / len(vals)

    print("      turn   oracle 1/|H_t|   system answer   gap")
    for T in Ts:
        s = 100.0 * sum(sysP[T]) / n
        print("      %4d   %12.1f   %13.1f   %+5.1f" % (T, 100.0 * orc[T], s, s - 100.0 * orc[T]))

    # Per-episode movement between consecutive budgets, AREW's one-sided split.
    up = down = 0
    lost = 0
    for e in range(n):
        seq = [sysP[T][e] for T in Ts]
        up += sum(1 for i in range(1, len(seq)) if seq[i] and not seq[i - 1])
        down += sum(1 for i in range(1, len(seq)) if seq[i - 1] and not seq[i])
        if any(seq[:-1]) and not seq[-1]:
            lost += 1
    print("      gained %d  lost %d   episodes right at some turn and wrong at the end: %.1f%%"
          % (up, down, 100.0 * lost / n))

    ctl = [(max(Ts), False), (max(Ts), True)]
    if all(c in runs for c in ctl):
        a, b = (runs[c] for c in ctl)
        dis = sum(1 for x, y in zip(a[:n], b[:n])
                  if bool(x.get("correctness")) != bool(y.get("correctness")))
        print("      noise floor: the same budget run twice disagrees on %.1f%% of episodes"
              % (100.0 * dis / n))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        curves(p.rstrip("/"))
