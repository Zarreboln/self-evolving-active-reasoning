"""Separate what the questions were worth from what the harness did with them.

AREW (ICML 2026) defines AS informativeness as the belief quality a *Bayesian* updater would
reach when driven by the harness's own action-selection kernel, and BT as how much of that
progress the agent actually absorbs. Both are stated over an oracle belief, which on Guess Number
is exact and free: the feedback is deterministic, so the set of targets consistent with a
transcript is computable from the transcript alone -- no label, no model call.

So, per episode, with H_t the consistent set and a uniform prior over it,

    Psi_T  := 1 / |H_T|             the posterior mass on the truth after the last question
    yield  := mean_e Psi_T(e)       what a perfect reader of this harness's own transcript scores
    uptake := accuracy / yield      the share of that mass the harness turned into a right answer

and accuracy = yield x uptake identically. One number for each ability, and they multiply back to
the headline. Run it on a seed and on an evolved harness and the two columns say which one moved.

The names are ours, deliberately. `yield` is the on-policy counterpart of AREW's I_AS -- the same
potential Psi, but conditioned on the questions the harness actually asked rather than on actions
re-drawn under an oracle belief, so a harness whose belief tracking is broken asks worse questions
and pays for it here. `uptake` is not their C_BT: that one reads the agent's own belief every turn
and sums the positive increments, which needs a belief readout -- and requiring a harness to
maintain and report a belief vector installs the very scaffold we are trying to measure. AREW also
calls its positive increments "absorbed", so that word is spoken for.

The turn-level columns are T3's (ICLR 2026) criterion on the same transcripts: a turn is dead
when d(H_{t-1}, H_t) = |H_{t-1}| - |H_t| <= 0, and T3 truncates GN at the first one.

    python meta/tools/as_bt.py ~/results/mh_runs/gn_*_test.json

Needs `history` in the rows, which benchmark.py now records.
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

ALL = ["".join(p) for p in permutations("0123456789", 4)]   # 5040, AR-Bench's target space


def feedback(target, guess):
    """AR-Bench's compare_guess, inlined so this runs without the AR-Bench checkout."""
    a = sum(1 for i in range(4) if target[i] == guess[i])
    b = sum(min(target.count(d), guess.count(d)) for d in set(guess)) - a
    return a, b


def episode(history):
    """Returns (Psi_T, dead turns, longest dead run, turns)."""
    H = ALL
    sizes = [len(H)]
    for g, a, b in history:
        H = [s for s in H if feedback(s, g) == (a, b)]
        sizes.append(len(H))
    dead = run = best = 0
    for i in range(1, len(sizes)):
        if sizes[i] - sizes[i - 1] >= 0:        # d(H_{t-1}, H_t) <= 0
            dead += 1
            run += 1
            best = max(best, run)
        else:
            run = 0
    # The consistent set always contains the target, so it is never empty and Psi is well defined.
    return (1.0 / len(H) if H else 0.0), dead, best, len(history)


def report(path):
    rows = json.load(open(path))
    rows = [r for r in rows if r.get("history")]
    if not rows:
        print("  %-34s no transcripts recorded" % Path(path).name)
        return
    n = len(rows)
    acc = 100.0 * sum(1 for r in rows if r.get("correctness")) / n
    stats = [episode([tuple(h) for h in r["history"]]) for r in rows]
    AS = 100.0 * sum(s[0] for s in stats) / n
    turns = sum(s[3] for s in stats)
    print("  %-34s n=%-4d acc=%5.1f  yield=%5.1f  uptake=%5.2f  dead turns=%4.1f%%  "
          "longest dead run=%4.1f  pinned=%4.1f%%  pinned&wrong=%4.1f%%"
          % (Path(path).stem, n, acc, AS, (acc / AS if AS else float("nan")),
             100.0 * sum(s[1] for s in stats) / max(turns, 1),
             sum(s[2] for s in stats) / n,
             100.0 * sum(1 for s in stats if s[0] == 1.0) / n,
             100.0 * sum(1 for s, r in zip(stats, rows)
                         if s[0] == 1.0 and not r.get("correctness")) / n))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
