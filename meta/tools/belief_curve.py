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
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ia_bt import ALL, feedback  # noqa: E402


def scores(path, rows):
    """One number per episode. Guess Number and Detective Cases score 0/1 in the run file;
    Situation Puzzles does not -- its episodes are graded against the gold text by the equivalence
    judge, so score_val.py --per-episode writes them alongside, and that file is read instead."""
    side = path[:-5] + ".scored.json"
    if os.path.exists(side):
        per = json.load(open(side))
        return [float(p["score"]) for p in per][:len(rows)]
    return [1.0 if r.get("correctness") else 0.0 for r in rows]


def curves(prefix):
    """prefix is e.g. .../gn_template_zero_shot; files are <prefix>_b<T>.json (+ _b25r2)."""
    runs, sc = {}, {}
    for f in sorted(glob.glob(prefix + "_b*.json")):
        m = re.search(r"_b(\d+)(r2)?\.json$", f)
        if not m or f.endswith(".scored.json"):
            continue
        k = (int(m.group(1)), bool(m.group(2)))
        runs[k] = json.load(open(f))
        sc[k] = scores(f, runs[k])
    if not runs:
        print("  no budget sweep under %s" % prefix)
        return

    Ts = sorted({t for t, r2 in runs if not r2})
    n = min(len(v) for v in runs.values())
    print("  %s   n=%d  budgets=%s" % (Path(prefix).name, n, Ts))

    # The system's belief: how it scores when the episode is cut at T. 0/1 on gn and dc, graded
    # on sp, where a partial answer is worth half.
    sysP = {T: sc[(T, False)][:n] for T in Ts}

    # The oracle's belief on the same episodes, replayed from the longest run's transcripts --
    # only where a consistent set is computable at all. Detective Cases logs (name, question,
    # free-text reply) and Situation Puzzles a judge's prose, so neither has one, and replaying
    # them would quietly filter every candidate away and print a confident zero.
    full = runs[(max(Ts), False)]

    def is_gn(h):
        return (len(h) == 3 and isinstance(h[0], str) and h[0].isdigit() and len(h[0]) == 4
                and all(isinstance(x, int) for x in h[1:]))

    hist = [[tuple(h) for h in r.get("history", [])] for r in full[:n]]
    replayable = any(hist) and all(is_gn(h) for ep in hist for h in ep)

    orc = {}
    if replayable:
        for T in Ts:
            vals = []
            for ep in hist:
                H = ALL
                for g, a, b in ep[:T]:
                    H = [s for s in H if feedback(s, g) == (a, b)]
                vals.append(1.0 / len(H) if H else 0.0)
            orc[T] = sum(vals) / len(vals)

    # Both abilities are process quantities, as AREW states them: I_AS telescopes a sum of
    # one-step oracle gains, C_BT sums the agent's own POSITIVE one-step gains. So the columns
    # that matter are the increments between budget points, not the levels.
    #
    #   IA_t  = Psi_oracle(t) - Psi_oracle(t-1)   what this stretch of questions was worth
    #   BT_t  = Psi_system(t) - Psi_system(t-1)   what the system actually took from it
    #
    # and the episode-level gained/lost counts below are the same split one episode at a time.
    print("      turn   oracle Psi   IA (gain)     system Psi   BT (gain)")
    po = ps = 0.0
    for T in Ts:
        sv = 100.0 * sum(sysP[T]) / n
        ov = 100.0 * orc[T] if replayable else None
        if replayable:
            print("      %4d   %10.1f   %+9.1f   %12.1f   %+9.1f" % (T, ov, ov - po, sv, sv - ps))
            po = ov
        else:
            print("      %4d   %10s   %9s   %12.1f   %+9.1f" % (T, "n/a", "n/a", sv, sv - ps))
        ps = sv
    if replayable:
        print("      ---- over the whole episode:  IA +%.1f   BT +%.1f" % (po, ps))
    else:
        print("      ---- over the whole episode:  IA n/a    BT %+.1f" % ps)

    # Per-episode movement between consecutive budgets, AREW's one-sided split.
    up = down = 0
    lost = 0
    for e in range(n):
        seq = [sysP[T][e] for T in Ts]
        up += sum(1 for i in range(1, len(seq)) if seq[i] > seq[i - 1])
        down += sum(1 for i in range(1, len(seq)) if seq[i] < seq[i - 1])
        if max(seq[:-1]) > seq[-1]:
            lost += 1
    # The one-sided split, per episode: AREW's absorbed progress against its self-destructive
    # drift, counted over the transitions between budget points.
    print("      per-episode steps:  absorbed %d   destroyed %d   "
          "(right at some turn, wrong at the end: %.1f%%)" % (up, down, 100.0 * lost / n))

    ctl = [(max(Ts), False), (max(Ts), True)]
    if all(c in runs for c in ctl):
        sa, sb = sc[ctl[0]][:n], sc[ctl[1]][:n]
        dis = sum(1 for x, y in zip(sa, sb) if x != y)
        print("      noise floor: the same budget run twice disagrees on %.1f%% of episodes"
              % (100.0 * dis / n))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        curves(p.rstrip("/"))
