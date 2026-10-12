"""Apply the Stage 0 instrument offline to AR-Bench's own GuessNumbers logs.

The instrument needs only (secret, ordered guesses); AR-Bench logs both as
`true_number` and `guess_list`. Nothing here runs a model or alters a rollout,
so AR-Bench's protocol and its `correctness` stay exactly as they shipped.

Reference rows measured on synthetic policies (stage0/survey_families.py):

    policy          solved(100)  turns   ex-ante eff   zero-info
    oracle             100         4.48     1.000         0.0%
    greedy-4sample     100         5.31     0.893         0.4%
    random             100         6.42     0.734         9.2%
    random-fresh       100         7.01     0.691        14.1%
    sequential           4        24.61     0.157        71.1%

Columns: `correct` is AR-Bench's own metric (final guess only). `hit@any` is
how often the secret was ever guessed -- the gap between them is agents that
found the answer and then lost it, which is its own failure mode.
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage0"))
from gn_oracle import GNGame, build_feedback_matrix, build_space  # noqa: E402


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def score_episode(game, full_avail, secret, guesses):
    alive = np.arange(len(game.cands))
    rows, prev = [], None
    for t, g in enumerate(guesses, 1):
        if g not in game.index:                       # parse failure, logged ""
            rows.append({"turn": t, "illegal": True})
            prev = None
            continue
        gi = game.index[g]
        avail = full_avail if len(alive) == len(game.cands) \
            else game.available_bits(alive)[0]
        chosen = game.expected_bits(gi, alive)
        code = game.code(g, secret)
        after = alive[game.fb[gi, alive] == code]
        rows.append({
            "turn": t, "guess": g,
            "n_before": int(len(alive)), "n_after": int(len(after)),
            "efficiency": None if avail <= 0 else chosen / avail,
            "bits_chosen": chosen,
            "dead_guess": gi not in set(alive.tolist()),
            "anchor_dist": None if prev is None else hamming(prev, g),
            "zero_info": chosen < 1e-9,
            "solved": g == secret,
        })
        prev = g
        alive = after if len(after) else alive
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()

    cands, digits, masks = build_space()
    game = GNGame(build_feedback_matrix(digits, masks), cands)
    full_avail = game.available_bits(np.arange(len(cands)))[0]

    hdr = (f"{'log':<22}{'correct':>9}{'hit@any':>9}{'1st hit':>9}"
           f"{'eff':>8}{'dead%':>8}{'ham1%':>8}{'zero%':>8}{'illeg%':>8}"
           f"{'out_tok/ep':>11}")
    print(hdr)
    print("-" * len(hdr))
    for path in a.logs:
        recs = json.load(open(path))
        corr = hits = 0
        toks = []
        first, eff, dead, ham1, zero, illeg = [], [], [], [], [], []
        for r in recs:
            # _run_traditional_evaluation logs true_number as "8362";
            # _run_proactive_cot_evaluation logs it as [8, 3, 6, 2].
            tn = r["true_number"]
            secret = tn if isinstance(tn, str) else "".join(map(str, tn))
            guesses = [g[0] for g in r["guess_list"]]
            corr += bool(r.get("correctness"))
            toks.append(r.get("output_token", 0))
            rows = score_episode(game, full_avail, secret, guesses)
            hit = [x["turn"] for x in rows if x.get("solved")]
            if hit:
                hits += 1
                first.append(hit[0])
            for x in rows:
                illeg.append(bool(x.get("illegal")))
                if x.get("illegal"):
                    continue
                # AR-Bench never stops early: once the feasible set is a
                # singleton every remaining turn is necessarily worth zero
                # bits, which swamped the statistic (the oracle scored 82.7%
                # "zero-info" purely from post-solution padding). Only turns
                # where information was still available count.
                if x["n_before"] <= 1:
                    continue
                if x["efficiency"] is not None:
                    eff.append(x["efficiency"])
                dead.append(x["dead_guess"])
                zero.append(x["zero_info"])
                if x["anchor_dist"] is not None:
                    ham1.append(x["anchor_dist"] == 1)
        n = len(recs)
        m = lambda v, pct=False: ("n/a" if not v else
                                  f"{100*st.mean(v):.1f}%" if pct
                                  else f"{st.mean(v):.3f}")
        print(f"{Path(path).stem:<22}{corr}/{n:<7}{hits}/{n:<7}"
              f"{(st.mean(first) if first else float('nan')):>9.2f}"
              f"{m(eff):>8}{m(dead, 1):>8}{m(ham1, 1):>8}"
              f"{m(zero, 1):>8}{m(illeg, 1):>8}"
              f"{(st.mean(toks) if toks else 0):>11.0f}")
    print("\nreference -- oracle eff 1.000 zero 0.0% | random .734 / 9.2% | "
          "sequential .157 / 71.1%")


if __name__ == "__main__":
    main()
