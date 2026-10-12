"""Label-free proxy signals of one episode, computed by the controller from PUBLIC information
(the inventory and the answers the environment gave). None of them needs the truth table.

    health      the harness returned a parsable 24-bit guess and crashed nowhere
    n_queries   queries actually spent (<= 18)
    uniq_end    the feasible set implied by the answers has exactly one element
    n_alive_end size of that feasible set
    zero_info   share of queries whose answer was already determined by earlier answers
    repeat      share of queries identical to an earlier query
    oracle_agree share of queries equal to the max-entropy query at that moment
    eff         mean ex-ante efficiency over queries made while >1 hypothesis remained
These are evidence for the proposer and the judge, not a reward: a harness can game any one of
them (e.g. uniq_end by letting the harness choose the queries -- which the audit forbids).
"""

from __future__ import annotations

import numpy as np

from bridge import game_for, K


def compute(env, guess_text, crashed=False):
    g = game_for(env.inventory)
    alive = np.arange(len(g.H))
    seen, zero, rep, agree, eff = set(), [], [], [], []
    for (c, a, b, x), bit in env.history:
        q = "ABC".index(c) * (1 << K) + (a << 2) + (b << 1) + x
        if len(alive) > 1:
            avail, best = g.available_bits(alive)
            chosen = g.expected_bits(q, alive)
            if avail > 0:
                eff.append(chosen / avail)
            zero.append(chosen < 1e-9)
            agree.append(best is not None and q == best)
            rep.append(q in seen)
        seen.add(q)
        alive = alive[g.H[alive, q] == bit]
    import re
    parsed = bool(re.findall(r"[01]{24}", re.sub(r"\s", "", guess_text or "")))
    return {"health": (not crashed) and parsed, "crashed": crashed, "parsed_guess": parsed,
            "n_queries": len(env.history), "uniq_end": int(len(alive)) == 1, "n_alive_end": int(len(alive)),
            "zero_info": float(np.mean(zero)) if zero else None, "repeat": float(np.mean(rep)) if rep else None,
            "oracle_agree": float(np.mean(agree)) if agree else None, "eff": float(np.mean(eff)) if eff else None}
