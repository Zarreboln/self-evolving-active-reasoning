"""The Stage 0 instrument, factored out of the per-family oracles.

Any active-reasoning task whose latent state is enumerable and whose answerer
is a program reduces to the same object: a hypothesis set H, a query set Q, and
a feedback matrix F[q, h]. Everything the instrument measures is computed on
the feasible set, never on the agent's internal belief -- the agent's belief is
not observable, and the one intervention that would expose it (forcing an
explicit hypothesis list) is also one of the treatments, so using it to measure
would confound the diagnosis with the cure.

Primary measure: ex-ante efficiency = expected bits of the query the agent
chose / expected bits of the best available query, in [0, 1] and free of
outcome luck. The realized reduction is recorded too but is NOT normalised --
entropy is an average over answers, so a lucky outcome legitimately exceeds it,
and dividing by it would measure the agent's luck rather than its policy.
"""

from math import log2

import numpy as np


class DiscreteGame:
    """H hypotheses x Q queries, F[q, h] = the answer code the answerer returns
    for query q when hypothesis h is the truth."""

    def __init__(self, F, n_codes=None, labels=None):
        self.F = np.asarray(F)
        self.n_q, self.n_h = self.F.shape
        self.n_codes = int(n_codes or self.F.max() + 1)
        self.labels = labels

    # ---- belief bookkeeping -------------------------------------------

    def feasible(self, history, alive=None):
        alive = np.arange(self.n_h) if alive is None else alive
        for q, code in history:
            alive = alive[self.F[q, alive] == code]
        return alive

    def expected_bits(self, q, alive):
        """Entropy of the answer to q over the feasible set, in bits. Exactly
        0 when the history already determines the answer -- the legal,
        non-contradictory, wholly wasted move."""
        n = len(alive)
        if n <= 1:
            return 0.0
        counts = np.bincount(self.F[q, alive], minlength=self.n_codes)
        p = counts[counts > 0] / n
        return float(-(p * np.log2(p)).sum())

    def available_bits(self, alive):
        """Best any legal query can do right now -> (bits, query index)."""
        if len(alive) <= 1:
            return 0.0, None
        sub = self.F[:, alive]
        n = len(alive)
        ent = np.zeros(self.n_q)
        for c in range(self.n_codes):
            p = (sub == c).sum(1) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                ent -= np.where(p > 0, p * np.log2(p, where=p > 0), 0.0)
        q = int(ent.argmax())
        return float(ent[q]), q

    def efficiency(self, alive, q):
        avail, _ = self.available_bits(alive)
        chosen = self.expected_bits(q, alive)
        return (float("nan") if avail <= 0 else chosen / avail), chosen, avail

    # ---- rollout ------------------------------------------------------

    def rollout(self, truth_idx, policy, budget, rng=None, win_requires_unique=True):
        """Run one policy to exhaustion. Returns a per-step record plus whether
        the latent state ended up uniquely identified."""
        alive = np.arange(self.n_h)
        rows, state = [], {}
        for t in range(1, budget + 1):
            if len(alive) <= 1:
                break
            q = policy(self, alive, rng, state)
            eff, chosen, avail = self.efficiency(alive, q)
            code = int(self.F[q, truth_idx])
            after = alive[self.F[q, alive] == code]
            rows.append({"turn": t, "query": int(q), "code": code,
                         "n_before": int(len(alive)), "n_after": int(len(after)),
                         "bits_chosen": chosen, "bits_available": avail,
                         "efficiency": eff, "zero_info": chosen < 1e-9})
            alive = after
        solved = len(alive) == 1 if win_requires_unique else truth_idx in alive
        return rows, solved, len(alive)


# ---------------------------------------------------------------- policies

def p_oracle(g, alive, rng, st):
    return g.available_bits(alive)[1]


def p_random(g, alive, rng, st):
    return int(rng.integers(g.n_q))


def p_random_fresh(g, alive, rng, st):
    """Random but never repeats a query -- isolates 'repeating' from 'choosing
    an entry the prior already pins down'."""
    used = st.setdefault("used", set())
    pool = [q for q in range(g.n_q) if q not in used] or list(range(g.n_q))
    q = int(rng.choice(pool))
    used.add(q)
    return q


def p_sequential(g, alive, rng, st):
    st["i"] = st.get("i", -1) + 1
    return st["i"] % g.n_q


def p_greedy_feasible(g, alive, rng, st):
    """Query that best splits the feasible set but chosen myopically among a
    small random sample -- a bounded-rationality stand-in."""
    pool = rng.choice(g.n_q, size=min(4, g.n_q), replace=False)
    return int(max(pool, key=lambda q: g.expected_bits(q, alive)))


POLICIES = {
    "oracle": p_oracle,
    "greedy-4sample": p_greedy_feasible,
    "random-fresh": p_random_fresh,
    "sequential": p_sequential,
    "random": p_random,
}


def benchmark(games_and_truths, budget, seed=0, policies=None):
    """games_and_truths: iterable of (DiscreteGame, truth_index).
    Returns {policy: dict of aggregate readings}."""
    policies = policies or POLICIES
    out = {}
    for name, pol in policies.items():
        rng = np.random.default_rng(seed)
        solved = n = 0
        effs, zeros, turns = [], 0, []
        steps = 0
        for g, truth in games_and_truths:
            rows, ok, remaining = g.rollout(truth, pol, budget, rng)
            n += 1
            solved += ok
            turns.append(len(rows))
            for r in rows:
                steps += 1
                zeros += r["zero_info"]
                if not np.isnan(r["efficiency"]):
                    effs.append(r["efficiency"])
        out[name] = {
            "solved": solved, "n": n,
            "turns": float(np.mean(turns)) if turns else float("nan"),
            "eff": float(np.mean(effs)) if effs else float("nan"),
            "zero_info_pct": 100.0 * zeros / max(steps, 1),
        }
    return out


def print_table(title, results, extra=""):
    print(f"\n=== {title} ===   {extra}")
    print(f"{'policy':<16}{'solved':>10}{'turns':>8}{'eff':>8}{'zero-info':>11}")
    print("-" * 53)
    for k, v in results.items():
        print(f"{k:<16}{v['solved']}/{v['n']:<8}{v['turns']:>8.2f}"
              f"{v['eff']:>8.3f}{v['zero_info_pct']:>10.1f}%")
