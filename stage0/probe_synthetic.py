"""Zero-cost instrument validation: run synthetic agents with *known*
pathologies through the Stage 0 measure and see what it can distinguish.

If the measure cannot separate these, it will not separate real agents either,
and no API spend can fix that.

  oracle        max-entropy play
  consistent    uniform over the feasible set (the null policy)
  sloppy(q)     with prob q ignore history entirely  -> belief-TRACKING failure
  anchored      only mutate one digit of the running guess, feasible or not
                -> the strategy the AR-Bench prompt literally teaches
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gn_oracle import GNGame, build_feedback_matrix, build_space  # noqa: E402

MAX_TURNS = 25


class Scorer:
    """Stage 0 measure with the turn-1 available-bits value cached."""

    def __init__(self, game):
        self.game = game
        self._full = None

    def available(self, alive):
        if len(alive) == len(self.game.cands):
            if self._full is None:
                self._full = self.game.available_bits(alive)[0]
            return self._full
        return self.game.available_bits(alive)[0]

    def run(self, policy, secret, rng):
        g = self.game
        alive = np.arange(len(g.cands))
        rows = []
        state = {}
        for t in range(1, MAX_TURNS + 1):
            guess = policy(g, alive, rng, state)
            gi = g.index[guess]
            dead = gi not in set(alive.tolist())
            avail = self.available(alive)
            chosen = g.expected_bits(gi, alive)
            code = g.code(guess, secret)
            after = alive[g.fb[gi, alive] == code]
            rows.append({
                "turn": t, "dead": dead,
                "eff": np.nan if avail <= 0 else chosen / avail,
                "n_before": len(alive),
            })
            if guess == secret:
                return t, rows
            # A dead guess carries no admissible evidence for the agent, but the
            # true feedback still constrains the real feasible set.
            alive = after if len(after) else alive
        return None, rows


def p_oracle(g, alive, rng, st):
    return g.cands[int(alive[0])] if len(alive) == 1 else g.cands[g.available_bits(alive)[1]]


def p_consistent(g, alive, rng, st):
    return g.cands[int(rng.choice(alive))]


def make_sloppy(q):
    def p(g, alive, rng, st):
        if rng.random() < q:
            return g.cands[int(rng.integers(len(g.cands)))]
        return g.cands[int(rng.choice(alive))]
    return p


def p_anchored(g, alive, rng, st):
    cur = st.get("cur")
    if cur is None:
        cur = g.cands[int(rng.integers(len(g.cands)))]
        st["cur"] = cur
        return cur
    for _ in range(40):
        pos = int(rng.integers(4))
        d = str(int(rng.integers(10)))
        cand = cur[:pos] + d + cur[pos + 1:]
        if len(set(cand)) == 4:
            st["cur"] = cand
            return cand
    return cur


def main():
    t0 = time.time()
    cands, digits, masks = build_space()
    game = GNGame(build_feedback_matrix(digits, masks), cands)
    scorer = Scorer(game)
    secrets = json.load(open(ROOT / "AR-Bench/data/gn/test.json"))[:40]

    agents = [
        ("oracle", p_oracle),
        ("consistent", p_consistent),
        ("sloppy q=.2", make_sloppy(0.2)),
        ("sloppy q=.5", make_sloppy(0.5)),
        ("sloppy q=.8", make_sloppy(0.8)),
        ("sloppy q=1.", make_sloppy(1.0)),
        ("anchored", p_anchored),
    ]

    print(f"{'agent':<13} {'solved':>7} {'turns':>7} {'dead%':>7} {'eff':>7} "
          f"{'eff|alive':>10}")
    print("-" * 56)
    for name, pol in agents:
        rng = np.random.default_rng(0)
        solved, turns, dead, eff, eff_alive = 0, [], [], [], []
        for s in secrets:
            k, rows = scorer.run(pol, s, rng)
            if k:
                solved += 1
                turns.append(k)
            dead += [r["dead"] for r in rows]
            eff += [r["eff"] for r in rows if not np.isnan(r["eff"])]
            eff_alive += [r["eff"] for r in rows
                          if not r["dead"] and not np.isnan(r["eff"])]
        print(f"{name:<13} {solved}/{len(secrets):<5} "
              f"{np.mean(turns) if turns else float('nan'):>7.2f} "
              f"{100*np.mean(dead):>6.1f}% {np.mean(eff):>7.3f} "
              f"{np.mean(eff_alive) if eff_alive else float('nan'):>10.3f}")
    print(f"\n{time.time()-t0:.1f}s   (eff = ex-ante efficiency, all turns; "
          f"eff|alive = restricted to feasible guesses)")


if __name__ == "__main__":
    main()
