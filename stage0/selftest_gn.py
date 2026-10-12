"""Stage 0 self-test: agreement with AR-Bench's own feedback function,
oracle K distribution (the solvability certificate), and the random-consistent
null policy. No model calls."""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "AR-Bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arbench.utils.utils_gn import compare_guess  # noqa: E402
from gn_oracle import GNGame, build_feedback_matrix, build_space  # noqa: E402


def main():
    t0 = time.time()
    cands, digits, masks = build_space()
    print(f"|H| = {len(cands)}  (expected 5040)")
    fb = build_feedback_matrix(digits, masks)
    print(f"feedback matrix {fb.shape} {fb.nbytes/1e6:.1f} MB  "
          f"built in {time.time()-t0:.1f}s")

    game = GNGame(fb, cands)

    # 1. agreement with AR-Bench's compare_guess
    rng = random.Random(0)
    bad = 0
    for _ in range(20000):
        g, s = rng.choice(cands), rng.choice(cands)
        sp, dp, _ = compare_guess([int(c) for c in s], [int(c) for c in g])
        if game.code(g, s) != sp * 5 + dp:
            bad += 1
    print(f"agreement with arbench.compare_guess: {20000-bad}/20000 "
          f"({'OK' if bad == 0 else 'MISMATCH'})")

    # 2. test secrets all inside the space
    secrets = json.load(open(ROOT / "AR-Bench/data/gn/test.json"))
    missing = [s for s in secrets if s not in game.index]
    print(f"test secrets in H: {len(secrets)-len(missing)}/{len(secrets)}")
    lead0 = sum(1 for s in secrets if s[0] == "0")
    print(f"test secrets with leading zero: {lead0} "
          f"(AR-Bench's own generator forbids these -> space must be 5040, not 4536)")

    # 3. greedy max-entropy oracle -> the certificate K
    t1 = time.time()
    ks = [game.solve_greedy(s) for s in secrets]
    ks = np.array([k if k else 99 for k in ks])
    print(f"\noracle (max-entropy greedy) over {len(secrets)} secrets "
          f"in {time.time()-t1:.1f}s")
    print(f"  K: mean {ks.mean():.2f}  median {np.median(ks):.0f}  "
          f"min {ks.min()}  max {ks.max()}  failures(>25) {(ks==99).sum()}")
    print(f"  histogram {dict(zip(*[x.tolist() for x in np.unique(ks, return_counts=True)]))}")

    # 4. random-consistent null policy
    t2 = time.time()
    nrng = np.random.default_rng(0)
    rc = np.array([[game.solve_random_consistent(s, nrng) or 99 for s in secrets]
                   for _ in range(3)])
    print(f"\nrandom-consistent null policy, 3 seeds, in {time.time()-t2:.1f}s")
    print(f"  K: mean {rc.mean():.2f}  median {np.median(rc):.0f}  "
          f"max {rc.max()}  failures(>25) {(rc==99).sum()}/{rc.size}")

    # 5. sanity: the oracle's own per-step efficiency should sit at ~1.0
    s = secrets[0]
    alive = np.arange(len(cands))
    effs = []
    for _ in range(12):
        if len(alive) <= 1:
            break
        _, g = game.available_bits(alive)
        guess = cands[g]
        m = game.efficiency(alive, guess, game.code(guess, s))
        alive = m.pop("alive_after")
        effs.append(round(m["efficiency"], 3))
        if guess == s:
            break
    print(f"\noracle per-step efficiency on secret {s}: {effs}")
    print(f"total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
