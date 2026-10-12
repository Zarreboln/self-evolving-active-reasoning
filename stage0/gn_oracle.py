"""
Stage 0 for AR-Bench GuessNumbers: exact feasible set, oracle solver,
and the per-step information-efficiency measure.

No model calls. Pure CPU. This is the diagnostic instrument, the generator's
cheap surrogate, and the source of the solvability certificate.

Hypothesis space: 4-digit strings with distinct digits, leading zero allowed
(the AR-Bench prompt says "For example: 0123 or 9468", and data/gn/*.json
contains secrets like "0521"). |H| = 10*9*8*7 = 5040.

Feedback: (same_pos, diff_pos) exactly as arbench.utils.utils_gn.compare_guess.
"""

from itertools import permutations
from math import log2

import numpy as np

N_CODES = 25  # code = same_pos * 5 + diff_pos


def build_space():
    """All distinct-digit 4-digit strings, plus digit matrix and bitmasks."""
    cands = ["".join(p) for p in permutations("0123456789", 4)]
    digits = np.array([[int(c) for c in s] for s in cands], dtype=np.int8)
    masks = np.zeros(len(cands), dtype=np.uint16)
    for i, s in enumerate(cands):
        m = 0
        for c in s:
            m |= 1 << int(c)
        masks[i] = m
    return cands, digits, masks


_POPCNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def build_feedback_matrix(digits, masks, chunk=512):
    """FB[g, s] = feedback code when guessing g against secret s."""
    n = len(masks)
    fb = np.empty((n, n), dtype=np.uint8)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        same = np.zeros((hi - lo, n), dtype=np.uint8)
        for p in range(4):
            same += (digits[lo:hi, p][:, None] == digits[None, :, p]).astype(np.uint8)
        common = _POPCNT16[np.bitwise_and(masks[lo:hi, None], masks[None, :])]
        fb[lo:hi] = same * 5 + (common - same)
    return fb


class GNGame:
    """Exact belief bookkeeping over the GuessNumbers hypothesis space."""

    def __init__(self, fb, cands):
        self.fb = fb
        self.cands = cands
        self.index = {s: i for i, s in enumerate(cands)}
        # digit matrix (len(cands), 4): digits[i, p] = the digit at position p
        # of candidate i. Used to compute a compact, externally-derived belief
        # summary (per-digit in/out status, per-position known value) that is
        # O(1) in size regardless of how many candidates remain feasible.
        self.digits = np.array([[int(c) for c in s] for s in cands], dtype=np.int8)

    # ---- feasible set -------------------------------------------------

    def feasible(self, history):
        """Indices consistent with every (guess, code) pair in history."""
        alive = np.arange(len(self.cands))
        for guess, code in history:
            g = self.index[guess]
            alive = alive[self.fb[g, alive] == code]
        return alive

    def code(self, guess, secret):
        return int(self.fb[self.index[guess], self.index[secret]])

    # ---- information measures ----------------------------------------

    def partition_sizes(self, g, alive):
        """Counts of alive candidates per feedback code for guess index g."""
        return np.bincount(self.fb[g, alive], minlength=N_CODES)

    def available_bits(self, alive, restrict=None):
        """Max entropy over guesses: the most information any legal guess can
        extract from the current feasible set. Returns (bits, best_guess_idx)."""
        if len(alive) <= 1:
            return 0.0, None
        pool = np.arange(len(self.cands)) if restrict is None else restrict
        sub = self.fb[np.ix_(pool, alive)]
        n = len(alive)
        counts = np.zeros((len(pool), N_CODES), dtype=np.int32)
        for c in range(N_CODES):
            counts[:, c] = (sub == c).sum(1)
        p = counts / n
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, -p * np.log2(p, where=p > 0), 0.0)
        ent = terms.sum(1)
        # Tie-break toward guesses that are themselves still possible, so the
        # oracle can win outright instead of only splitting.
        in_alive = np.zeros(len(pool), dtype=bool)
        in_alive[np.isin(pool, alive)] = True
        best = int(np.lexsort((~in_alive, -ent))[0])
        return float(ent[best]), int(pool[best])

    def realized_bits(self, alive_before, guess, code):
        """Information the agent's actual move actually extracted."""
        g = self.index[guess]
        after = alive_before[self.fb[g, alive_before] == code]
        if len(after) == 0:
            return float("nan"), after  # inconsistent history
        return log2(len(alive_before) / len(after)), after

    def expected_bits(self, g, alive):
        """Entropy of the feedback distribution for guess index g, under a
        uniform prior over the feasible set. This is what the guess is *worth*
        before the answer is seen -- no luck in it."""
        n = len(alive)
        if n <= 1:
            return 0.0
        counts = self.partition_sizes(g, alive)
        p = counts[counts > 0] / n
        return float(-(p * np.log2(p)).sum())

    def efficiency(self, alive_before, guess, code):
        """Primary measure: ex-ante efficiency = expected bits of the guess the
        agent actually chose / expected bits of the best available guess. In
        [0, 1] by construction, and free of outcome luck.

        bits_realized is kept as a secondary readout, but note it is NOT
        bounded by bits_available: entropy is an average over feedback codes,
        so a lucky outcome legitimately exceeds it. Normalising the realized
        reduction would measure the agent's luck, not its policy."""
        avail, _ = self.available_bits(alive_before)
        g = self.index[guess]
        chosen = self.expected_bits(g, alive_before)
        got, after = self.realized_bits(alive_before, guess, code)
        return {
            "n_before": int(len(alive_before)),
            "n_after": int(len(after)),
            "bits_chosen": chosen,
            "bits_available": avail,
            "efficiency": float("nan") if avail <= 0 else chosen / avail,
            "bits_realized": got,
            "alive_after": after,
        }

    # ---- policies -----------------------------------------------------

    def solve_greedy(self, secret, max_turns=25):
        """Max-entropy oracle. Returns turns used (None if it fails)."""
        alive = np.arange(len(self.cands))
        for t in range(1, max_turns + 1):
            if len(alive) == 1:
                g = int(alive[0])
            else:
                _, g = self.available_bits(alive)
            guess = self.cands[g]
            if guess == secret:
                return t
            code = self.code(guess, secret)
            alive = alive[self.fb[g, alive] == code]
        return None

    def solve_random_consistent(self, secret, rng, max_turns=25):
        """Null policy: always guess a uniformly random still-feasible candidate.
        Strong in Mastermind-like games; an agent below this line is failing at
        belief *tracking*, not belief *use*."""
        alive = np.arange(len(self.cands))
        for t in range(1, max_turns + 1):
            g = int(rng.choice(alive))
            guess = self.cands[g]
            if guess == secret:
                return t
            code = self.code(guess, secret)
            alive = alive[self.fb[g, alive] == code]
        return None

    # ---- trajectory scoring -------------------------------------------

    def score_trajectory(self, secret, guesses, max_turns=25):
        """Per-step efficiency for one agent trajectory.

        Returns a list of per-turn dicts. Flags two distinct pathologies:
          dead    -- the guess was already ruled out by the agent's own history
                     (belief-tracking failure)
          stalled -- the guess was feasible but extracted ~no information
        """
        alive = np.arange(len(self.cands))
        rows = []
        for t, guess in enumerate(guesses[:max_turns], start=1):
            if guess not in self.index:
                rows.append({"turn": t, "guess": guess, "illegal": True})
                continue
            g = self.index[guess]
            dead = bool(g not in set(alive.tolist()))
            code = self.code(guess, secret)
            m = self.efficiency(alive, guess, code)
            alive = m.pop("alive_after")
            m.update(
                turn=t,
                guess=guess,
                code=code,
                dead_guess=dead,
                stalled=bool(m["bits_realized"] < 0.05),
                solved=(guess == secret),
            )
            rows.append(m)
            if guess == secret:
                break
        return rows
