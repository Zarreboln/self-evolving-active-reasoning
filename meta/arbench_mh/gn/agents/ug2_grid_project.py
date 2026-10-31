"""The model writes each guess from a grid of what the feedback proves; the harness projects it.

Instead of the raw record, the model is shown what the harness can prove from it exactly (P(10,4)
filtered by each (guess, in place, misplaced) line, the oracle's counting rule applied to
hypotheses, never to the target): which digits are certainly in the secret, which are certainly
out, which digits each position can still hold, and how many numbers remain. The model writes a
number freely. If it agrees with every feedback line it is played as written; otherwise the
harness plays the surviving number nearest to it (most digits in the same place, then most
digits shared; among equally near survivors, the one leaving the fewest expected survivors).
One call per guess, a capped reply, no retries.

The grid is a necessary condition only: a number built from it can still break the joint
constraint, which is what the projection repairs. Its size is fixed (four short position lines)
however long the game runs, so the prompt does not grow with the record.

Hypothesis (iteration 2): given the per-position facts, a 7B model's own proposals are mostly
consistent already and the projection rarely moves them, so the search is as short as a
consistent-guess strategy's at a few hundred tokens per guess. Falsified if mean guess rounds are
clearly above the native-prompt projection's (the grid then hurts the model's aim) or accuracy
drops below it.

Prototyped against stub models only (the cluster login had expired; no real-model probe was
possible), 200 random secrets per stub: a stub building its number from the grid, one writing
random numbers, and one writing no number all solved 100% in 5.3-5.4 guesses (max 8), ~820
tokens/ep (rough estimate).
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
_MASK = {n: sum(1 << int(c) for c in n) for n in ALL}
_POP = [bin(i).count("1") for i in range(1024)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")

N_TIE = 60        # how many equally near survivors are scored for the tie-break
N_REF = 300       # against how many survivors each is scored

SYSTEM = ("You are playing Guess Number. The secret is a 4-digit number with four different "
          "digits. Each guess is answered with how many digits are right and in the right place, "
          "and how many are right but in the wrong place.")


def _mask(n):
    m = _MASK.get(n)
    return m if m is not None else sum(1 << int(c) for c in set(n))


def score(guess, secret):
    """(in place, misplaced) if `secret` were the answer -- the oracle's rule, on a hypothesis."""
    a = sum(1 for x, y in zip(guess, secret) if x == y)
    return a, _POP[_mask(guess) & _mask(secret)] - a


def consistent(history):
    s = ALL
    for g, a, b in history:
        s = [n for n in s if score(g, n) == (a, b)]
    return s


def expected_left(g, ref):
    cells = {}
    for s in ref:
        k = score(g, s)
        cells[k] = cells.get(k, 0) + 1
    cells.pop((4, 0), None)
    return sum(c * c for c in cells.values())


def project(proposal, survivors, rng):
    if len(survivors) == 1:
        return survivors[0]
    pm = _mask(proposal) if proposal else 0
    best, near = None, []
    for n in survivors:
        key = 2 * sum(1 for x, y in zip(proposal, n) if x == y) + _POP[pm & _MASK[n]]
        if best is None or key > best:
            best, near = key, [n]
        elif key == best:
            near.append(n)
    if len(near) == 1:
        return near[0]
    cands = near if len(near) <= N_TIE else rng.sample(near, N_TIE)
    ref = survivors if len(survivors) <= N_REF else rng.sample(survivors, N_REF)
    return min(cands, key=lambda g: expected_left(g, ref))


def grid(survivors, history):
    """What every surviving number agrees on, in a few fixed-size lines."""
    if not history:
        return "No guesses yet: any four different digits are possible."
    present = [sum(1 for n in survivors if str(d) in n) for d in range(10)]
    sure_in = [str(d) for d in range(10) if present[d] == len(survivors)]
    sure_out = [str(d) for d in range(10) if present[d] == 0]
    lines = ["Last guess: %s, %d right place, %d wrong place." % history[-1],
             "Certainly in the secret: %s" % (" ".join(sure_in) or "none known"),
             "Certainly not in the secret: %s" % (" ".join(sure_out) or "none known")]
    for i in range(4):
        lines.append("Position %d can only be: %s"
                     % (i + 1, " ".join(sorted({n[i] for n in survivors}))))
    lines.append("%d numbers still fit every answer." % len(survivors))
    return "\n".join(lines)


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        last = None
        while env.remaining > 0 and not env.solved:
            survivors = consistent(env.history)
            alive = set(survivors)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "%s\n\nWrite your next guess: four different "
                     "digits that fit all of the above. Reply with the number only."
                     % grid(survivors, env.history)}]
            out = self.llm(msgs, max_tokens=12) or ""
            proposals = FOUR.findall(out)
            proposal = proposals[0] if proposals else ""
            last = proposal if proposal in alive else \
                project(proposal, survivors, random.Random(repr(env.history)))
            env.guess(last)
        return last
