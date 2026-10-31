"""The harness keeps the belief state; the model picks from it.

Every number that still agrees with all the feedback is enumerated exactly (P(10,4) filtered by
each (guess, in place, misplaced) line of env.history, with the oracle's own counting rule applied
to hypotheses, never to the target). The few most informative of them -- smallest expected number
of survivors after the next feedback -- are shown to the model as a short menu, and the model
writes one back. Every played guess is therefore the model's choice and, being taken from the
menu, cannot contradict anything already learned.

The prompt is the game in one sentence, the record in a compact form, and the menu; the reply is
capped at a few tokens. Nothing of AR-Bench's long rule text is resent each turn.

Hypothesis (iteration 1): a 7B model's failures on this task are failures of belief tracking --
playing numbers that earlier feedback already rules out -- rather than of choosing. If so, taking
the tracking out of the model and leaving it the choice solves nearly every episode well inside
the budget, at a small fraction of the tokens a free-form loop spends.

Fallbacks, which a well-behaved model never reaches: a reply naming no listed number is answered
once with the list again; a second miss plays the top-ranked entry.

Prototyped against stub models only; the cluster login had expired, so no real-model probe was
possible this session. On the real GNEnv (AR-Bench's compare_guess), 300 random secrets each:
a stub picking uniformly from the menu, one always naming the first entry inside a sentence, and
one that answers with no number 30% of the time and an off-list number 20% of the time all
solved 100%, in 5.3 guesses on average (max 8), at roughly 1.0k / 1.0k / 1.6k estimated tokens
per episode. Ranking costs about 30 ms of CPU per turn.
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

MENU = 6          # numbers shown per turn
N_CAND = 120      # how many survivors are scored for the menu
N_REF = 400       # against how many survivors each is scored

SYSTEM = ("You are playing a number-guessing game. The secret is a 4-digit number whose digits are "
          "all different. After each guess you are told how many digits are right and in the right "
          "place, and how many are right but in the wrong place.")


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


def ranked(survivors, rng):
    """Survivors ordered by the expected number left after playing them (fewest first).

    With nothing learned yet every number splits the space identically, so any order will do.
    """
    if len(survivors) == len(ALL):
        return rng.sample(survivors, MENU)
    cands = survivors if len(survivors) <= N_CAND else rng.sample(survivors, N_CAND)
    ref = survivors if len(survivors) <= N_REF else rng.sample(survivors, N_REF)

    def left(g):
        cells = {}
        for s in ref:
            k = score(g, s)
            cells[k] = cells.get(k, 0) + 1
        cells.pop((4, 0), None)                    # that cell is the win, nobody is left in it
        return sum(c * c for c in cells.values())

    return sorted(cands, key=left)


def record(history):
    if not history:
        return "none yet"
    return "\n".join("%d. %s: %d in the right place, %d right but misplaced" % (i + 1, g, a, b)
                     for i, (g, a, b) in enumerate(history))


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        last = None
        while env.remaining > 0 and not env.solved:
            last = self.pick(final=False)
            env.guess(last)
        if env.solved:
            return last
        return self.pick(final=True)               # out of guesses: the answer is still a pick

    def pick(self, final):
        hist = self.env.history
        played = {g for g, _, _ in hist}
        survivors = [n for n in consistent(hist) if n not in played] or \
                    [n for n in ALL if n not in played]
        menu = ranked(survivors, random.Random(repr(hist)))[:MENU]
        ask = ("Your final answer must be one of these numbers, which are the only ones that agree "
               "with every guess above:" if final else
               "Each of these numbers agrees with every guess above, so any of them could be the "
               "secret. They are listed from most to least informative:")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Guesses so far:\n%s\n\n%s\n%s\n\nReply with one number "
                 "from the list and nothing else." % (record(hist), ask, "\n".join(menu))}]
        allowed = set(survivors)
        for _ in range(2):
            out = self.llm(msgs, max_tokens=16) or ""
            nums = FOUR.findall(out)
            for n in nums:
                if n in menu:
                    return n
            for n in nums:                         # off the menu but still possible: the model's call
                if n in allowed:
                    return n
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": "That is not one of the listed numbers. Reply with "
                      "exactly one of: %s" % ", ".join(menu)}]
        return menu[0]
