"""The model commits to an ordered plan; the harness carries it across turns and prunes it.

The harness keeps the exact belief state (every number that agrees with all the feedback, with
the oracle's counting rule applied to hypotheses, never to the target) and shows the model a
short, lettered menu of the most informative survivors (fewest expected survivors afterwards).
The model answers with the letters in the order it wants to try them. The harness then plays
that plan turn after turn, dropping any entry the newest feedback has ruled out, and asks again
only when the plan runs dry. When exactly one number is left there is nothing to choose, and it
is played without a call.

What is new against a one-pick-per-turn menu: one model reply can cover several guesses, so the
number of calls falls below the number of guesses; and the prompt is constant-size -- one line of
rules plus the menu, no echoed record, since the menu already encodes everything the record says.
Letters instead of numbers keep the reply to a few tokens (Qwen spells numbers digit by digit).

Hypothesis (iteration 2): the cost of a menu harness is calls x prompt, not the search. Plan reuse
plus a record-free prompt solves as many episodes as a per-turn menu (every played number is
still consistent, so the episode ends well inside the budget) at well under 1k tokens/ep.
Falsified if accuracy drops below the per-turn menu, or if calls/ep stay near guesses/ep (the
model lists one letter, so plans never carry over).

Fallbacks: a reply naming no letter or listed number is read as the menu's own order.

Prototyped against stub models only (the cluster login had expired; no real-model probe was
possible), on GNEnv with AR-Bench's compare_guess, 200 random secrets per stub: every stub solved
100% in 5.3-5.4 guesses (max 8). A stub listing all six letters made 3.3 calls/ep (~350 tokens,
rough estimate); one naming only the first letter, i.e. no plan reuse, made 4.7 (~480); one
giving junk or prose 50% of the time made 3.7 (~400).
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
LETTER = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")

LABELS = "ABCDEF"
N_CAND = 120      # how many survivors are scored for the menu
N_REF = 400       # against how many survivors each is scored

SYSTEM = "You pick guesses in a code-breaking game. Answer with letters only."


def _mask(n):
    m = _MASK.get(n)
    return m if m is not None else sum(1 << int(c) for c in set(n))


def score(guess, secret):
    """(in place, misplaced) if `secret` were the answer -- the oracle's rule, on a hypothesis."""
    a = sum(1 for x, y in zip(guess, secret) if x == y)
    return a, _POP[_mask(guess) & _mask(secret)] - a


def consistent(history, pool=ALL):
    s = pool
    for g, a, b in history:
        s = [n for n in s if score(g, n) == (a, b)]
    return s


def ranked(survivors, rng):
    """Survivors ordered by the expected number left after playing them (fewest first)."""
    if len(survivors) == len(ALL):
        return rng.sample(survivors, len(LABELS))   # nothing learned: every number is alike
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


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        plan, last = [], None
        while env.remaining > 0 and not env.solved:
            survivors = consistent(env.history)    # a played miss never agrees with its own feedback
            alive = set(survivors)
            plan = [n for n in plan if n in alive]
            if not plan:
                plan = survivors if len(survivors) <= 1 else self.ask(survivors)
            if not plan:                           # unreachable with a consistent oracle
                plan = [n for n in ALL if n not in {g for g, _, _ in env.history}][:1]
            last = plan.pop(0)
            env.guess(last)
        if env.solved or not plan:
            return last
        return plan[0]

    def ask(self, survivors):
        hist = self.env.history
        menu = ranked(survivors, random.Random(repr(hist)))[:len(LABELS)]
        if len(menu) == 1:
            return menu
        lines = "\n".join("%s) %s" % (LABELS[i], n) for i, n in enumerate(menu))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                 "The secret is 4 different digits. Every number below still fits all the feedback "
                 "so far, listed from most to least useful:\n%s\n\nReply with the letters in the "
                 "order you want to try them, best first." % lines}]
        out = self.llm(msgs, max_tokens=16) or ""
        plan = []
        for tok in LETTER.findall(out) + FOUR.findall(out):   # capitals only: "I'd", "a" are words
            n = menu[LABELS.index(tok)] if len(tok) == 1 and LABELS.index(tok) < len(menu) else tok
            if n in menu and n not in plan:
                plan.append(n)
        return plan or menu
