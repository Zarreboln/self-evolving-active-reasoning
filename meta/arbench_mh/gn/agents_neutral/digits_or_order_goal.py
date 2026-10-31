"""The model sets each turn's sub-goal, which digits or their order, and code plays to it.

Mechanism (axis C, how the next guess is chosen, with D, what the model is shown). The menu and
pair harnesses ask the model to name one of several possible secrets. Code has already judged
those secrets equally good, so the answer mostly follows position. A player of this game makes a
different decision: whether the next guess should find out which four digits are in the secret,
or how the digits it already suspects are ordered. That choice depends on the shape of what is
still unknown, not on any particular digits.

So belief stays in code (the secrets consistent with every feedback). Each turn code builds two
guesses among the possible secrets. The digits guess leaves the fewest digit sets on average.
The order guess leaves the fewest orders on average for the most-ordered set still in play. If
the two coincide, or only one kind of uncertainty is left (one digit set, or one order per set),
the guess is played without asking. Otherwise the model is shown the shape of the uncertainty
in coarse bands (how many secrets, how many digit sets, how many orders at most for one set) and
asked which to pin down, A the digits or B their order. Banding makes the prompt repeat across
episodes, so it is asked at temperature 0 and cached for the whole process. The opening is the
model's own free choice. Ties within a goal fall to fewer expected survivors. An unreadable
reply or a failed call falls back to the guess with fewer expected survivors.

Stub (100 random secrets, 16 threads; tokens estimated as one per digit, word or mark, which
puts pairwise_canon_tourney at 16.4 on the same stub against its measured 20.1): 100% solved,
5.38-5.44 guesses (max 7), 12-13 calls per process, 13.6-14.7 tokens/ep, whether the stub
always names A, always B or answers at random. Garbage replies: 24 calls (each prompt re-asked
once), play falls to fewest expected survivors, 100%. An endpoint that raises: 0 calls, 100%.
No real-model probe: the cluster login was down this session.

Nothing is learned about any secret: the cache holds the model's replies to prompts, starts
empty in every process, and the first episode is played exactly as it would be without it.
"""
from __future__ import annotations

import re
import threading
from itertools import permutations

from ..harness_base import GNHarness

FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
LETTER = re.compile(r"\b([AB])\b")
SECRETS = ["".join(p) for p in permutations("0123456789", 4)]
POP = [bin(i).count("1") for i in range(1024)]
SETKEY = {s: "".join(sorted(s)) for s in SECRETS}
WIN = (4, 0)


def mask(num):
    m = 0
    for c in num:
        m |= 1 << int(c)
    return m


MASK = {s: mask(s) for s in SECRETS}


def score(guess, secret):
    """(right place, misplaced) for `guess` if `secret` were the answer; AR-Bench's counting."""
    a = sum(g == s for g, s in zip(guess, secret))
    return a, POP[mask(guess) & MASK[secret]] - a


def spread(pool, n):
    """At most n items of pool, evenly spaced, so a large set is sampled without randomness."""
    if len(pool) <= n:
        return list(pool)
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def shape(pool):
    """(digit sets, most orders for one set) among the secrets in pool."""
    sets = {}
    for s in pool:
        k = SETKEY[s]
        sets[k] = sets.get(k, 0) + 1
    return len(sets), max(sets.values(), default=0)


def outcomes(g, evals):
    """Expected (digit sets, orders for the most-ordered set, survivors) left after playing g."""
    split = {}
    mg = MASK[g]
    for s in evals:
        a = (g[0] == s[0]) + (g[1] == s[1]) + (g[2] == s[2]) + (g[3] == s[3])
        k = (a, POP[mg & MASK[s]] - a)
        if k != WIN:
            split.setdefault(k, []).append(s)
    sets = orders = survivors = 0
    for members in split.values():
        m = len(members)
        d, o = shape(members)
        sets += m * d
        orders += m * o
        survivors += m * m
    return sets, orders, survivors


def band(v):
    if v <= 2:
        return str(v)
    if v <= 5:
        return "3 to 5"
    if v <= 20:
        return "6 to 20"
    return "more than 20"


SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

FIRST = "No guesses yet. Make your first guess: any four different digits. Reply with the number only."

GOAL = ("Possible secrets left: %s. They use %s different sets of four digits, and one set "
        "can still be ordered in up to %s ways.\n"
        "What should your next guess pin down?\n"
        "A) which four digits are in the secret\n"
        "B) the order of the digits\n"
        "Reply with the letter only.")


class Harness(GNHarness):
    # Shared by every instance in the process: prompt -> {"done": Event, "out": reply or None}.
    _memo = {}
    _memo_lock = threading.Lock()

    def solve(self) -> str:
        env = self.env
        alive = list(SECRETS)
        while not env.solved and env.remaining > 0:
            g = self.choose(alive)
            a, b = env.guess(g)
            alive = [s for s in alive if score(g, s) == (a, b)]
            if not alive:                     # cannot happen with a truthful oracle
                alive = list(SECRETS)
        if env.solved:
            return env.history[-1][0]
        return self.guesses(alive)[2]

    def ask(self, messages):
        """The model at temperature 0, asked at most once per distinct prompt across episodes.

        Returns None if the call fails, so the caller falls back to code's choice."""
        key = tuple((m["role"], m["content"]) for m in messages)
        cls = type(self)
        with cls._memo_lock:
            slot = cls._memo.get(key)
            owner = slot is None
            if owner:
                slot = cls._memo[key] = {"done": threading.Event(), "out": None}
        if owner:
            try:
                slot["out"] = self.llm(messages, temperature=0.0, max_tokens=8)
            except Exception:
                slot["out"] = None
            finally:
                if slot["out"] is None:       # the call failed: let the next asker retry
                    with cls._memo_lock:
                        cls._memo.pop(key, None)
                slot["done"].set()
            return slot["out"]
        slot["done"].wait()
        if slot["out"] is not None:
            self._trace.append({"step": "llm_memo", "messages": [dict(m) for m in messages],
                                "response": slot["out"]})
        return slot["out"]

    def first(self):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
        out = self.ask(msgs)
        for n in FOUR.findall(out or ""):
            if len(set(n)) == 4:
                return n
        if out is not None:
            out = self.ask(msgs + [{"role": "assistant", "content": out},
                                   {"role": "user", "content": "Four different digits, number only."}])
            for n in FOUR.findall(out or ""):
                if len(set(n)) == 4:
                    return n
        return SECRETS[0]

    @staticmethod
    def guesses(alive, pool_cap=200, eval_cap=500):
        """(digits guess, order guess, fewest-survivors guess) among the possible secrets."""
        evals = spread(alive, eval_cap)
        best_d = best_o = best_s = None
        for g in spread(alive, pool_cap):
            sets, orders, surv = outcomes(g, evals)
            kd, ko, ks = (sets, surv, g), (orders, surv, g), (surv, g)
            best_d = kd if best_d is None or kd < best_d else best_d
            best_o = ko if best_o is None or ko < best_o else best_o
            best_s = ks if best_s is None or ks < best_s else best_s
        return best_d[-1], best_o[-1], best_s[-1]

    def goal(self, alive, sets, orders):
        """The model's sub-goal: A (which digits) or B (their order); None if it gives none."""
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": GOAL % (band(len(alive)), band(sets), band(orders))}]
        out = self.ask(msgs)
        m = LETTER.search(out or "")
        if m is None and out is not None:
            out = self.ask(msgs + [{"role": "assistant", "content": out},
                                   {"role": "user", "content": "Reply with one letter, A or B."}])
            m = LETTER.search(out or "")
        return m.group(1) if m else None

    def choose(self, alive):
        if not self.env.history:
            return self.first()
        if len(alive) == 1:                   # nothing to choose between
            return alive[0]
        by_digits, by_order, by_survivors = self.guesses(alive)
        sets, orders = shape(alive)
        if sets == 1:                         # only the order is unknown
            return by_order
        if orders == 1:                       # only the digits are unknown
            return by_digits
        if by_digits == by_order:
            return by_digits
        pick = self.goal(alive, sets, orders)
        if pick == "A":
            return by_digits
        if pick == "B":
            return by_order
        return by_survivors
