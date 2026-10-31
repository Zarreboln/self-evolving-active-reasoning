"""The model chooses, once per process, the rule every later guess is played by.

Mechanism (axis C, the algorithm that decides what to play, with B, what carries across episodes:
a charter written once and followed for the whole run). The menu harnesses pay a prompt for every
new kind of choice, and the model's answer to each one is a tie-break: in the pairwise probe it
named B five times out of six. Here the model makes one decision about how to choose instead of
deciding each choice. On one sheet, before any feedback, it writes its first guess and picks a
rule for choosing among the secrets that are still possible:

  A) the one whose feedback would leave the fewest others possible on average
  B) the one sharing the most digits with the other possible secrets
  C) the smallest number
  D) the largest number

Belief is kept in code (the secrets consistent with every feedback), and every turn after the
first plays the possible secret the chosen rule selects. Every rule picks a possible secret, so
the budget of 25 is never close whichever one the model takes. The rules differ in how many guesses
they need, not in whether the episode ends solved.

One call per process, at temperature 0, shared by every episode. If the reply names no rule the
sheet is asked once more. If that also fails, or the endpoint is down, rule A is used and the code
writes the opening.

Real model (Qwen2.5-7B, temperature 0): "1) 1234 2) A" for 153 input and 12 output tokens. Stub
(100 secrets, 16 threads): 100% solved under every rule. Mean guesses were 5.42 (A), 5.70 (B),
5.72 (C) and 5.54 (D), with a max of 8. An endpoint raising, or a garbage reply: still 100%.

Nothing is learned about any secret: the charter is the model's reply to a fixed prompt, starts
empty in every process, and is the same for the first episode as for the last.
"""
from __future__ import annotations

import re
import threading
from itertools import permutations

from ..harness_base import GNHarness

FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
SECRETS = ["".join(p) for p in permutations("0123456789", 4)]
POP = [bin(i).count("1") for i in range(1024)]


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


def informative(live, pool_cap=200, eval_cap=500):
    """Rule A: fewest expected survivors after playing it."""
    evals = spread(live, eval_cap)
    best = None
    for g in spread(live, pool_cap):
        cells = {}
        for s in evals:
            k = score(g, s)
            cells[k] = cells.get(k, 0) + 1
        key = (sum(v * v for v in cells.values()), g)
        if best is None or key < best:
            best = key
    return best[1]


def typical(live, pool_cap=200, eval_cap=500):
    """Rule B: most digits shared with the other possible secrets (summed over a sample)."""
    evals = spread(live, eval_cap)
    best = None
    for g in spread(live, pool_cap):
        key = (-sum(POP[MASK[g] & MASK[s]] for s in evals if s != g), g)
        if best is None or key < best:
            best = key
    return best[1]


RULES = {"A": informative, "B": typical, "C": min, "D": max}

SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

SHEET = ("Answer both questions.\n"
         "1) Your first guess: any four different digits.\n"
         "2) Later in the game, several secrets will still be possible. Which should I guess?\n"
         "A) the one whose feedback would leave the fewest others possible\n"
         "B) the one sharing the most digits with the other possible secrets\n"
         "C) the smallest number\n"
         "D) the largest number\n"
         "Reply in the form: 1) your number 2) your letter")

AGAIN = "Reply in exactly this form: 1) your number 2) your letter"


def read_sheet(out):
    """(first guess or None, rule letter or None) from the model's reply."""
    first = rule = None
    text = out or ""
    m = re.search(r"(?s)(?:^|\n|\s)2\s*[).:\-]+(.*)", text)
    head = text[:m.start()] if m else text
    for g in FOUR.findall(head):
        if len(set(g)) == 4:
            first = g
            break
    if m:
        c = re.search(r"\b([ABCD])\b", m.group(1))
        rule = c.group(1) if c else None
    return first, rule


class Harness(GNHarness):
    # Shared by every instance in the process: {"done": Event, "first": str, "rule": str}.
    _charter = None
    _lock = threading.Lock()

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
        return self.choose(alive)

    def call(self, messages):
        try:
            return self.llm(messages, temperature=0.0, max_tokens=24)
        except Exception:                     # endpoint down: the default rule decides
            return None

    def charter(self):
        """The model's opening and rule, asked once per process (twice if the reply is unusable)."""
        cls = type(self)
        with cls._lock:
            slot = cls._charter
            owner = slot is None
            if owner:
                slot = cls._charter = {"done": threading.Event(), "first": None, "rule": None}
        if owner:
            try:
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": SHEET}]
                out = self.call(msgs)
                first, rule = read_sheet(out)
                if first is None or rule is None:
                    f2, r2 = read_sheet(self.call(msgs + [
                        {"role": "assistant", "content": out or ""},
                        {"role": "user", "content": AGAIN}]))
                    first, rule = first or f2, rule or r2
                slot["first"] = first or informative(SECRETS)
                slot["rule"] = rule or "A"
            finally:
                slot["done"].set()
        else:
            slot["done"].wait()
            self._trace.append({"step": "charter_memo", "response": dict(
                first=slot["first"], rule=slot["rule"])})
        return slot

    def choose(self, alive):
        hist = self.env.history
        c = self.charter()
        if not hist:
            return c["first"]
        played = {g for g, _, _ in hist}
        live = [s for s in alive if s not in played] or alive
        if len(live) == 1:                    # nothing to choose between
            return live[0]
        return RULES[c["rule"]](live)
