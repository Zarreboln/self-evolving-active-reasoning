"""Belief tracked in code, choice made by the model: a ranked menu of the secrets still possible.

Mechanism (axes B state + C choosing what to play + D record shape). The zero-shot loop leaves
the model to do two jobs at once -- remember what every feedback rules out, and decide what to
try next -- and a 7B model does the first one badly enough that most of its guesses could not be
the secret. This harness splits the jobs:

  * State, in code: the set of secrets consistent with every feedback so far, kept by filtering
    the 5040 candidates after each guess. Nothing the record implies is ever forgotten.
  * Ranking, in code: each surviving secret is scored by how finely it would split the others if
    played (the expected number of secrets left after its feedback), so the best probes come
    first.
  * Choice, by the model: it sees the record as a table and a short lettered menu -- the
    top-ranked survivors, or all of them once few are left -- and names the one to play. Its
    first guess is its own free choice, and a possible secret it writes out instead of a letter
    is played as written.

The menu is answered by letter because, asked for the number itself, Qwen2.5-7B retyped a menu
item with one digit changed on about two thirds of menu turns (probe: 6 episodes, 6/6 solved,
4.8 guesses, 7.5 calls and ~1.2k tokens per episode), and each slip cost a correction call. By
letter, 12/12 replies on the same states were valid, at ~161 input and 2 output tokens per call,
and the model did not simply take the first option (A 5, B 1, C 2, D 4). A reply naming nothing
on the menu gets one correction; after that the top-ranked survivor is played.
"""
from __future__ import annotations

import re
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


def rank(alive, pool_cap=200, eval_cap=500):
    """Survivors ordered by expected number of survivors after playing them (fewest first)."""
    evals = spread(alive, eval_cap)
    scored = []
    for g in spread(alive, pool_cap):
        cells = {}
        for s in evals:
            k = score(g, s)
            cells[k] = cells.get(k, 0) + 1
        scored.append((sum(v * v for v in cells.values()), g))
    scored.sort()
    return [g for _, g in scored]


SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

FIRST = "No guesses yet. Make your first guess: any four different digits. Reply with the number only."

MENU = 5
LETTERS = "ABCDE"


class Harness(GNHarness):
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
        return self.choose(alive, final=True)

    def ask(self, messages):
        return self.llm(messages, temperature=0.0, max_tokens=8)

    def prompt(self, alive, final=False):
        env = self.env
        rows = ["Record (guess: right place, misplaced):"]
        rows += ["%s: %d, %d" % (g, a, b) for g, a, b in env.history]
        if len(alive) <= MENU:
            menu = alive if len(alive) < 2 else rank(alive)
            rows.append("Only these secrets are still possible:")
        else:
            menu = rank(alive)[:MENU]
            rows.append("%d secrets are still possible. The most informative of them, best "
                        "first:" % len(alive))
        rows += ["%s) %s" % (k, m) for k, m in zip(LETTERS, menu)]
        rows.append("Which is your final answer? Reply with its letter only." if final
                    else "Which do you guess next? Reply with its letter only.")
        return "\n".join(rows), menu

    @staticmethod
    def pick(out, menu, ok):
        """The model's choice: a possible secret it wrote out, else the menu letter it named."""
        for n in FOUR.findall(out or ""):
            if n in ok:
                return n
        m = re.search(r"\b([A-%s])\b" % LETTERS[len(menu) - 1], out or "")
        if m and menu[LETTERS.index(m.group(1))] in ok:
            return menu[LETTERS.index(m.group(1))]
        return None

    def choose(self, alive, final=False):
        played = {g for g, _, _ in self.env.history}
        if not self.env.history:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
            out = self.ask(msgs)
            nums = FOUR.findall(out or "")
            for n in nums:
                if len(set(n)) == 4:
                    return n
            if nums:
                return nums[0]
            out = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                                   {"role": "user", "content": "Four different digits, number only."}])
            nums = FOUR.findall(out or "")
            return nums[0] if nums else alive[0]
        text, menu = self.prompt(alive, final)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        ok = set(alive) - played
        out = self.ask(msgs)
        g = self.pick(out, menu, ok)
        if g:
            return g
        out = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                               {"role": "user", "content": "Reply with one letter, %s."
                                % " or ".join(LETTERS[:len(menu)])}])
        g = self.pick(out, menu, ok)
        if g:
            return g
        return next((m for m in menu if m in ok), menu[0])
