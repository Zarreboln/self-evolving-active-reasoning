"""A ranked menu of the secrets still possible, plus one memory shared by every episode.

Mechanism (axis B, what carries across episodes). Episodes are solved one after another by the
same harness class, and their early turns are not private: before any feedback every episode is
in the same state, after one guess there are only a handful of states, after two a few dozen.
Starting each episode fresh pays for the same decision again in each of them.

So the decisions are remembered. Belief tracking and ranking are as in a menu harness: the set of
secrets consistent with every feedback is kept in code, its members are ranked by how finely they
would split the others, and the model sees the record as a table plus a short menu and replies
with the letter of the guess to play (its first guess is its own free choice; see
ranked_menu_pick for why letters: the model retypes numbers with a digit changed). It is asked at
temperature 0, so its answer is a function of the prompt -- and a class-level cache keyed by the
exact prompt returns that answer to any later episode that reaches the same state, without
calling the model again. Re-asking would have produced the same reply; the cache only stops it
being paid for twice. Concurrent episodes asking the same question wait for one call rather than
racing. Nothing is learned about any particular secret: the cache holds the model's own replies
to prompts, and it starts empty, so the first episode it sees is solved exactly as it would be
without it.

Measured with a stub model over 100 random secrets (16 threads): same accuracy and guesses as
the uncached harness, model calls per episode 5.24 -> 2.66. On the real model, 6 episodes run in
sequence already saved 1.2 calls per episode, before the cache had seen many states.
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
        return self.choose(alive, final=True)

    def ask(self, messages):
        """The model at temperature 0, asked at most once per distinct prompt across episodes."""
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
            finally:
                if slot["out"] is None:       # the call failed: let the next asker retry
                    with cls._memo_lock:
                        cls._memo.pop(key, None)
                slot["done"].set()
            return slot["out"]
        slot["done"].wait()
        if slot["out"] is None:
            return self.llm(messages, temperature=0.0, max_tokens=8)
        self._trace.append({"step": "llm_memo", "messages": [dict(m) for m in messages],
                            "response": slot["out"]})
        return slot["out"]

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
