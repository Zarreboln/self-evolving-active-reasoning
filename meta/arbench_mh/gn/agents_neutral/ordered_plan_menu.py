"""One reply per plan, not per guess: the model orders the menu and the harness plays down it.

Mechanism (axis A/C, what decides the next guess: a plan fixed in advance). In a menu harness
the model is asked once per guess, and almost every call after the second turn is a fresh state
no other episode has seen, so a cache shared across episodes cannot remove it. Yet the menu it
answers already lists every secret it might play over the next few turns, and the only thing a
later feedback can do to those items is rule some of them out -- it never makes a ruled-in item
a worse candidate in a way the model was shown.

So the model is asked for the order in which it would try the menu items, and that order is the
plan. The harness plays the first item of the plan that is still possible, updates the set of
possible secrets in code, and moves on to the next surviving item without asking again; the model
is asked for a new plan only when every item of the current one has been played or ruled out.
Belief tracking, ranking and the lettered menu are as in shared_menu_cache, and so is the
temperature-0 cache shared across episodes, which still serves the opening turns. A reply that
names only one letter is a one-item plan, so the worst case is the per-guess harness. Once a
single secret remains it is played without asking.

Every played guess is a secret consistent with all feedback so far, so the number of guesses is
that of playing possible secrets in some order -- far inside the budget -- whatever the plan.

Stub (100 random secrets, 16 threads, two seeds): 100% solved, ~5.2 guesses, model calls per
episode 2.85 for shared_menu_cache against 1.42 when the stub returns a full order and 1.73 when
it names only two letters. Real model (Qwen2.5-7B, temperature 0, 5 plan prompts drawn from stub
episodes): 5/5 replies parse as plans -- "A,C,B,D,E", "A,B,C,D,E", "B,C", "A,E,C,B,D",
"A,C,E,B,D" -- so the model reorders rather than echoing, and sometimes commits to only part of
the menu; 163-195 input and 3-6 output tokens per call.
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
        self.plan = []
        while not env.solved and env.remaining > 0:
            g = self.choose(alive)
            a, b = env.guess(g)
            alive = [s for s in alive if score(g, s) == (a, b)]
            if not alive:                     # cannot happen with a truthful oracle
                alive = list(SECRETS)
        if env.solved:
            return env.history[-1][0]
        return self.choose(alive, final=True)

    def ask(self, messages, max_tokens=24):
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
                slot["out"] = self.llm(messages, temperature=0.0, max_tokens=max_tokens)
            finally:
                if slot["out"] is None:       # the call failed: let the next asker retry
                    with cls._memo_lock:
                        cls._memo.pop(key, None)
                slot["done"].set()
            return slot["out"]
        slot["done"].wait()
        if slot["out"] is None:
            return self.llm(messages, temperature=0.0, max_tokens=max_tokens)
        self._trace.append({"step": "llm_memo", "messages": [dict(m) for m in messages],
                            "response": slot["out"]})
        return slot["out"]

    def prompt(self, alive, final=False):
        env = self.env
        rows = ["Record (guess: right place, misplaced):"]
        rows += ["%s: %d, %d" % (g, a, b) for g, a, b in env.history]
        if len(alive) <= MENU:
            menu = rank(alive)
            rows.append("Only these secrets are still possible:")
        else:
            menu = rank(alive)[:MENU]
            rows.append("%d secrets are still possible. The most informative of them, best "
                        "first:" % len(alive))
        rows += ["%s) %s" % (k, m) for k, m in zip(LETTERS, menu)]
        if final:
            rows.append("Which is your final answer? Reply with its letter only.")
        else:
            rows.append("In what order would you try them? Reply with the letters only, first "
                        "choice first, separated by commas.")
        return "\n".join(rows), menu

    @staticmethod
    def parse(out, menu, ok):
        """The plan the reply names, in its order: menu letters and possible secrets written out."""
        plan = []
        hi = LETTERS[len(menu) - 1]
        for tok in re.findall(r"(?<!\d)\d{4}(?!\d)|\b[A-%s]\b" % hi, out or ""):
            g = tok if tok.isdigit() else menu[LETTERS.index(tok)]
            if g in ok and g not in plan:
                plan.append(g)
        return plan

    def first(self):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
        out = self.ask(msgs, max_tokens=8)
        for n in FOUR.findall(out or ""):
            if len(set(n)) == 4:
                return n
        out = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                               {"role": "user", "content": "Four different digits, number only."}],
                       max_tokens=8)
        for n in FOUR.findall(out or ""):
            if len(set(n)) == 4:
                return n
        return SECRETS[0]

    def choose(self, alive, final=False):
        hist = self.env.history
        if not hist:
            return self.first()
        played = {g for g, _, _ in hist}
        live = set(alive) - played
        if len(live) == 1:                    # nothing to choose between
            return next(iter(live))
        if not final:
            # Keep following the current plan while any of it is still possible.
            self.plan = [g for g in self.plan if g in live]
            if self.plan:
                return self.plan.pop(0)
        text, menu = self.prompt([s for s in alive if s in live] or alive, final)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        out = self.ask(msgs, max_tokens=8 if final else 24)
        plan = self.parse(out, menu, live)
        if not plan:
            out = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                                   {"role": "user", "content": "Reply with letters only, from %s."
                                    % ", ".join(LETTERS[:len(menu)])}],
                           max_tokens=24)
            plan = self.parse(out, menu, live)
        if not plan:
            plan = [m for m in menu if m in live] or list(menu)
        if final:
            return plan[0]
        self.plan = plan[1:]
        return plan[0]
