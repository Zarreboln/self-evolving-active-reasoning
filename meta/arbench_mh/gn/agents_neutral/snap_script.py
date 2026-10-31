"""The model writes its whole sequence of guesses once; each turn plays the possible secret nearest it.

Mechanism (axis A, where guesses come from, and C, a plan fixed in advance). The menu harnesses
have the model pick among candidates that code wrote, so every guess is code's and the model only
breaks ties between them, and they pay a prompt for every new kind of choice. Here the model
writes the guesses itself, once per process. Before any feedback it is asked for the guesses it
would make, in order. That reply is a script shared by every episode, asked at temperature 0.

A script written before any feedback would mostly contradict the record by its second line, so
each turn it is projected onto belief, which is kept in code as the secrets consistent with every
feedback. Turn t plays the possible secret nearest the script's t-th guess: the most digits in
common, then the most in the same place, then the fewest expected survivors. The opening line is
played exactly as written, since every secret is still possible then. If an episode outlasts the
script, the script starts over from its first line. Every played guess could be the secret, so
the budget of 25 is never close.

One call per process: the cost is one short prompt and one short reply for the whole split. If the
reply holds no usable number, the harness asks once more. If that also fails, or the endpoint is
down, belief alone picks the guess (fewest expected survivors).

Real model (Qwen2.5-7B, temperature 0). Asked to "write the eight guesses ... one per line" it
answered in prose and hit the token limit after two numbers. Asked for "the eight numbers only"
(this prompt), it gave eight valid guesses: 97 input and 40 output tokens. Stub (200 secrets,
replaying that script): 100% solved, 5.33 guesses (max 7). Ordering nearness by in-place digits
first gave the same result. Ignoring the script and playing by information alone gave 5.30, so
following the model's plan costs about 0.03 guesses per episode.

Nothing is learned about any secret: the script is the model's reply to a fixed prompt, starts
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


def rank(alive, pool=None, pool_cap=200, eval_cap=500):
    """Items of pool (default: alive) by expected survivors in alive after playing them."""
    evals = spread(alive, eval_cap)
    scored = []
    for g in spread(pool if pool is not None else alive, pool_cap):
        cells = {}
        for s in evals:
            k = score(g, s)
            cells[k] = cells.get(k, 0) + 1
        scored.append((sum(v * v for v in cells.values()), g))
    scored.sort()
    return [g for _, g in scored]


def nearest(target, live, alive):
    """The item of live closest to target: shared digits, then digits in place, then information."""
    best, ties = None, []
    for s in live:
        a, b = score(target, s)
        k = (a + b, a)
        if best is None or k > best:
            best, ties = k, [s]
        elif k == best:
            ties.append(s)
    return ties[0] if len(ties) == 1 else rank(alive, ties)[0]


SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

PLAN = ("Plan the game before it starts: the eight guesses you would make, in order. "
        "Reply with the eight numbers only, one per line, each four different digits.")

AGAIN = "Eight numbers, one per line, each with four different digits. Numbers only."


def read_script(out):
    """The distinct four-different-digit numbers in the reply, in the order written."""
    seen = []
    for g in FOUR.findall(out or ""):
        if len(set(g)) == 4 and g not in seen:
            seen.append(g)
    return seen


class Harness(GNHarness):
    # Shared by every instance in the process: {"done": Event, "script": [str, ...]}.
    _script = None
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
            return self.llm(messages, temperature=0.0, max_tokens=80)
        except Exception:                     # endpoint down: belief alone decides
            return None

    def script(self):
        """The model's planned guesses, asked once per process (twice if the first reply is empty)."""
        cls = type(self)
        with cls._lock:
            slot = cls._script
            owner = slot is None
            if owner:
                slot = cls._script = {"done": threading.Event(), "script": []}
        if owner:
            try:
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PLAN}]
                out = self.call(msgs)
                plan = read_script(out)
                if not plan:
                    plan = read_script(self.call(msgs + [
                        {"role": "assistant", "content": out or ""},
                        {"role": "user", "content": AGAIN}]))
                slot["script"] = plan
            finally:
                slot["done"].set()
        else:
            slot["done"].wait()
            self._trace.append({"step": "script_memo", "response": list(slot["script"])})
        return slot["script"]

    def choose(self, alive):
        hist = self.env.history
        played = {g for g, _, _ in hist}
        live = [s for s in alive if s not in played] or alive
        if len(live) == 1:                    # nothing to choose between
            return live[0]
        plan = self.script()
        if not plan:
            return rank(live)[0]
        return nearest(plan[len(hist) % len(plan)], live, live)
