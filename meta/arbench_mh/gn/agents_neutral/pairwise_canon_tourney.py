"""Each choice is decided by the model through pairwise comparisons, and a pair has few shapes.

Mechanism (axis C, the algorithm that decides what to play, with B, what carries across
episodes). A menu harness asks the model to pick one of several candidates, and since the
menu's contents are new in almost every state, a cache of the model's replies shared across
episodes stops helping after the opening turns. On a stub of 100 episodes, keying the cache by
the record rewritten up to digit renaming and position reordering cut the model's calls from
2.85 to 2.19 per episode -- but skipping forced choices alone already reached 2.22, so the
symmetry bought almost nothing: records three or more guesses deep are rarely symmetric.

A choice between two candidates is different. Renaming the ten digits one-to-one and reordering
the four positions the same way in every number leaves every (right place, misplaced) count
unchanged, so two pairs related by such a renaming are the same question -- and up to renaming
there are only a few dozen kinds of pair of four-digit secrets. So the menu is decided as a
knockout: belief is kept in code (the secrets consistent with every feedback), the menu is every
possible secret once few remain, else the most informative few (ranked by expected survivors),
and the model is asked, for the current favourite and each challenger in turn, which of the two
it would guess first. Each pair is shown in canonical form (the smaller sorted renaming), asked
at temperature 0, and the reply is cached for every later episode, so the model's preference
between two kinds of candidate is paid for once in the whole run. The winner of the knockout is
played. The opening guess is the model's own free choice, as before. Once a single secret
remains it is played without asking.

Stub (100 random secrets, 16 threads, two stub models): 100% solved, 5.3 guesses, 0.20 model
calls per episode against 2.85 for shared_menu_cache -- about 20 distinct prompts in the whole
run. Real model (Qwen2.5-7B, temperature 0, 6 canonical pairs drawn from stub episodes): 6/6
replies a valid letter, B five times and A once, at 99 input and 2 output tokens per call.

Nothing is learned about any secret: the cache holds the model's replies to prompts and
starts empty, so the first episode is played exactly as it would be without it.
"""
from __future__ import annotations

import re
import threading
from itertools import permutations

from ..harness_base import GNHarness

FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
SECRETS = ["".join(p) for p in permutations("0123456789", 4)]
POP = [bin(i).count("1") for i in range(1024)]
ORDERS = list(permutations(range(4)))          # the 24 ways to reorder the four positions


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


def canon_set(items):
    """(canonical sorted list, position order, digit names) for a small set of numbers.

    For a fixed position order and a fixed reading order of the items, naming digits 0, 1, 2, ...
    as they first appear gives the smallest renamed text; trying every reading order and every
    position order therefore reaches the smallest sorted list any renaming can produce.
    """
    best = None
    for order in ORDERS:
        for seq in permutations(items):
            names = {}
            for g in seq:
                for i in order:
                    if g[i] not in names:
                        names[g[i]] = str(len(names))
            key = tuple(sorted("".join(names[g[i]] for i in order) for g in items))
            if best is None or key < best[0]:
                best = (key, order, names)
    key, order, names = best
    names = dict(names)
    for d in "0123456789":
        if d not in names:
            names[d] = str(len(names))
    return list(key), order, names


def from_frame(num, order, names):
    back = {v: k for k, v in names.items()}
    out = [""] * 4
    for j, i in enumerate(order):
        out[i] = back[num[j]]
    return "".join(out)


SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

FIRST = "No guesses yet. Make your first guess: any four different digits. Reply with the number only."

PAIR = ("Two secrets that are still possible:\nA) %s\nB) %s\n"
        "Which would you guess first? Reply with its letter only.")

MENU = 5


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
        return self.choose(alive)

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

    @staticmethod
    def pick(out, pair):
        """The model's choice: an item it wrote out, else the letter it named, else None."""
        for n in FOUR.findall(out or ""):
            if n in pair:
                return n
        m = re.search(r"\b([AB])\b", out or "")
        return pair["AB".index(m.group(1))] if m else None

    def prefer(self, x, y):
        """Which of two possible secrets the model would guess first, asked in canonical form."""
        pair, order, names = canon_set([x, y])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PAIR % tuple(pair)}]
        out = self.ask(msgs)
        g = self.pick(out, pair)
        if g is None:
            out = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                                   {"role": "user", "content": "Reply with one letter, A or B."}])
            g = self.pick(out, pair)
        if g is None:
            g = pair[0]
        return from_frame(g, order, names)

    def first(self):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
        out = self.ask(msgs)
        for n in FOUR.findall(out or ""):
            if len(set(n)) == 4:
                return n
        out = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                               {"role": "user", "content": "Four different digits, number only."}])
        for n in FOUR.findall(out or ""):
            if len(set(n)) == 4:
                return n
        return SECRETS[0]

    def choose(self, alive):
        hist = self.env.history
        if not hist:
            return self.first()
        played = {g for g, _, _ in hist}
        live = [s for s in alive if s not in played] or alive
        if len(live) == 1:                    # nothing to choose between
            return live[0]
        menu = rank(live)[:MENU]
        champ = menu[0]
        for challenger in menu[1:]:
            champ = self.prefer(champ, challenger)
        return champ
