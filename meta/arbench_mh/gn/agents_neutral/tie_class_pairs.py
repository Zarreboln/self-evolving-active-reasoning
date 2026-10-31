"""The model is asked only between candidates code cannot tell apart, by pairwise knockout.

Mechanism (axis E, when the model is consulted, built on pairwise_canon_tourney's axis C/B).
The tourney keeps belief in code (the secrets consistent with every feedback), takes the five
possible secrets with the fewest expected survivors, and decides them by a knockout of
two-candidate questions. Each pair is written canonically up to digit renaming and position
reordering and cached across episodes. It paid for 20 prompts in its whole split. But the five
are not equally good. The model's pick can play a candidate that code ranks worse, and its pair
answers looked like a habit of position (B five times in six), so the choice is mostly
arbitrary.

Here a candidate's merit is the whole way it splits the possible secrets: the sorted sizes of
its feedback groups. Two candidates with the same sizes are interchangeable for every measure
code could use (expected survivors, worst case, chance of a win). So the knockout runs only over
the tie class, the candidates whose split has exactly the best sizes. If one candidate is best on
its own, code plays it without asking. The model then decides exactly where code is indifferent,
which is the only place its preference cannot cost information. Pairs are canonical, asked at
temperature 0 and cached as before. The opening is the model's own free choice. A lone survivor
is played without asking. A failed call or unreadable reply keeps code's favourite.

Stub (100 random secrets, 16 threads; tokens estimated as one per digit, word or mark, which
puts pairwise_canon_tourney at 16.4 and 20 calls on the same stub against its measured 20.1):
100% solved, 5.17-5.35 guesses (max 7) against the base's 5.29-5.32 (max 8), with 18-19 calls
(14.7-15.5 tokens/ep). So the gate barely removes kinds of pair: ties are common late in an
episode, where every candidate splits the rest perfectly, and the kinds met there are most of
the 19. Its effect is on play (no candidate code ranks worse is ever chosen) more than on cost.
Garbage replies: 40 calls, favourite kept, 100%. An endpoint that raises: 0 calls, 100%. No
real-model probe: the cluster login was down this session.

Nothing is learned about any secret: the cache holds the model's replies to prompts, starts
empty in every process, and the first episode is played exactly as it would be without it.
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


def splits(alive, pool_cap=200, eval_cap=500):
    """[(sorted group sizes, guess)] for candidates in alive, best split first.

    Best means fewest expected survivors, then the smaller worst case; the sizes include the
    winning group, so a candidate that could be the secret is told apart from one that could not.
    """
    evals = spread(alive, eval_cap)
    scored = []
    for g in spread(alive, pool_cap):
        cells = {}
        for s in evals:
            k = score(g, s)
            cells[k] = cells.get(k, 0) + 1
        sizes = tuple(sorted(cells.values(), reverse=True))
        scored.append((sum(v * v for v in sizes), sizes, g))
    scored.sort()
    return [(sizes, g) for _, sizes, g in scored]


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

MENU = 5            # at most this many of the tie class go into the knockout


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
        """The model at temperature 0, asked at most once per distinct prompt across episodes.

        Returns None if the call fails, so the caller keeps code's choice."""
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
        if g is None and out is not None:
            out = self.ask(msgs + [{"role": "assistant", "content": out},
                                   {"role": "user", "content": "Reply with one letter, A or B."}])
            g = self.pick(out, pair)
        if g is None:
            return x                          # no readable answer: keep code's favourite
        return from_frame(g, order, names)

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

    def choose(self, alive):
        hist = self.env.history
        if not hist:
            return self.first()
        played = {g for g, _, _ in hist}
        live = [s for s in alive if s not in played] or alive
        if len(live) == 1:                    # nothing to choose between
            return live[0]
        ranked = splits(live)
        best = ranked[0][0]
        ties = [g for sizes, g in ranked if sizes == best][:MENU]
        champ = ties[0]
        for challenger in ties[1:]:           # one candidate best on its own: no question
            champ = self.prefer(champ, challenger)
        return champ
