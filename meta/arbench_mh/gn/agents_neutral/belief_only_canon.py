"""The model is shown the belief instead of the record, in a canonical form shared across episodes.

Mechanism (axis D, the shape of what is fed back, and B, what carries across episodes). A menu
harness keeps the set of possible secrets in code and shows the model the record as well as a
menu drawn from that set. But once belief lives in code the record tells the model nothing the
menu does not already settle -- every menu item is consistent with all of it -- and it is what
makes each prompt unique: two episodes facing the same choice between the same kind of candidates
hold different records, so a cross-episode cache keyed by the prompt rarely hits after the
second turn.

So the record is dropped. The model is shown only the candidates it chooses between: every
possible secret once at most five remain, otherwise the three most informative (ranked in code by
expected survivors, as in ranked_menu_pick). And the candidates are shown in a canonical form: renaming the
ten digits one-to-one and reordering the four positions the same way in every number leaves
every pairwise (right place, misplaced) count unchanged, so a set of candidates and its renamed
copy are the same choice. The set is rewritten as the smallest sorted list reachable by such a
renaming (searched over the 24 position orders and the orders in which digits are first named),
the model picks a letter at temperature 0, the reply is cached under that canonical prompt for
every later episode, and the item named is mapped back to the episode's own digits before it is
played. The opening guess is the model's own free choice, as before. Once a single secret
remains it is played without asking.

Why three for a partial menu: the fewer items a canonical set has, the fewer kinds of set
there are, so the more often a later episode asks a question already answered. Stub (100 random
secrets, 16 threads, a deterministic stub model, two seeds): 100% solved, ~5.3 guesses; model
calls per episode 2.85 for shared_menu_cache, 1.37-1.42 with five-item partial menus (88 of 137
fresh calls were partial menus), 0.80-0.83 with three. Real model (Qwen2.5-7B, temperature 0,
5 menu prompts drawn from stub episodes): 5/5 replies a valid letter (B, B, D, C, B -- not just
the first item), 107-128 input and 2 output tokens per call.

Nothing is learned about any secret: the cache holds the model's replies to prompts, starts
empty, and the first episode is played exactly as it would be without it.
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

MENU = 5                                       # a complete menu: every possible secret
PARTIAL = 3                                    # else the most informative few
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

    @staticmethod
    def prompt(menu, complete, final=False):
        rows = ["Only these secrets are still possible:" if complete
                else "Some of the secrets still possible, the most informative ones:"]
        rows += ["%s) %s" % (k, m) for k, m in zip(LETTERS, menu)]
        rows.append("Which is your final answer? Reply with its letter only." if final
                    else "Which do you guess next? Reply with its letter only.")
        return "\n".join(rows)

    @staticmethod
    def pick(out, menu):
        """The model's choice: a menu item it wrote out, else the menu letter it named."""
        for n in FOUR.findall(out or ""):
            if n in menu:
                return n
        m = re.search(r"\b([A-%s])\b" % LETTERS[len(menu) - 1], out or "")
        return menu[LETTERS.index(m.group(1))] if m else None

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

    def choose(self, alive, final=False):
        hist = self.env.history
        if not hist:
            return self.first()
        played = {g for g, _, _ in hist}
        live = [s for s in alive if s not in played] or alive
        if len(live) == 1:                    # nothing to choose between
            return live[0]
        complete = len(live) <= MENU
        shown = live if complete else rank(live)[:PARTIAL]
        menu, order, names = canon_set(shown)
        text = self.prompt(menu, complete, final)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        out = self.ask(msgs)
        g = self.pick(out, menu)
        if g is None:
            out = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                                   {"role": "user", "content": "Reply with one letter, %s."
                                    % " or ".join(LETTERS[:len(menu)])}])
            g = self.pick(out, menu)
        if g is None:
            g = menu[0]
        return from_frame(g, order, names)
