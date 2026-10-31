"""Every pairwise question the knockout can ever ask, answered by the model in one batched call.

Mechanism (axis F, the shape and number of model passes, with B, what carries across episodes).
pairwise_canon_tourney keeps belief in code (the secrets consistent with every feedback), ranks the
survivors by expected survivors, and decides the menu by a knockout of two-candidate questions,
each written canonically up to digit renaming and position reordering and cached across
episodes. Its whole split then cost 20 prompts: the opening question and 19 pairs at 99 input and
2 output tokens each, about a third of it the same system preamble repeated 20 times.

Up to renaming there are only 19 kinds of pair of distinct secrets (enumerated below at import),
so the set of questions the knockout can ever ask is known before the first episode starts. Here
the model is asked all of them at once, with the opening guess on the same sheet: one numbered
prompt per process ("k) A <pair> B <pair>" for k = 1..19, then "20) your first guess"), at
temperature 0, filed in a ledger shared by every episode. The knockout then reads the model's
answer for each pair from the ledger. A line the reply leaves unparsed falls back to the
single-pair question, asked and cached as in the base. A failed call never crashes an episode;
the code's ranking decides instead.

Real model (Qwen2.5-7B, temperature 0), three sheet layouts. With the opening as line 0 it
answered all 19 pairs but wrote "0" for the opening. Naming the opening in the reply format made
it stop after two lines. With the opening asked last, it answered all 20 lines: 432 input and 95
output tokens, opening 0123. The letters flipped with the layout (mostly B in one, all A in
another), so a pair answer is presentation as much as preference. Stub (100 secrets, 16 threads,
replaying that reply): 1 call per process against 19 for pairwise_canon_tourney, 100% solved,
5.35 guesses (max 7). Endpoint raising, garbage replies, or a half-answered sheet: still 100%.

Nothing is learned about any secret: the ledger holds the model's replies to prompts, starts
empty in every process, and the first episode is played exactly as later ones are.
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


# Every pair of distinct secrets is a renaming of ("0123", y) for some y, so these are all kinds.
KINDS = sorted({tuple(canon_set(["0123", y])[0]) for y in SECRETS if y != "0123"})
LINE = {k: i + 1 for i, k in enumerate(KINDS)}
OPENING = len(KINDS) + 1                       # the sheet's last line: the first guess

SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

SHEET = ("Answer every numbered line.\n"
         "Lines 1-%d: later, two secrets are still possible. Which would you guess first, A or B?\n"
         "%s\n"
         "%d) Your first guess: any four different digits.\n"
         "Reply with one line per number: 1) your letter, 2) your letter, and so on, then %d) your "
         "number.")

PAIR = ("Two secrets that are still possible:\nA) %s\nB) %s\n"
        "Which would you guess first? Reply with its letter only.")

MENU = 5


def sheet():
    lines = "\n".join("%d) A %s B %s" % (LINE[k], k[0], k[1]) for k in KINDS)
    return SHEET % (len(KINDS), lines, OPENING, OPENING)


def read_sheet(out):
    """(first guess or None, {kind: "A"/"B"}) from the model's reply to the sheet."""
    first, ledger = None, {}
    by_line = {v: k for k, v in LINE.items()}
    for m in re.finditer(r"(?m)^[\s*#-]*(\d{1,2})\s*[).:\-]+\s*(.*)$", out or ""):
        n, rest = int(m.group(1)), m.group(2)
        if n == OPENING:
            for g in FOUR.findall(rest):
                if len(set(g)) == 4:
                    first = g
                    break
            continue
        kind = by_line.get(n)
        if kind is None or kind in ledger:
            continue
        c = re.search(r"\b([AB])\b", rest)
        if c:
            ledger[kind] = c.group(1)
            continue
        for g in FOUR.findall(rest):          # the model wrote the number instead of its letter
            if g in kind:
                ledger[kind] = "AB"[kind.index(g)]
                break
    return first, ledger


class Harness(GNHarness):
    # Shared by every instance in the process.
    _memo = {}                                # prompt -> {"done": Event, "out": reply or None}
    _memo_lock = threading.Lock()
    _sheet = None                             # {"done": Event, "first": str|None, "ledger": dict}

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

    def call(self, messages, max_tokens):
        try:
            return self.llm(messages, temperature=0.0, max_tokens=max_tokens)
        except Exception:                     # endpoint down: the code's own ranking decides
            return None

    def ask(self, messages, max_tokens=8):
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
                slot["out"] = self.call(messages, max_tokens)
            finally:
                if slot["out"] is None:       # the call failed: let the next asker retry
                    with cls._memo_lock:
                        cls._memo.pop(key, None)
                slot["done"].set()
            return slot["out"]
        slot["done"].wait()
        if slot["out"] is None:
            return self.call(messages, max_tokens)
        self._trace.append({"step": "llm_memo", "messages": [dict(m) for m in messages],
                            "response": slot["out"]})
        return slot["out"]

    def ledger(self):
        """The model's answers to the whole sheet, asked once per process."""
        cls = type(self)
        with cls._memo_lock:
            slot = cls._sheet
            owner = slot is None
            if owner:
                slot = cls._sheet = {"done": threading.Event(), "first": None, "ledger": {}}
        if owner:
            try:
                out = self.call([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": sheet()}], 400)
                slot["first"], slot["ledger"] = read_sheet(out)
            finally:
                slot["done"].set()
        slot["done"].wait()
        return slot

    @staticmethod
    def pick(out, pair):
        """The model's choice: an item it wrote out, else the letter it named, else None."""
        for n in FOUR.findall(out or ""):
            if n in pair:
                return n
        m = re.search(r"\b([AB])\b", out or "")
        return pair["AB".index(m.group(1))] if m else None

    def prefer(self, x, y):
        """Which of two possible secrets the model would guess first: its ledger line, else asked."""
        pair, order, names = canon_set([x, y])
        letter = self.ledger()["ledger"].get(tuple(pair))
        if letter:
            self._trace.append({"step": "ledger", "pair": list(pair), "response": letter})
            return from_frame(pair["AB".index(letter)], order, names)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PAIR % tuple(pair)}]
        g = self.pick(self.ask(msgs), pair)
        return x if g is None else from_frame(g, order, names)

    def first(self):
        g = self.ledger()["first"]
        if g:
            return g
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "No guesses yet. Make your first guess: any four "
                                            "different digits. Reply with the number only."}]
        for n in FOUR.findall(self.ask(msgs) or ""):
            if len(set(n)) == 4:
                return n
        return rank(SECRETS)[0]

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
