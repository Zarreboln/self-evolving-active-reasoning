"""Every kind of pair is put to the model in both orders; only an order-proof preference counts.

Mechanism (axis C, the rule that decides what to play, with F, the shape of the model's passes).
pairwise_canon_tourney keeps belief in code (the secrets consistent with every feedback), ranks
the survivors by expected survivors, and decides the top five by a knockout of two-candidate
questions written canonically up to digit renaming and position reordering. In its probe the
model named B five times out of six, and on a batched sheet its letters flipped with the layout:
much of what the knockout reads as preference is where a number was printed.

Here each of the 19 kinds of pair (all pairs of distinct secrets, up to renaming, enumerated at
import) is asked twice, on two numbered sheets that list the same pairs with A and B swapped.
The model's choice between two numbers counts only if it names the same number on both sheets;
a letter habit picks A on one sheet and A on the other, which names different numbers, so it
cancels. Where the two sheets disagree, or a line is missing, the knockout keeps its favourite,
the candidate code ranks better. The opening is the last line of the first sheet, the layout
the model answered in full in iteration 3's probe (all 20 lines, 432 input and 95 output tokens).

Two calls per process at temperature 0, shared by every episode; no call is made per pair, so
the cost does not grow with the kinds of state met. A failed or unreadable call never crashes an
episode: missing answers fall to code, a missing opening to a short opening question, then to code.

Stub (100 random secrets, 16 threads; tokens estimated by counting each digit, word and mark as
one token, which puts pairwise_canon_tourney at 18.7 against its measured 20.1): 2 calls per
process, 8.6 tokens/ep, 100% solved, 5.35 guesses (max 7). A stub that always names A, or always
B, leaves 0 of 19 kinds order-proof, so play is code's ranking; a stub that names the smaller
number leaves 19 of 19. Garbage replies: 3 calls (the opening re-asked), 100%. An endpoint that
raises: 0 calls, 100%. No real-model probe: the cluster login was down this session, so how many
kinds the 7B model answers the same way in both orders is the open question.

Nothing is learned about any secret: the sheets hold the model's replies to fixed prompts, start
empty in every process, and are the same for the first episode as for the last.
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
OPENING = len(KINDS) + 1                       # the first sheet's last line: the first guess

SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

HEAD = ("Answer every numbered line.\n"
        "Lines 1-%d: later, two secrets are still possible. Which would you guess first, A or B?\n"
        "%s\n")
WITH_OPENING = ("%d) Your first guess: any four different digits.\n"
                "Reply with one line per number: 1) your letter, 2) your letter, and so on, then "
                "%d) your number.")
LETTERS_ONLY = "Reply with one line per number: 1) your letter, 2) your letter, and so on."

FIRST = "No guesses yet. Make your first guess: any four different digits. Reply with the number only."

MENU = 5


def sheet(swapped):
    lines = "\n".join("%d) A %s B %s" % ((LINE[k],) + ((k[1], k[0]) if swapped else k))
                      for k in KINDS)
    tail = LETTERS_ONLY if swapped else WITH_OPENING % (OPENING, OPENING)
    return HEAD % (len(KINDS), lines) + tail


def read_sheet(out, swapped):
    """(first guess or None, {kind: the number chosen}) from the model's reply to one sheet."""
    first, chosen = None, {}
    by_line = {v: k for k, v in LINE.items()}
    for m in re.finditer(r"(?m)^[\s*#-]*(\d{1,2})\s*[).:\-]+\s*(.*)$", out or ""):
        n, rest = int(m.group(1)), m.group(2)
        if n == OPENING and not swapped:
            for g in FOUR.findall(rest):
                if len(set(g)) == 4:
                    first = g
                    break
            continue
        kind = by_line.get(n)
        if kind is None or kind in chosen:
            continue
        shown = (kind[1], kind[0]) if swapped else kind
        hit = None
        for g in FOUR.findall(rest):          # the model wrote the number instead of its letter
            if g in shown:
                hit = g
                break
        if hit is None:
            c = re.search(r"\b([AB])\b", rest)
            if c:
                hit = shown["AB".index(c.group(1))]
        if hit is not None:
            chosen[kind] = hit
    return first, chosen


class Harness(GNHarness):
    # Shared by every instance in the process.
    _memo = {}                                # prompt -> {"done": Event, "out": reply or None}
    _memo_lock = threading.Lock()
    _sheets = None                            # {"done": Event, "first": str|None, "agreed": dict}

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

    def sheets(self):
        """Both sheets, asked once per process: the opening and the order-proof preferences."""
        cls = type(self)
        with cls._memo_lock:
            slot = cls._sheets
            owner = slot is None
            if owner:
                slot = cls._sheets = {"done": threading.Event(), "first": None, "agreed": {}}
        if not owner:
            slot["done"].wait()
            return slot
        try:
            replies = []
            for swapped in (False, True):
                out = self.call([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": sheet(swapped)}], 400)
                replies.append(read_sheet(out, swapped))
            (first, plain), (_, swapped) = replies
            slot["first"] = first
            slot["agreed"] = {k: g for k, g in plain.items() if swapped.get(k) == g}
            self._trace.append({"step": "sheets", "plain": len(plain), "swapped": len(swapped),
                                "agreed": len(slot["agreed"])})
        finally:
            slot["done"].set()
        return slot

    def prefer(self, x, y):
        """Which of two possible secrets to guess first: the model's if order-proof, else x."""
        pair, order, names = canon_set([x, y])
        g = self.sheets()["agreed"].get(tuple(pair))
        if g is None:
            return x
        self._trace.append({"step": "agreed", "pair": list(pair), "chose": g})
        return from_frame(g, order, names)

    def first(self):
        g = self.sheets()["first"]
        if g:
            return g
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
        for n in FOUR.findall(self.ask(msgs) or ""):
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
