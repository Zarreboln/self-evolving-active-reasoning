"""The model writes its second move for every possible feedback to its first, as one table.

Mechanism (axis A, where guesses come from, with C, a plan fixed in advance -- but one that is
conditioned on feedback, a policy rather than a script). The menu harnesses ask the model about
one state at a time, and the state is new in almost every episode, so the cost is one prompt per
kind of question met. The pairwise tourney cut that to 20 prompts per split by showing two
numbers with no context, at the price of asking a question the model mostly answered by position.

The first branching of the game is small and known before any episode starts. Once the model's
opening is fixed (asked at temperature 0, the same for every episode in a process), its feedback
can take only thirteen values short of a win, and each value leaves a known set of possible
secrets. So the model is shown all of them at once: one row per feedback, each with the three
possible secrets that leave the fewest survivors on average (fewer if fewer remain), and it picks
a letter per row. The table is asked once per process and shared by every episode. The second
guess is read from the row the real feedback selects; every later guess is the possible secret
with the fewest expected survivors, and a lone survivor is played without asking. A row the reply
leaves unreadable, or an endpoint that raises, falls back to the row's first item.

So the model makes the contextual decision at the game's widest branching -- it sees the
feedback when it chooses -- and the whole split pays for two prompts.

Stub (100 random secrets, 16 threads; tokens estimated by counting each digit, word and mark as
one token, which puts pairwise_canon_tourney at 18.7 against its measured 20.1): 2 calls per
process, 6.7 tokens/ep, 100% solved, 5.35-5.47 guesses (max 7) whether the stub answers A, B or
letters at random. With garbage replies: 3 calls (the opening re-asked), every row on code's first
item, 100%. With an endpoint that raises: 0 calls, 100%. The table for opening 1234 has 13 rows of
three options, about 540 input tokens. No real-model probe: the cluster login was down this
session, so whether the 7B model answers all 13 rows is unmeasured; rows it skips fall to code.

Nothing is learned about any secret: the table holds the model's replies to prompts built from
its own opening, starts empty in every process, and is the same for the first episode as for
the last.
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

TABLE = ("Your first guess is %s. Below is every feedback it can get. For each one, the options "
         "are secrets that would still be possible. Pick the second guess you would make.\n"
         "%s\n"
         "Reply with one line per row: 1) your letter, 2) your letter, and so on.")

ROW = "%d) %d in the right place, %d in a different place: %s"

MENU = 3
LETTERS = "ABC"


def read_table(out, rows):
    """{row number: chosen number} for the rows the reply answers readably."""
    got = {}
    menus = {i + 1: menu for i, (_, menu) in enumerate(rows)}
    for m in re.finditer(r"(?m)^[\s*#-]*(\d{1,2})\s*[).:\-]+\s*(.*)$", out or ""):
        n, rest = int(m.group(1)), m.group(2)
        menu = menus.get(n)
        if menu is None or n in got:
            continue
        hit = None
        for g in FOUR.findall(rest):           # the model wrote the number instead of its letter
            if g in menu:
                hit = g
                break
        if hit is None:
            c = re.search(r"\b([A-C])\b", rest)
            if c and LETTERS.index(c.group(1)) < len(menu):
                hit = menu[LETTERS.index(c.group(1))]
        if hit is not None:
            got[n] = hit
    # Rows answered on one line ("1) A 2) C ..."): read every "number) letter" left unread.
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s*[).:]\s*\(?([A-C])\b", out or ""):
        n, letter = int(m.group(1)), m.group(2)
        menu = menus.get(n)
        if menu is not None and n not in got and LETTERS.index(letter) < len(menu):
            got[n] = menu[LETTERS.index(letter)]
    return got


class Harness(GNHarness):
    # Shared by every instance in the process.
    _memo = {}                  # prompt -> {"done": Event, "out": reply or None}
    _memo_lock = threading.Lock()
    _tables = {}                # opening -> {"done": Event, "second": {(a, b): guess}}

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

    def table(self, opening):
        """{feedback: second guess} for this opening, built and asked once per process."""
        cls = type(self)
        with cls._memo_lock:
            slot = cls._tables.get(opening)
            owner = slot is None
            if owner:
                slot = cls._tables[opening] = {"done": threading.Event(), "second": {}}
        if not owner:
            slot["done"].wait()
            return slot["second"]
        try:
            groups = {}
            for s in SECRETS:
                if s != opening:
                    groups.setdefault(score(opening, s), []).append(s)
            second, rows = {}, []
            for fb in sorted(groups):
                menu = rank(groups[fb])[:MENU]
                second[fb] = menu[0]
                if len(menu) > 1:
                    rows.append((fb, menu))
            lines = "\n".join(
                ROW % (i + 1, fb[0], fb[1],
                       " ".join("%s) %s" % (LETTERS[j], g) for j, g in enumerate(menu)))
                for i, (fb, menu) in enumerate(rows))
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": TABLE % (opening, lines)}]
            got = read_table(self.ask(msgs, max_tokens=12 * len(rows) + 16), rows)
            for i, (fb, _) in enumerate(rows):
                if i + 1 in got:
                    second[fb] = got[i + 1]
            self._trace.append({"step": "table", "rows": len(rows), "answered": len(got)})
            slot["second"] = second
        finally:
            slot["done"].set()
        return slot["second"]

    def choose(self, alive):
        hist = self.env.history
        if not hist:
            return self.first()
        played = {g for g, _, _ in hist}
        live = [s for s in alive if s not in played] or alive
        if len(live) == 1:                    # nothing to choose between
            return live[0]
        if len(hist) == 1:
            g, a, b = hist[0]
            pick = self.table(g).get((a, b))
            if pick in live:
                return pick
        return rank(live)[0]
