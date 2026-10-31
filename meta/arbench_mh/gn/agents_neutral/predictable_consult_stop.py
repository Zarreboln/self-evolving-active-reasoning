"""The model is consulted until its replies become predictable, then a fitted rule answers for it.

Mechanism (axis E, when to stop consulting the model, with B, what carries across episodes: a
rule fitted to the model's replies rather than a table of them). pairwise_canon_tourney keeps
belief in code (the secrets consistent with every feedback), ranks the survivors by expected
survivors, and decides the top five by a knockout of two-candidate questions, each written
canonically up to digit renaming and position reordering and cached across episodes. It paid
for 20 prompts in its whole split, one per kind of pair met, and in the probe its replies looked
like a habit of position (B five times in six) more than a judgement about the two numbers.

A reply that a cheap rule predicts carries no information the rule does not, so here the harness
learns the rule and stops paying for the reply. Every fresh consultation is logged process-wide
with the model's choice and what each of four rules would have chosen: the first or the second
number as shown, or the knockout's favourite (the candidate code ranks better) or its challenger.
Once one rule has matched the model on each of the last STREAK fresh consultations, consultation
stops: a new kind of pair is answered by that rule, except every AUDIT-th new kind, which is
still put to the model. A disagreement on an audit drops the rule and consultation resumes until
some rule earns a new streak. Kinds already answered by the model keep its cached reply. If no
rule ever earns a streak, every kind is asked, exactly as in the base.

Stub (100 random secrets, 16 threads, four stub models; tokens estimated by counting each digit,
word and mark as one token, which puts the base at 18.7 against its measured 20.1). All four
stubs solved 100%, 5.3-5.5 guesses, max 8. The base made 19-20 calls whatever the stub. Here, with
a stub that always names B, 8 calls (7.4 tokens/ep) and the rule "shown second" adopted after 4;
with one that names B five times in six (the real model's rate in the base's probe), 10 calls
(9.3); with one that names the smaller number, 8 calls; with letters no rule predicts, 20 calls
(18.7) and no rule adopted. Variants compared: without the 80% overall-agreement condition a rule
was adopted by chance on the unpredictable stub (11 calls, its answers matching the stub about
60% of the time); auditing every third new kind instead of every fourth cost 1-4 more calls on
the predictable stubs and changed nothing else. Dropping two more rules, the smaller or the
larger real number, which the model never sees, changed no stub's numbers. No real-model probe:
the cluster login was down this session.

Nothing is learned about any secret: the log holds the model's replies to prompts and starts
empty in every process, so the first episode is played exactly as it would be without it.
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
# The rules a reply is checked against, in the order a tie between them is broken. The model sees
# only the renamed pair, so the rules read what it sees (which line a number is on) or what code
# thinks (the knockout's favourite); a rule on the real numbers could only match by coincidence.
RULES = ("shown_second", "shown_first", "favourite", "challenger")
STREAK = 4      # fresh consultations in a row a rule must match before it answers for the model
AGREE = 0.8     # ... and the share of all fresh consultations it must match
AUDIT = 4       # once it does, every AUDIT-th new kind of pair is still put to the model


class Harness(GNHarness):
    # Shared by every instance in the process.
    _memo = {}                  # prompt -> {"done": Event, "out": reply or None}
    _memo_lock = threading.Lock()
    _log = []                   # one {rule: matched?} per fresh consultation, in order
    _rule = None                # the rule answering for the model, or None while consulting
    _ruled = set()              # kinds of pair the rule has answered without asking
    _new_since = 0              # new kinds met since the rule was adopted

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
            return self.llm(messages, temperature=0.0, max_tokens=8)
        except Exception:                     # endpoint down: the code's own ranking decides
            return None

    def ask(self, messages):
        """(reply, fresh): the model at temperature 0, asked once per distinct prompt per process."""
        key = tuple((m["role"], m["content"]) for m in messages)
        cls = type(self)
        with cls._memo_lock:
            slot = cls._memo.get(key)
            owner = slot is None
            if owner:
                slot = cls._memo[key] = {"done": threading.Event(), "out": None}
        if owner:
            try:
                slot["out"] = self.call(messages)
            finally:
                if slot["out"] is None:       # the call failed: let the next asker retry
                    with cls._memo_lock:
                        cls._memo.pop(key, None)
                slot["done"].set()
            return slot["out"], True
        slot["done"].wait()
        if slot["out"] is None:
            return self.call(messages), True
        self._trace.append({"step": "llm_memo", "messages": [dict(m) for m in messages],
                            "response": slot["out"]})
        return slot["out"], False

    @staticmethod
    def pick(out, pair):
        """The model's choice: an item it wrote out, else the letter it named, else None."""
        for n in FOUR.findall(out or ""):
            if n in pair:
                return n
        m = re.search(r"\b([AB])\b", out or "")
        return pair["AB".index(m.group(1))] if m else None

    def learn(self, would, chose):
        """Log a fresh consultation; adopt a rule on a streak, drop it on a failed audit."""
        cls = type(self)
        with cls._memo_lock:
            cls._log.append({r: would[r] == chose for r in RULES})
            if cls._rule is not None:         # this consultation was an audit
                if not cls._log[-1][cls._rule]:
                    self._trace.append({"step": "rule_dropped", "rule": cls._rule})
                    cls._rule = None
                return
            tail = cls._log[-STREAK:]
            if len(tail) < STREAK:
                return
            for r in RULES:
                agree = sum(e[r] for e in cls._log) / len(cls._log)
                if all(e[r] for e in tail) and agree >= AGREE:
                    cls._rule, cls._new_since = r, 0
                    self._trace.append({"step": "rule_adopted", "rule": r})
                    return

    def prefer(self, x, y):
        """Which of two possible secrets to guess first: x is the knockout's favourite so far."""
        pair, order, names = canon_set([x, y])
        shown = [from_frame(p, order, names) for p in pair]
        would = {"shown_first": shown[0], "shown_second": shown[1], "favourite": x,
                 "challenger": y}
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PAIR % tuple(pair)}]
        key = tuple((m["role"], m["content"]) for m in msgs)
        cls = type(self)
        with cls._memo_lock:
            rule = None if key in cls._memo else cls._rule
            if rule is not None and key not in cls._ruled:
                cls._new_since += 1
                if cls._new_since % AUDIT == 0:
                    rule = None               # an audit: this kind is put to the model
                else:
                    cls._ruled.add(key)
        if rule is not None:
            self._trace.append({"step": "rule", "rule": rule, "pair": list(pair),
                                "as_letter": "AB"[shown.index(would[rule])]})
            return would[rule]
        out, fresh = self.ask(msgs)
        g = self.pick(out, pair)
        if g is None:
            out, _ = self.ask(msgs + [{"role": "assistant", "content": out or ""},
                                      {"role": "user", "content": "Reply with one letter, A or B."}])
            g = self.pick(out, pair)
        if g is None:
            return x
        chose = from_frame(g, order, names)
        if fresh:
            self.learn(would, chose)
        return chose

    def first(self):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
        out, _ = self.ask(msgs)
        for n in FOUR.findall(out or ""):
            if len(set(n)) == 4:
                return n
        out, _ = self.ask(msgs + [{"role": "assistant", "content": out or ""},
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
