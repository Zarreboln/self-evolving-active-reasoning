"""The harness works out what the feedback proves; the model composes the guess from those facts.

Every number still consistent with all the feedback is enumerated exactly (the oracle's counting
rule applied to hypotheses, never to the target). The model is NOT shown those numbers. It is
shown what they have in common, which is what a careful human player would have written in the
margin: how many remain, which digits are certainly in or certainly out, and for every digit that
might still be in, the positions it can occupy. From that it writes a guess.

A guess that does not fit is sent back once or twice with the specific reason -- the fact it
breaks, or the feedback line it contradicts -- and, failing that, the model's own last fresh
number is played, so every guess on the record is one the model wrote.

Hypothesis (iteration 1): a 7B model cannot track the joint constraint set over several feedback
lines, but it can satisfy a handful of per-digit, per-position facts when they are spelled out.
If so, accuracy lands far above a gate-only loop (ug1_gate_resample) at a few thousand tokens;
if it lands near the gate-only loop, marginal facts are not enough and only the joint set (a
menu of the numbers themselves, ug1_menu_pick) helps.

Prototyped against stub models only; the cluster login had expired, so no real-model probe was
possible this session. On the real GNEnv over 100 secrets, a stub that obeys the stated facts
30% / 70% / 100% of the time (random number otherwise) solved 100% each time, in 6.6 / 5.9 /
5.8 guesses and 15 / 11 / 10 calls (about 6.0k / 4.4k / 4.0k estimated tokens). Even full
obedience to the marginal facts needs retries, because facts about single digits do not pin
down a number that fits every feedback line jointly; the gate catches those.
"""
from __future__ import annotations

import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
_MASK = {n: sum(1 << int(c) for c in n) for n in ALL}
_POP = [bin(i).count("1") for i in range(1024)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
POS = ("first", "second", "third", "fourth")

RETRIES = 2

SYSTEM = ("You are playing a number-guessing game. The secret is a 4-digit number whose digits are "
          "all different. After each guess you are told how many digits are right and in the right "
          "place, and how many are right but in the wrong place.")


def _mask(n):
    m = _MASK.get(n)
    return m if m is not None else sum(1 << int(c) for c in set(n))


def score(guess, secret):
    """(in place, misplaced) if `secret` were the answer -- the oracle's rule, on a hypothesis."""
    a = sum(1 for x, y in zip(guess, secret) if x == y)
    return a, _POP[_mask(guess) & _mask(secret)] - a


def consistent(history):
    s = ALL
    for g, a, b in history:
        s = [n for n in s if score(g, n) == (a, b)]
    return s


def record(history):
    if not history:
        return "none yet"
    return "\n".join("%d. %s: %d in the right place, %d right but misplaced" % (i + 1, g, a, b)
                     for i, (g, a, b) in enumerate(history))


def facts(fits):
    """What every remaining candidate agrees on, as short sentences."""
    where = {d: sorted({n.index(d) for n in fits if d in n}) for d in "0123456789"}
    always = [d for d in "0123456789" if fits and all(d in n for n in fits)]
    never = [d for d in "0123456789" if not where[d]]
    lines = ["- %d numbers still fit all the feedback." % len(fits) if len(fits) != 1 else
             "- Exactly one number still fits all the feedback."]
    lines.append("- Certainly in the secret: %s." % (", ".join(always) or "none known yet"))
    lines.append("- Certainly NOT in the secret: %s." % (", ".join(never) or "none known yet"))
    for d in "0123456789":
        if where[d] and len(where[d]) < 4:
            lines.append("- If %s is in the secret, it can only be in the %s position."
                         % (d, " or ".join(POS[i] for i in where[d]))
                         if d not in always else
                         "- %s is in the secret, in the %s position."
                         % (d, " or ".join(POS[i] for i in where[d])))
    for i in range(4):
        col = sorted({n[i] for n in fits})
        if len(col) < 10:
            lines.append("- The %s digit is one of: %s." % (POS[i], ", ".join(col)))
    return "\n".join(lines)


def why_not(n, history, fits_facts):
    """One sentence saying why `n` cannot be the secret, or None if it can."""
    if len(set(n)) < 4:
        return "%s repeats a digit, but the secret's four digits are all different." % n
    for i, (g, a, b) in enumerate(history):
        if g == n:
            return "%s was already guessed (guess %d) and it was not the secret." % (n, i + 1)
    always, where = fits_facts
    for d in always:
        if d not in n:
            return "%s leaves out %s, which is certainly in the secret." % (n, d)
    for i, d in enumerate(n):
        if i not in where[d]:
            return ("%s puts %s in the %s position, which the feedback rules out."
                    % (n, d, POS[i]) if where[d] else
                    "%s uses %s, which is certainly not in the secret." % (n, d))
    for i, (g, a, b) in enumerate(history):
        x, y = score(g, n)
        if (x, y) != (a, b):
            return ("if %s were the secret, guess %d (%s) would have scored %d in the right place "
                    "and %d misplaced, but it scored %d and %d." % (n, i + 1, g, x, y, a, b))
    return None


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        last = None
        for _ in range(env.budget):
            if env.solved or env.remaining <= 0:
                break
            pick = self.compose(final=False)
            if pick is None:
                continue                         # nothing playable written: the turn is spent
            env.guess(pick)
            last = pick
        if env.solved:
            return last
        return self.compose(final=True) or last or ""

    def compose(self, final):
        hist = self.env.history
        played = {g for g, _, _ in hist}
        fits = consistent(hist)
        fitset = set(fits)
        where = {d: {n.index(d) for n in fits if d in n} for d in "0123456789"}
        always = [d for d in "0123456789" if fits and all(d in n for n in fits)]
        ask = ("You have no guesses left. Give your final answer: the 4-digit number you think is "
               "the secret, which must fit every fact above." if final else
               "Write your next guess: a 4-digit number with all digits different that fits every "
               "fact above.")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Guesses so far:\n%s\n\nWhat the feedback proves "
                 "(worked out exactly from every guess above):\n%s\n\n%s Reply with the number "
                 "only." % (record(hist), facts(fits), ask)}]
        fallback = None
        for attempt in range(1 + RETRIES):
            out = self.llm(msgs, max_tokens=24) or ""
            nums = [n for n in FOUR.findall(out) if len(set(n)) == 4]
            fresh = [n for n in nums if final or n not in played]
            if fresh:
                fallback = fresh[-1]
            ok = [n for n in fresh if n in fitset]
            if ok:
                return ok[-1]
            if attempt == RETRIES:
                break
            raw = FOUR.findall(out)
            r = why_not(raw[-1], hist, (always, where)) if raw else None
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": "That cannot be the secret: %s Write another "
                      "number that fits every fact. Reply with the number only."
                      % (r or "no 4-digit number with all digits different was found in your "
                         "reply.")}]
        return fallback
