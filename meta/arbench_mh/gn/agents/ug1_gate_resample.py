"""AR-Bench's own loop, with every proposal checked against the feedback before it is played.

The model sees exactly what the zero-shot loop shows it (env.turn_prompt(), which asks for three
numbers) and does all the reasoning itself. The harness changes one thing: a proposal is played
only if it could still be the secret -- four different digits, not guessed before, and scoring
exactly what every earlier guess scored had it been the answer. The check applies the oracle's
counting rule to hypotheses; it never touches the target.

When none of the three passes, the model is told why in the same conversation, one concrete line
per number ("if <n> were the secret, guess 2 would have scored 2 in place and 0 misplaced; it
scored 1 and 1"), and asked again, up to RETRIES times. If it still has nothing that fits, its own
first fresh proposal is played anyway, so every guess on the record is one the model wrote.

Hypothesis (iteration 1): the zero-shot loop loses most episodes by spending guesses that earlier
feedback already rules out, and a model told precisely which line a number breaks can repair it.
If so, gating alone lifts accuracy well above the zero-shot loop; if accuracy stays near zero,
the model cannot produce consistent numbers even when shown its contradictions, and the belief
state has to be given to it rather than checked after the fact (see ug1_fact_digest,
ug1_menu_pick).

Prototyped against stub models only; the cluster login had expired, so no real-model probe was
possible this session. On the real GNEnv over 100 secrets, a stub writing three random numbers
per call won 3% and used about 73 calls (roughly 85k estimated tokens) per episode, which is the
cost ceiling of this design; a stub writing a consistent number 30% of the time won 100% in 5.5
guesses and 7.4 calls (about 3.8k tokens). So the result on the real model reads directly as how
often Qwen can write a number that fits.
"""
from __future__ import annotations

import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
_MASK = {n: sum(1 << int(c) for c in n) for n in ALL}
_POP = [bin(i).count("1") for i in range(1024)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
GUESS_LINE = re.compile(r"guess", re.I)

RETRIES = 2


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


def proposals(text):
    """Four-digit numbers in the order the model gave them, from its last 'Guess' line if it has
    one (so numbers quoted from the record while reasoning are not mistaken for proposals)."""
    lines = [ln for ln in (text or "").splitlines() if GUESS_LINE.search(ln) and FOUR.search(ln)]
    found = FOUR.findall(lines[-1] if lines else (text or ""))
    seen, out = set(), []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def why_not(n, history):
    """One sentence saying why `n` cannot be the secret, or None if it can."""
    if len(set(n)) < 4:
        return "%s repeats a digit, but the secret's four digits are all different." % n
    for i, (g, a, b) in enumerate(history):
        if g == n:
            return "%s was already guessed (guess %d) and it was not the secret." % (n, i + 1)
    for i, (g, a, b) in enumerate(history):
        x, y = score(g, n)
        if (x, y) != (a, b):
            return ("if %s were the secret, guess %d (%s) would have scored %d in the correct "
                    "position and %d in a different position, but it scored %d and %d."
                    % (n, i + 1, g, x, y, a, b))
    return None


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        last = None
        for _ in range(env.budget):
            if env.solved or env.remaining <= 0:
                break
            fits = set(consistent(env.history))
            played = {g for g, _, _ in env.history}
            msgs = [{"role": "user", "content": env.turn_prompt()}]
            pick = fallback = None
            for attempt in range(1 + RETRIES):
                out = self.llm(msgs) or ""
                props = proposals(out)
                fresh = [n for n in props if len(set(n)) == 4 and n not in played]
                if fallback is None and fresh:
                    fallback = fresh[0]
                ok = [n for n in fresh if n in fits]
                if ok:
                    pick = ok[0]
                    break
                if attempt == RETRIES:
                    break
                reasons = [why_not(n, env.history) for n in props] if props else []
                note = ("\n".join("- " + r for r in reasons if r) if reasons else
                        "- No four-digit number was found in your reply.")
                msgs += [{"role": "assistant", "content": out},
                         {"role": "user", "content":
                          "None of those can be the secret:\n%s\n\nSuggest 3 different numbers "
                          "that agree with every feedback line in the record. Your output should "
                          "be only in this format:\nGuess: [num1], [num2], [num3]" % note}]
            pick = pick or fallback
            if pick is None:
                continue                         # nothing playable written: the turn is spent
            env.guess(pick)
            last = pick
        if env.solved:
            return last
        return self.final(last)

    def final(self, last):
        """Out of guesses: AR-Bench's final prompt, with the same check and one retry."""
        env = self.env
        fits = set(consistent(env.history))
        msgs = [{"role": "user", "content": env.final_prompt}]
        answer = None
        for attempt in range(2):
            out = self.llm(msgs) or ""
            nums = [n for n in FOUR.findall(out) if len(set(n)) == 4]
            if nums and answer is None:
                answer = nums[-1]
            ok = [n for n in nums if n in fits]
            if ok:
                return ok[-1]
            r = why_not(nums[-1], env.history) if nums else "No four-digit number was found."
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": "That cannot be the secret: %s Answer again with "
                      "only a 4-digit number." % r}]
        return answer or last or ""
