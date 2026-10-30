"""Each call the model puts several of the still-possible secrets in order, and the harness plays them
one after another until the feedback rules the rest out, so one call buys several guesses.

verified_menu spent exactly one call per guess (5.4 per episode), and each call re-sent the AR-Bench
rules and the transcript, about 60% of its tokens. Neither is needed: every listed number already
fits all of the feedback, so the record cannot tell them apart. Played guesses are free and the
budget is 25, about four times what a consistent player needs. Calls are what the score charges for.

  * State: the set of secrets consistent with every (guess, feedback) pair, recomputed from
    env.history with the harness's own feedback function, so it knows only what the player was told.
  * Menu: as in verified_menu. A sample of the survivors, ranked by the expected number of
    survivors each would leave, best few kept, shuffled. The whole surviving set if it is that small.
  * One call, several guesses: the model puts the whole menu in order, most likely first, and the
    harness plays down that order. (Prototype, 80 live episodes: ordering 3 of 5 cost 204 tok/ep,
    ordering all 5 cost 163, one pick per call cost 311.) A pick that the feedback since has ruled out is still played
    if its feedback is not certain yet, since a guess costs nothing and still splits the
    survivors. A pick that would teach nothing is skipped, and the model is asked again only when
    its whole order has been used up.
  * Prompt: a bare list and one instruction under an empty system message; no rules, no
    transcript.

Every guess played, and the final answer, is a number the model wrote.
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
MENU = 5             # numbers offered per call
PICKS = 5            # numbers the model puts in order per call: the whole menu
POOL = 30            # surviving numbers scored to fill the menu
MAX_CALLS = 40       # hard cap per episode; a well-behaved episode uses about three
MAX_STALLS = 3       # consecutive calls that yield no playable guess before giving up


def feedback(guess, secret):
    """(in place, misplaced) that `guess` would get if `secret` were the answer."""
    bulls = sum(g == s for g, s in zip(guess, secret))
    return bulls, len(set(guess) & set(secret)) - bulls


def expected_survivors(guess, S):
    counts = {}
    for s in S:
        k = feedback(guess, s)
        counts[k] = counts.get(k, 0) + 1
    return sum(c * c for c in counts.values()) / len(S)


def numbers(text):
    """Distinct four-digit numbers, in the order written."""
    return list(dict.fromkeys(FOUR.findall(text or "")))


class Harness(GNHarness):
    def __init__(self, env, solver):
        super().__init__(env, solver)
        self._S, self._seen = list(ALL), 0
        self._rng = random.Random()

    def survivors(self):
        h = self.env.history
        while self._seen < len(h):
            g, a, b = h[self._seen]
            self._S = [s for s in self._S if feedback(g, s) == (a, b)]
            self._seen += 1
        return self._S

    def played(self, g):
        return any(g == pg for pg, _, _ in self.env.history)

    def ask(self, content, max_tokens, temperature=0.7):
        """One call on a bare user message. The empty system message replaces the chat template's
        default one, which costs 16 tokens a call and says nothing about the game."""
        return self.llm([{"role": "system", "content": ""}, {"role": "user", "content": content}],
                        temperature=temperature, max_tokens=max_tokens)

    def worth_playing(self, g):
        """A fresh number with four different digits whose feedback is not already certain."""
        if len(set(g)) < 4 or self.played(g):
            return False
        S = self.survivors()
        if g in S:
            return True                    # it could be the secret
        if self.env.remaining <= 1:
            return False                   # the last guess goes to a possible secret
        first = feedback(g, S[0])
        return any(feedback(g, s) != first for s in S[1:])

    def menu(self):
        S = self.survivors()
        if len(S) <= MENU:
            out = list(S)
            self._rng.shuffle(out)
            return out
        pool = list(S) if len(S) <= POOL else self._rng.sample(S, POOL)
        if len(S) < len(ALL):                  # before any feedback every number splits alike
            pool.sort(key=lambda g: expected_survivors(g, S))
        top = pool[:MENU]
        self._rng.shuffle(top)
        return top

    def order(self, options):
        """The model's picks from `options`, most likely first, from one call."""
        k = min(PICKS, len(options))
        head = "My secret number is one of these:\n" + "\n".join(options)
        tail = ("\nGuess which one. Reply with the number only." if k == 1 else
                f"\nGuess {k} of them, most likely first. Reply with the {k} numbers only.")
        return numbers(self.ask(head + tail, max_tokens=6 * k + 8))[:k]

    def solve(self) -> str:
        env, stalls = self.env, 0
        while env.remaining > 0 and not env.solved and len(self._trace) < MAX_CALLS:
            n = len(env.history)
            for g in self.order(self.menu()):
                if env.solved or env.remaining <= 0:
                    break
                if self.worth_playing(g):
                    env.guess(g)
            if len(env.history) > n:
                stalls = 0
                continue
            stalls += 1
            if stalls >= MAX_STALLS:
                break
        if env.solved:
            return env.history[-1][0]
        S = self.survivors()
        picks = self.order(S if len(S) <= MENU else self.menu()) if len(self._trace) < MAX_CALLS else []
        return next((g for g in picks if g in S), picks[0] if picks else
                    (env.history[-1][0] if env.history else ""))
