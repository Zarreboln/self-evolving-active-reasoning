"""The model writes eight guesses in one call before seeing any feedback; the harness plays them,
works out what they leave possible, and only then asks the model to pick from that.

Every model call costs tokens whatever it says, while a played guess costs none, and the 25-guess
budget is about three times what a consistent player needs. So this harness spends guesses instead
of calls. One call asks the model for a covering design: eight numbers that together use every digit
and never repeat a digit in a position, which the 7B model writes readily. All eight are played
before the model is consulted again. Eight well-spread guesses almost pin the secret down. On the
plan the model writes, one secret is left for 83% of possible secrets and at most two for 98%, so
one more call, a pick from what is left, usually ends the game.

  * State: the set of secrets consistent with every (guess, feedback) pair, recomputed from
    env.history with the harness's own feedback function, so it knows only what the player was told.
  * Plan play: planned numbers are played in the model's order. A number with a repeated digit, a
    repeat of an earlier guess, or one whose feedback is already certain is skipped, since it would
    teach nothing.
  * Endgame: the surviving secrets (the most informative few, if many survive) as a bare list, and
    the model picks one. The rules and the transcript are left out: every listed number fits all of
    the feedback, so neither can tell the listed numbers apart, and in verified_menu they were about
    60% of every call.

If the plan comes back short or unreadable, the endgame loop just has more to do. It can play a
whole game from no information. Every guess played, and the final answer, is a number the model
wrote.
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
PLAN = 8             # guesses the model writes before any feedback
MENU = 5             # numbers offered per endgame turn
POOL = 30            # surviving numbers scored to fill the menu
MAX_CALLS = 40       # hard cap per episode; a well-behaved episode uses about two
MAX_STALLS = 3       # consecutive turns without a playable pick before giving up

PLAN_PROMPT = (f"Write {PLAN} four-digit numbers, each with 4 different digits, that together use every "
               "digit 0-9 and never put the same digit in the same position twice. "
               f"Reply with the {PLAN} numbers only.")


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
        first = feedback(g, S[0])
        return any(feedback(g, s) != first for s in S[1:])

    def play_plan(self):
        out = self.ask(PLAN_PROMPT, max_tokens=6 * PLAN + 8)
        for g in numbers(out)[:PLAN]:
            if self.env.solved or self.env.remaining <= 1:
                break
            if self.worth_playing(g):
                self.env.guess(g)

    def menu(self):
        S = self.survivors()
        if len(S) <= MENU:
            return list(S)
        pool = list(S) if len(S) <= POOL else self._rng.sample(S, POOL)
        if len(S) < len(ALL):                  # before any feedback every number splits alike
            pool.sort(key=lambda g: expected_survivors(g, S))
        top = pool[:MENU]
        self._rng.shuffle(top)
        return top

    def choose(self, options):
        """The model's pick, asked twice at most. Returns a playable pick, else its last answer."""
        S = self.survivors()
        head = "My secret number is one of these:\n" + "\n".join(options)
        last = None
        for attempt in range(2):
            if len(self._trace) >= MAX_CALLS:
                break
            tail = ("\nGuess which one. Reply with the number only." if attempt == 0
                    else "\nReply with one number from the list only.")
            for g in numbers(self.ask(head + tail, max_tokens=16, temperature=0.7 + 0.3 * attempt)):
                last = g
                if g in S and not self.played(g):
                    return g
        return last

    def solve(self) -> str:
        env = self.env
        self.play_plan()
        stalls = 0
        while env.remaining > 0 and not env.solved and len(self._trace) < MAX_CALLS:
            g = self.choose(self.menu())
            if g is None or not self.worth_playing(g):
                stalls += 1
                if stalls >= MAX_STALLS:
                    break
                continue
            stalls = 0
            env.guess(g)
        if env.solved:
            return env.history[-1][0]
        S = self.survivors()
        return self.choose(list(S) if len(S) <= MENU else self.menu()) or \
            (env.history[-1][0] if env.history else "")
