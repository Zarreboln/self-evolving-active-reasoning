"""Survivor menu: the harness keeps the exact set of numbers that fit every feedback, and the model
picks each guess from a short menu of them.

Mechanism -- state + record shape. AR-Bench's loop hands the model a transcript and leaves the
constraint reasoning to it; a 7B model cannot do that reasoning (probe: 0 of 24 guesses it drafted
from an explicit per-position constraint table fit the feedback). UoT fixes this by showing the
model the whole answer set, which is why it costs ~800k tokens an episode. Here the harness does
the elimination exactly -- the 5040 numbers filtered by every (guess, feedback) pair -- and shows
the model at most MENU survivors with the instruction to pick one. Every played guess is then
consistent with everything learned so far, and consistent play identifies the secret within ~8
guesses whichever survivor is picked (stub-oracle simulation: 400/400 secrets, mean 5.5, max 8).

The prompt is a compact rule statement plus a one-line-per-guess record, not AR-Bench's Game_rule
and transcript: the model is choosing among survivors, so the verbose record is cost without use.

Stopping: when one number survives it is played without a model call; nothing is left to choose.
"""
from __future__ import annotations

import itertools
import random
import re

from ..harness_base import GNHarness

FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
ALL = ["".join(p) for p in itertools.permutations("0123456789", 4)]
_MASK = {c: sum(1 << int(d) for d in c) for c in ALL}
_POP = [bin(i).count("1") for i in range(1024)]

MENU = 8                                  # survivors shown per turn

SYSTEM = ("You are playing a guessing game. The secret is a 4-digit number with 4 different digits "
          "(0-9). After each guess you learn how many of its digits are in the secret and in the "
          "right position, and how many are in the secret but in a different position.")
FIRST = "Make your first guess. Reply with the number only."
REASK_FIRST = "Reply with one 4-digit number whose 4 digits are all different."
REASK_MENU = "That is not one of the listed numbers. Reply with exactly one number from the list."


def feedback(guess, secret):
    """(in place, elsewhere) for two numbers with distinct digits; symmetric in its arguments."""
    a = (guess[0] == secret[0]) + (guess[1] == secret[1]) + (guess[2] == secret[2]) \
        + (guess[3] == secret[3])
    return a, _POP[_MASK[guess] & _MASK[secret]] - a


def record(history):
    return "\n".join(f"{g}: {a} in the right position, {b} in a different position"
                     for g, a, b in history)


def first_in(text, alive):
    """The first number in the reply that is still possible, else None."""
    text = text or ""
    for m in FOUR.findall(text):
        if m in alive:
            return m
    digits = re.sub(r"\D", "", text)
    return digits if digits in alive else None


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        rng = random.Random(0)
        survivors = list(ALL)
        last = None
        while not env.solved and env.remaining > 0 and survivors:
            if len(survivors) == 1:
                guess = survivors[0]              # forced: the feedback has fixed the answer
            else:
                guess = self._choose(survivors, rng)
            a, b = env.guess(guess)
            last = guess
            survivors = [s for s in survivors if feedback(guess, s) == (a, b)]
        if env.solved:
            return last
        if len(survivors) > 1:                    # budget spent: the model still picks the answer
            return self._choose(survivors, rng)
        return survivors[0] if survivors else (last or "0000")

    def _choose(self, survivors, rng):
        """One model call: pick the next guess from a menu of survivors (re-asked once if off-menu)."""
        hist = self.env.history
        menu = None
        if not hist:
            user = FIRST
        else:
            menu = survivors if len(survivors) <= MENU else rng.sample(survivors, MENU)
            user = (f"Guesses so far:\n{record(hist)}\n\n"
                    f"Numbers that agree with all the feedback ({len(menu)} of {len(survivors)}): "
                    f"{', '.join(menu)}\n"
                    "Your next guess must be one of these. Reply with the number only.")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        alive = set(survivors)
        for _ in range(2):
            out = self.llm(msgs, max_tokens=24)
            pick = first_in(out, alive)
            if pick:
                return pick
            msgs = msgs + [{"role": "assistant", "content": out or ""},
                           {"role": "user", "content": REASK_MENU if menu else REASK_FIRST}]
        return menu[0] if menu else rng.choice(survivors)
