"""Plan queue: the model commits to a short ranked plan, and the harness keeps executing it until the
feedback invalidates it -- the model is asked again only when its plan has run dry.

Mechanism -- consultation schedule. survivor_menu asks the model once per guess, but most of those
calls re-ask a question the model already answered: when the survivors are listed, its second and
third preferences are as good after the next feedback as before, unless that feedback eliminated
them. So each consultation here returns an ordered plan, and between consultations the harness
only does what a careful player would do with a written plan: skip every entry the latest
feedback rules out, and play the next one that still fits everything. The state is the same
exact survivor set (the 5040 numbers filtered by every (guess, feedback) pair), so every planned
guess played is still consistent with all the feedback.

Two kinds of plan:
  * the opening, before any feedback: two probes that share no digit, so one call tests 8 digits.
    Probes are played as planned -- with no feedback yet there is nothing to invalidate them;
  * afterwards: up to RANK numbers chosen, in order, from a menu of at most MENU survivors.

Stub-oracle simulation of this schedule: 400/400 secrets, mean 5.5 guesses (max 8), 3.2 model
calls an episode against 4.9 for one call per guess.

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

MENU = 8                                  # survivors shown per consultation
RANK = 4                                  # plan length asked for

SYSTEM = ("You are playing a guessing game. The secret is a 4-digit number with 4 different digits "
          "(0-9). After each guess you learn how many of its digits are in the secret and in the "
          "right position, and how many are in the secret but in a different position.")
OPENING = ("Make your first two guesses now: two numbers that share no digit, so together they "
           "test 8 different digits. Reply with the two numbers only.")
REASK_MENU = "None of those is on the list. Reply with numbers from the list only."


def feedback(guess, secret):
    """(in place, elsewhere) for two numbers with distinct digits; symmetric in its arguments."""
    a = (guess[0] == secret[0]) + (guess[1] == secret[1]) + (guess[2] == secret[2]) \
        + (guess[3] == secret[3])
    return a, _POP[_MASK[guess] & _MASK[secret]] - a


def record(history):
    return "\n".join(f"{g}: {a} in the right position, {b} in a different position"
                     for g, a, b in history)


def all_in(text, alive, limit):
    """Numbers in the reply, in order and deduplicated, that are in `alive`."""
    out = []
    for m in FOUR.findall(text or ""):
        if m in alive and m not in out:
            out.append(m)
            if len(out) == limit:
                break
    return out


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        rng = random.Random(0)
        survivors = list(ALL)
        last = None

        for probe in self._opening():             # the opening plan: played as written
            if env.solved or env.remaining <= 0:
                break
            a, b = env.guess(probe)
            last = probe
            survivors = [s for s in survivors if feedback(probe, s) == (a, b)]

        plan = []
        while not env.solved and env.remaining > 0 and survivors:
            alive = set(survivors)
            plan = [p for p in plan if p in alive]   # entries the feedback has ruled out are dropped
            if len(survivors) == 1:
                guess = survivors[0]                 # forced: the feedback has fixed the answer
            else:
                if not plan:
                    plan = self._rank(survivors, rng)
                guess = plan.pop(0)
            a, b = env.guess(guess)
            last = guess
            survivors = [s for s in survivors if feedback(guess, s) == (a, b)]
        if env.solved:
            return last
        if len(survivors) > 1:                       # budget spent: the model's plan picks the answer
            return self._rank(survivors, rng)[0]
        return survivors[0] if survivors else (last or "0000")

    def _opening(self):
        out = self.llm([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": OPENING}], max_tokens=32)
        return all_in(out, set(ALL), 2)              # a malformed reply leaves the plan short

    def _rank(self, survivors, rng):
        """One model call: an ordered plan of up to RANK survivors (re-asked once if off-menu)."""
        menu = survivors if len(survivors) <= MENU else rng.sample(survivors, MENU)
        user = (f"Guesses so far:\n{record(self.env.history)}\n\n"
                f"Numbers that agree with all the feedback ({len(menu)} of {len(survivors)}): "
                f"{', '.join(menu)}\n"
                f"Pick up to {RANK} of these, in the order you would try them. "
                "Reply with the numbers only, separated by spaces.")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        alive = set(survivors)
        for _ in range(2):
            out = self.llm(msgs, max_tokens=48)
            plan = all_in(out, alive, RANK)
            if plan:
                return plan
            msgs = msgs + [{"role": "assistant", "content": out or ""},
                           {"role": "user", "content": REASK_MENU}]
        return [menu[0]]
