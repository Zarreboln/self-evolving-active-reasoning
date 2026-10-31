"""Guess Number environment: the oracle a harness is allowed to reach, and nothing else.

AR-Bench's own prompts and scoring function are imported rather than restated, so a harness that
does nothing clever reproduces AR-Bench's zero-shot loop exactly. The target is private: the only
way to learn anything about it is to spend a guess.
"""
from __future__ import annotations

import os
import random
import sys
from itertools import permutations

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.gn.prompt import (  # noqa: E402
    Final_prompt, Game_rule, Guess_number_prompt,
)
from arbench.utils.utils_gn import compare_guess  # noqa: E402

BUDGET = 25

# AREW's (ICML 2026) intervention, adapted: take the information out of the feedback and change
# nothing else. What the score then loses is the causal contribution of the interaction -- a
# harness that asks well but does not listen loses little, and that is the self-locking symptom.
#
# AREW replaces the answer with the word "Unknown", which works because its feedback is prose.
# Here it does not. Feedback reaches a harness through TWO channels, the record text and the
# structured history, and blanking only the text leaves the second one carrying the real answer:
# a harness reading env.history directly scored 100% "ablated". Blanking the structured channel
# too is no better: a harness that computes anything from it is left with nothing and crashes,
# which scores zero for the wrong reason.
#
# So the oracle is replaced instead of the message: for the whole episode, feedback is computed
# against a DECOY secret. Every channel is blinded at once, the feedback stays well formed and
# self-consistent, the harness runs its machinery to the end and answers confidently -- and what
# it answers carries no information about the real target. Scoring still uses the real one.
ABLATE = os.environ.get("ABLATE_FEEDBACK") == "1"


class BudgetExhausted(Exception):
    pass


class GNEnv:
    def __init__(self, target, budget=BUDGET):
        self._target = [int(d) for d in str(target).zfill(4)]
        # Under ablation the oracle answers about a different secret, drawn once per episode and
        # independent of the real one. self._target is kept, and only score() reads it.
        self._oracle = self._target
        if ABLATE:
            # Seeded by the target, so the decoy is the same on every re-run of the same episode
            # and the ablated column is reproducible.
            pool = [list(p) for p in permutations(range(10), 4) if list(p) != self._target]
            self._oracle = random.Random("".join(map(str, self._target))).choice(pool)
        self.budget = budget
        self.history = []            # [(guess, a, b)]
        self.record = ""             # the feedback text, in AR-Bench's wording
        self.solved = False
        self.system_prompt = Game_rule

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def turn_prompt(self, t=None):
        return Game_rule + Guess_number_prompt.format(guess_record=self.record)

    @property
    def final_prompt(self):
        return Game_rule + Final_prompt.format(guess_record=self.record)

    def guess(self, number):
        """Play one guess. Returns (digits in place, digits present but misplaced)."""
        if self.remaining <= 0:
            raise BudgetExhausted()
        g = number if isinstance(number, str) else "".join(str(d) for d in number)
        a, b, _ = compare_guess(self._oracle, g)
        self.history.append((g, a, b))
        self.record += (
            f"The guess number {len(self.history)} is {g}, and the feedback is: "
            f"{a} digits are present in the answer and in the correct positions, "
            f"{b} digits are present in the answer but in the different positions.\n")
        if a == 4:
            self.solved = True
        return a, b

    def score(self, prediction):
        """Exact match, AR-Bench's metric: the final guess is the answer."""
        g = prediction if isinstance(prediction, str) else "".join(str(d) for d in (prediction or []))
        return {"correctness": bool(g) and [int(d) for d in g.zfill(4)] == self._target,
                "guess_round": len(self.history)}
