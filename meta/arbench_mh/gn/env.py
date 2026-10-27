"""Guess Number environment: the oracle a harness is allowed to reach, and nothing else.

AR-Bench's own prompts and scoring function are imported rather than restated, so a harness that
does nothing clever reproduces AR-Bench's zero-shot loop exactly. The target is private: the only
way to learn anything about it is to spend a guess.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.gn.prompt import (  # noqa: E402
    Final_prompt, Game_rule, Guess_number_prompt,
)
from arbench.utils.utils_gn import compare_guess  # noqa: E402

BUDGET = 25


class BudgetExhausted(Exception):
    pass


class GNEnv:
    def __init__(self, target, budget=BUDGET):
        self._target = [int(d) for d in str(target).zfill(4)]
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
        a, b, _ = compare_guess(self._target, g)
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
