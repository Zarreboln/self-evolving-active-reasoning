"""The plainest Guess Number harness: ask, play, repeat. The starting point to copy and edit.

This is not a scored baseline -- the seeds' frontier points come from AR-Bench's own evaluator --
it is a readable floor that shows the interface in use. It asks for a guess, plays it, and
repeats; it keeps nothing between turns.
"""
from __future__ import annotations

import re

from ..harness_base import GNHarness

FOUR = re.compile(r"\b(\d{4})\b")


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        last = "0000"
        for _ in range(env.budget):
            if env.solved:
                break
            out = self.llm([{"role": "user", "content": env.turn_prompt()}])
            m = FOUR.findall(out or "")
            if not m:
                continue                      # a turn that produced no guess is a turn spent
            last = m[-1]
            env.guess(last)
        return last
