"""The floor: one guess, no feedback, no search. The cheap anchor the frontier needs.

Meta-Harness's own seed set has `no_memory` sitting at zero cost with a respectable score, so the
cheap corner of its frontier is already occupied and a candidate has to beat it there to be
interesting. Ours had nothing below 21k tokens, which left "do almost nothing" as free frontier
real estate -- a harness that guessed once and stopped would have been Pareto-optimal forever, and
the proposer would have read that as success.

This also measures something worth knowing on its own: what the task is worth before any
interaction. For Guess Number that is 1/5040 per guess, so the floor is the floor.
"""
from __future__ import annotations

import re

from ..harness_base import GNHarness

FOUR = re.compile(r"\b(\d{4})\b")


class Harness(GNHarness):
    def solve(self) -> str:
        out = self.llm([{"role": "user", "content": self.env.final_prompt}])
        m = FOUR.findall(out or "")
        return m[-1] if m else "0000"
