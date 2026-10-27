"""The floor: accuse from the case background alone, with no interrogation at all.

The cheap anchor that keeps the frontier's low-cost corner from being free real estate. On this
task it measures the thing every other number should be read against: five suspects, so chance is
20%, and whatever this scores above 20 is what the background text alone gives away. Every method
that does interrogate has to be read as a gain over this, not over chance.
"""
from __future__ import annotations

from ..harness_base import DCHarness


class Harness(DCHarness):
    def solve(self) -> int:
        out = self.llm([{"role": "user", "content": self.env.final_prompt}])
        idx = self.env.parse_choice(out)
        return idx if idx is not None else 0
