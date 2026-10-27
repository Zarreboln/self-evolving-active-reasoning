"""The floor: write the explanation from the surface alone, asking the referee nothing.

The cheap anchor for the frontier, and a measurement in its own right: how much of a situation
puzzle a frozen model can reconstruct with no information beyond the paradox itself. Everything
the questioning methods achieve has to be read as a gain over this number.
"""
from __future__ import annotations

from ..harness_base import SPHarness


class Harness(SPHarness):
    def solve(self) -> str:
        return self.llm([{"role": "system", "content": self.env.system_prompt},
                         {"role": "user", "content": self.env.final_prompt}])
