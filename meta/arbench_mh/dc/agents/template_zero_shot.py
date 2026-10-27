"""The plainest Detective Cases harness: pick a suspect, ask them something, repeat, then accuse.

Not a scored baseline -- the seeds come from AR-Bench's own evaluator -- but a readable floor that
shows the interface. Note what it does NOT do: it never decides WHO to interrogate on evidence, it
round-robins. Whom to ask is a free axis here and AR-Bench's own methods barely use it.
"""
from __future__ import annotations

from ..harness_base import DCHarness


class Harness(DCHarness):
    def solve(self) -> int:
        env = self.env
        for t in range(env.budget):
            who = env.suspects[t % len(env.suspects)]
            q = self.llm([{"role": "system", "content": env.system_prompt},
                          {"role": "user", "content": env.turn_prompt(t + 1, suspect=who)}])
            env.interrogate(who, (q or "").strip().split("\n")[0])
        out = self.llm([{"role": "user", "content": env.final_prompt}])
        idx = env.parse_choice(out)
        return idx if idx is not None else 0
