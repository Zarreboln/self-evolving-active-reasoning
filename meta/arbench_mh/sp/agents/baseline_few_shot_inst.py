"""AR-Bench's few_shot_inst loop, wrapped in the harness interface and nothing more.

The strongest of AR-Bench's prompt baselines on this task (68/100 non-strict equivalence against
zero_shot's 63) and the accuracy end of the initial frontier, so the search starts from something
that already appears in a published table.

Identical to baseline_zero_shot except for which of AR-Bench's system prompts the environment was
built with -- `METHOD_DICT["few_shot_inst"]`, which carries one worked example and the hand-written
instructions. The loop itself is theirs, unchanged.
"""
from __future__ import annotations

from ..harness_base import SPHarness


class Harness(SPHarness):
    METHOD = "few_shot_inst"

    def solve(self) -> str:
        env = self.env
        agent = [{"role": "system", "content": env.system_prompt}]
        for t in range(env.budget):
            agent.append({"role": "user", "content": env.turn_prompt(t + 1)})
            question = self.llm(agent)
            answer = env.ask(question)
            agent.append({"role": "assistant", "content": question})
            agent.append({"role": "user", "content": answer})
        agent.append({"role": "user", "content": env.final_prompt})
        return self.llm(agent)
