"""AR-Bench's zero-shot Situation Puzzles loop, wrapped in the harness interface and nothing more.

This is the floor the search has to beat, and the fidelity check: run AR-Bench's own evaluator and
this file on the same puzzles and the scores have to match, because the two are the same loop. If
they do not, the interface is lying about what it reproduces and no later comparison means anything.

One growing conversation. Each turn appends "Turn t: please propose your next question.", the
model's question goes to the referee, and the referee's reply comes back as a user message. After
the budget, AR-Bench's `get_answer_prompt` asks for the explanation.
"""
from __future__ import annotations

from ..harness_base import SPHarness


class Harness(SPHarness):
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
