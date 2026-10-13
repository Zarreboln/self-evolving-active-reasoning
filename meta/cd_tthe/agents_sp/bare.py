"""The official AR-Bench zero-shot loop, nothing added: system prompt = AR-Bench's player
instruction; each turn the user message "Turn t: please propose your next question."; the model's
reply is sent to the referee as the question; the referee's reply is appended as the next user
message; after the budget, AR-Bench's final-answer prompt; the reply is the answer."""
from sp_harness_base import SPHarness


class Harness(SPHarness):
    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt}]
        for t in range(1, env.budget + 1):
            msgs.append({"role": "user", "content": env.turn_prompt(t)})
            q = self.llm(msgs) or ""
            msgs.append({"role": "assistant", "content": q})
            a = env.ask(q)
            msgs.append({"role": "user", "content": a})
        msgs.append({"role": "user", "content": env.final_prompt})
        return self.llm(msgs) or ""
