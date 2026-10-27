"""The HARNESS interface for Situation Puzzles (AR-Bench), in the TTHE sense: arbitrary Python
wrapping the FROZEN task model and playing one whole episode.

`solve()` owns the loop: what the model sees, how often it is called, whether its replies are
accepted, how the final answer is assembled. Two invariants (audit.py checks them):

  * FROZEN SOLVER -- never change WHO answers: use self.llm(...) only. No new clients, no
    base_url / api_key, no network or file access, no subprocesses.
  * LABEL-FREE -- the story (`bottom`) and the key questions live inside the environment and
    are reachable only through env.ask(question) -> "Yes" / "No" / "Unknown" (AR-Bench's referee).
    Never touch env._bottom / env._keypoints, the data file, or the AR-Bench package.

Episode protocol (fixed by the benchmark): up to BUDGET yes/no questions, then one free-text
final answer. `solve()` returns the final answer text. The model must be the one writing every
question that is sent and the final answer; the harness may re-ask it, change what it sees, veto a
reply, keep notes -- but may not write a question itself.

Helpers:
    self.env.surface            -> the puzzle text the player is shown
    self.env.system_prompt      -> AR-Bench's player instruction, filled for this puzzle
    self.env.turn_prompt(t)     -> "Turn t: please propose your next question."
    self.env.final_prompt       -> AR-Bench's final-answer prompt
    self.env.ask(question)      -> referee reply string (counts against the budget; BudgetExhausted after it)
    self.env.remaining / self.env.history ([(question, reply), ...]) / self.env.budget
    self.llm(messages, temperature=0.7, max_tokens=None) -> str   (the frozen 7B; one chat call)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BudgetExhausted(Exception):
    pass


class SPHarness(ABC):
    def __init__(self, env, solver):
        self.env = env
        self._solver = solver
        self._trace = []

    def llm(self, messages, temperature=0.7, max_tokens=None):
        snapshot = [dict(m) for m in messages]
        out = self._solver(snapshot, temperature=temperature, max_tokens=max_tokens)
        self._trace.append({"step": "llm", "messages": snapshot, "response": out})
        return out

    @abstractmethod
    def solve(self) -> str:
        """Play the episode through self.env / self.llm and return the final answer text."""
