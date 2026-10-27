"""The HARNESS interface for Detective Cases (AR-Bench): arbitrary Python wrapping the FROZEN model
and playing one whole episode.

`solve()` owns the loop. Two invariants (the audit checks them):

  * FROZEN SOLVER -- never change WHO answers: use self.llm(...) only. No new clients, no
    base_url / api_key, no network or file access, no subprocesses. The suspects are played by the
    same frozen model, from inside the environment; the harness never reaches them directly.
  * NO KEY-QUESTION PEEKING -- each suspect's story and the case's gold `key_question` list live
    inside the environment and are not reachable at all. Never touch env._suspects, env._label,
    the data file, or AR-Bench's keypoint_hits_prompt.

The second invariant is the one AR-Bench's own dc/tot breaks: it scores each candidate question by
how many hidden key questions it hit, so the search is steered by the answer key. Here the only
signal that comes back is what a suspect says.

Episode protocol: up to BUDGET interrogation turns. Each turn names one suspect and asks them one
question; the suspect replies in free text. After the last turn the player names the murderer.
`solve()` returns the 0-based index of the accused suspect.

Helpers:
    self.env.init_info            -> the case background the player is shown
    self.env.suspects             -> [name, ...] in the order the player sees them
    self.env.system_prompt        -> AR-Bench's detective instruction, filled for this case
    self.env.turn_prompt(t)       -> AR-Bench's question_propose_prompt for turn t
    self.env.final_prompt         -> AR-Bench's accusation prompt, with the lettered choices
    self.env.interrogate(name, q) -> the suspect's reply; counts against the budget
    self.env.history              -> [(name, question, reply), ...]
    self.env.record               -> the transcript as AR-Bench formats it
    self.env.remaining / self.env.budget
    self.llm(messages, temperature=0.7, max_tokens=None) -> str   (the frozen model; one chat call)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BudgetExhausted(Exception):
    pass


class DCHarness(ABC):
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
    def solve(self) -> int:
        """Play the episode through self.env / self.llm and return the accused suspect's index."""
