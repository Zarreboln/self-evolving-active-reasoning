"""The HARNESS interface for CircuitDecoding, in the TTHE sense (arXiv 2607.08124):
a harness is ARBITRARY PYTHON wrapping the FROZEN task model and playing one whole episode.

Unlike meta/cd_evolve (hooks around a fixed loop), here `solve()` owns the loop: it decides what
the model sees, how often it is called, whether its replies are accepted, how the final guess is
assembled. Two invariants are fixed (audit.py checks them; a candidate that breaks either is
invalid):

  * FROZEN SOLVER — never change WHO answers: use self.llm(...) only. No new clients, no
    base_url / api_key, no network or file access, no subprocesses.
  * LABEL-FREE — the truth table lives inside the environment and is reachable only through
    env.query(...) (one bit per query, 18 per episode). Never touch env._truth or the instance
    files; never import the runner's data loaders.

Episode protocol (fixed by the benchmark): up to BUDGET queries, then exactly one final guess,
a 24-bit string: circuit A rows 000..111, then B, then C. `solve()` returns that string (or any
text; the environment extracts the last 24-bit string). Queries beyond the budget raise
BudgetExhausted; a harness that raises anything is recorded as a crash for that episode.

Helpers:
    self.env.inventory        -> {"AND": a, "OR": o, "NOT": n}   (what the task tells the player)
    self.env.query(c, a, b, x) -> 0/1   (c in "ABC"; counts against the budget; answers are logged)
    self.env.remaining        -> queries left
    self.env.history          -> [((c, a, b, x), bit), ...] in order
    self.env.format_answer(c, a, b, x, bit) -> "A(0, 0, 1) = 1"   (the benchmark's wording)
    self.llm(messages, temperature=0.7, max_tokens=1200) -> str   (the frozen 7B; one chat call)
    self.env.system_prompt    -> the benchmark's instruction text, filled for this instance
    self.env.last_turn_prompt -> the benchmark's final-guess prompt
Optional in the `lib` arm only: `from cd_oracle import CDGame` (public hypothesis space, no truth).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BudgetExhausted(Exception):
    pass


class CDHarness(ABC):
    def __init__(self, env, solver):
        self.env = env
        self._solver = solver          # callable(messages, temperature, max_tokens) -> str
        self._trace = []

    def llm(self, messages, temperature=0.7, max_tokens=1200):
        out = self._solver(messages, temperature=temperature, max_tokens=max_tokens)
        self._trace.append({"step": "llm", "messages": messages, "response": out})
        return out

    @abstractmethod
    def solve(self) -> str:
        """Play the episode through self.env / self.llm and return the final guess."""
