"""The HARNESS interface for Guess Number (AR-Bench): arbitrary Python wrapping the FROZEN model
and playing one whole episode.

`solve()` owns the loop. Two invariants (the audit checks them):

  * FROZEN SOLVER -- never change WHO answers: use self.llm(...) only. No new clients, no
    base_url / api_key, no network or file access, no subprocesses.
  * NO ORACLE PEEKING -- the target lives inside the environment and is reachable only through
    env.guess(number) -> (correct_position, correct_digit_wrong_position). Never touch
    env._target, the data file, or AR-Bench's compare_guess directly.

The second invariant is the one this task gets wrong in the wild: AR-Bench's own gn/tot scores
each *candidate* with `compare_guess(target, guess)` before deciding which to play, so it consults
the oracle ~75 times against a budget of 25. Here every consultation is a played guess.

Episode protocol: up to BUDGET four-digit guesses, each answered with how many digits are in the
right place and how many are present but misplaced. The episode ends when a guess is exactly right
or the budget runs out. `solve()` returns the final guess as a four-character string.

Helpers:
    self.env.system_prompt       -> AR-Bench's Game_rule, as the player is shown it
    self.env.turn_prompt()       -> AR-Bench's Guess_number_prompt over the record so far
    self.env.final_prompt        -> AR-Bench's Final_prompt over the record so far
    self.env.guess(number)       -> (a, b); counts against the budget; BudgetExhausted after it
    self.env.record              -> the feedback text the player has accumulated
    self.env.history             -> [(guess, a, b), ...]
    self.env.remaining / self.env.budget / self.env.solved
    self.llm(messages, temperature=0.7, max_tokens=None) -> str   (the frozen model; one chat call)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BudgetExhausted(Exception):
    pass


class GNHarness(ABC):
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
        """Play the episode through self.env / self.llm and return the final four-digit guess."""
