"""The HARNESS interface for AMIGO (Guess My Preferred Dress): arbitrary Python wrapping the
FROZEN vision-language model and playing one whole episode.

`solve()` owns the loop and returns the 0-based index of the gallery image it names. Four
invariants, all audited:

  * FROZEN SOLVER -- never change WHO answers: use self.llm(...) only. No new clients, no
    base_url / api_key, no network, no subprocesses. The oracle and the protocol detector live
    inside the environment and are never reachable directly.
  * NO LOOKING AT THE PICTURES. The harness may hand images to the model and may reason over what
    the model says about them; it may not analyse pixels itself. No PIL, no numpy on image data,
    no hashing or comparing the image bytes. The player in this task is a vision model, and a
    harness that did its own vision would be a different system, not a harness for this one.
  * NO CROSS-EPISODE STATE. Every episode starts from nothing. No module-level cache, no file
    written in one episode and read in another, no memory of a gallery or a question that worked.
    Episodes are evaluated concurrently, and on Guess Number a harness discovered it could merge
    sixteen concurrent episodes into one model call and divide the cost by sixteen; that is a
    property of our runner, not of the task. The benchmark snapshots this module's globals before
    the run and compares them after, and a candidate that mutated them is rejected.
  * A QUESTION COSTS A TURN. The eleven protocol rules are published, so a harness may implement
    them and screen its own drafts before submitting -- that is legitimate and is where a quarter
    of the wasted budget sits. What it may not do is ask the environment whether a draft is legal.
    `env.ask` is the only way to find out, and the turn is spent either way.

Episode protocol (the paper's): a gallery of dresses is shown; one of them is the hidden target.
The player has BUDGET turns. Each turn is one yes/no question about one observable construction
attribute of the target. A question that breaks a protocol rule is answered `Skip` and the turn is
gone. After the last turn, or whenever it likes, the player names an image.

Helpers:
    self.env.gallery            -> [image_id, ...] in the order the player sees them
    self.env.n                  -> how many images are in the gallery
    self.env.image_block(i)     -> one chat content block holding gallery image i
    self.env.gallery_blocks()   -> the whole gallery as content blocks, in order
    self.env.rules              -> the protocol as the paper states it to the player
    self.env.ask(question)      -> "Yes" / "No" / "Unsure" / "Skip"; always spends a turn
    self.env.history            -> [(question, answer), ...], Skips included
    self.env.answered           -> [(question, answer), ...] with the Skips left out
    self.env.remaining / self.env.budget
    self.llm(messages, temperature=0.7, max_tokens=None) -> str   (the frozen model; one call)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BudgetExhausted(Exception):
    pass


class AMIGOHarness(ABC):
    def __init__(self, env, solver):
        self.env = env
        self._solver = solver
        self._trace = []

    def llm(self, messages, temperature=0.7, max_tokens=None):
        snapshot = [dict(m) for m in messages]
        out = self._solver(snapshot, temperature=temperature, max_tokens=max_tokens)
        self._trace.append({"step": "llm", "response": out})
        return out

    @abstractmethod
    def solve(self) -> int:
        """Play the episode and return the 0-based gallery index of the dress named."""
