"""One episode interface for every bench in the suite.

A harness sees a `TextEnv` and nothing else: the instructions it is given at the start, one text
observation after each action, and, where the bench provides them, the admissible actions. Every
bench-specific detail (TextArena's message tuples, TextWorld's info dict, AR-Bench's per-task
methods) is absorbed by that bench's adapter, which is fixed and never evolved. This is what lets a
component written for one bench be plugged into another: the only thing that differs between
benches is text.

Hidden state (the secret word, the murderer, the gold path) lives in underscore attributes of the
adapter. A harness never reads them; `_oracle_*` exist only for the smoke test, which checks
that the success path of each adapter scores what the bench says it should.

Scores are the bench's own metric, mapped to [0, 1] where the bench uses another range, and
`success` is the bench's own notion of solving the instance. Both are recorded; neither is
redefined here. The one exception to [0, 1] is a bench whose official metric is an error with no
fixed range (BoxingGym): its score is that metric as the paper standardises it, and
`higher_is_better` is False.
"""
from __future__ import annotations

import hashlib
import random
import threading
from abc import ABC, abstractmethod

import numpy as np

SPLITS = ("search", "selection", "test")

# Several benches (TextArena, BoxingGym) draw secrets and noise from the process-global `random` /
# numpy state. Any such call goes through `seeded`, which reseeds from a per-instance value under
# one lock, so an episode replays identically whatever else runs before it or in other threads.
GLOBAL_RNG_LOCK = threading.Lock()


def seeded(rng_seed, fn, /, *args, **kwargs):
    with GLOBAL_RNG_LOCK:
        s = int(rng_seed) % (2 ** 32)
        random.seed(s)
        np.random.seed(s)
        return fn(*args, **kwargs)


class EpisodeOver(Exception):
    pass


class TextEnv(ABC):
    bench: str = ""
    max_steps: int = 0
    higher_is_better: bool = True     # False for benches whose official metric is an error

    def __init__(self, subtask: str, instance: str):
        self.subtask = subtask
        # The instance key can be the answer itself (a Wordle word, a 20Q target), so the harness
        # only ever sees an opaque id.
        self._instance = instance
        self.instance_id = hashlib.md5(f"{self.bench}/{subtask}/{instance}".encode()).hexdigest()[:10]
        self.instructions = ""        # set by the adapter before the first action
        self.observation = ""         # the latest observation text
        self.steps = 0
        self.done = False
        self.transcript = []          # [(action, observation)]

    def act(self, action: str) -> str:
        """Take one action, return the next observation. Raises EpisodeOver once the episode ends."""
        if self.done:
            raise EpisodeOver()
        self.steps += 1
        obs = self._act(action)
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            self._on_budget_exhausted()
        self.observation = obs
        self.transcript.append((action, obs))
        return obs

    def admissible(self) -> list[str] | None:
        """Valid actions right now, when the bench itself provides them; otherwise None."""
        return None

    @abstractmethod
    def _act(self, action: str) -> str: ...

    def _on_budget_exhausted(self):
        """Hook for benches whose official protocol does something at the step limit."""

    @abstractmethod
    def score(self) -> float: ...

    @abstractmethod
    def success(self) -> bool | None:
        """The bench's own notion of solving the instance; None where the bench defines none."""

    def _oracle_next(self) -> str | None:
        """Smoke test only: the next action of a known-good player, or None if there is none."""
        plan = getattr(self, "_plan", None)
        if plan is None:
            plan = self._plan = list(self._oracle_plan() or [])
        return plan.pop(0) if plan else None

    def _oracle_plan(self) -> list[str] | None:
        """Smoke test only: a known-good action sequence, for benches where one can be fixed upfront."""
        return None

    def result(self) -> dict:
        return {"bench": self.bench, "subtask": self.subtask, "instance": self._instance,
                "instance_id": self.instance_id,
                "steps": self.steps, "done": self.done, "score": self.score(),
                "higher_is_better": self.higher_is_better, "success": self.success()}


def hash_split(key: str, test=0.4, selection=0.3) -> str:
    """Stable split for benches with no official one. Depends on the key only, never on order."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if h < test else "selection" if h < test + selection else "search"
