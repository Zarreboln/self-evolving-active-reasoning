"""ScienceWorld, one subtask per official task (30 tasks, 10 topics).

The official variation splits map onto ours: train -> search, dev -> selection, test -> test.
Played with the "easy" simplification, the setting most agent papers report. The step limit is
100 (250 for the three tasks whose gold path is longer); ScienceWorld has no single official one,
so it is fixed here and stated.

ScienceWorld scores a task 0-100 and returns a negative score when the agent commits to a wrong
answer (e.g. focuses on the wrong object). The raw score is kept in `raw_score`; `score` is
max(raw, 0) / 100, the way papers average it.

score   = max(raw, 0) / 100
success = raw == 100
Needs Java on PATH (source mb/env.sh).
"""
from __future__ import annotations

import functools
import json
import os

from scienceworld import ScienceWorldEnv

from mb.core import TextEnv

BENCH = "sciworld"
SUBTASKS = ["boil", "melt", "freeze", "change-the-state-of-matter-of",
            "use-thermometer", "measure-melting-point-known-substance",
            "measure-melting-point-unknown-substance",
            "power-component", "power-component-renewable-vs-nonrenewable-energy",
            "test-conductivity", "test-conductivity-of-unknown-substances",
            "find-living-thing", "find-non-living-thing", "find-plant", "find-animal",
            "grow-plant", "grow-fruit",
            "chemistry-mix", "chemistry-mix-paint-secondary-color",
            "chemistry-mix-paint-tertiary-color",
            "lifespan-longest-lived", "lifespan-shortest-lived",
            "lifespan-longest-lived-then-shortest-lived",
            "identify-life-stages-1", "identify-life-stages-2",
            "inclined-plane-determine-angle", "inclined-plane-friction-named-surfaces",
            "inclined-plane-friction-unnamed-surfaces",
            "mendelian-genetics-known-plant", "mendelian-genetics-unknown-plant"]
DIMENSIONS = {"world": "yes", "hypotheses": "structured-open", "observation": "text percept",
              "cost": "steps, some irreversible", "stop": "task done / focus commits"}
RESPONDER = None
MAX_STEPS = 100
# Three tasks whose own gold path is longer than 100 steps (measured on the first search variation:
# 178, 144, 130); at 100 they would be unsolvable by construction.
MAX_STEPS_OVERRIDE = {"inclined-plane-friction-unnamed-surfaces": 250,
                      "mendelian-genetics-known-plant": 250,
                      "mendelian-genetics-unknown-plant": 250}
SIMPLIFICATION = "easy"


def _cache_path():
    return os.path.join(os.environ.get("MBENCH_DATA", "."), "sciworld_variations.json")


@functools.lru_cache(maxsize=None)
def _variations():
    p = _cache_path()
    if os.path.exists(p):
        return json.load(open(p))
    env = ScienceWorldEnv("", envStepLimit=MAX_STEPS)
    out = {}
    try:
        for t in SUBTASKS:
            env.load(t, 0, SIMPLIFICATION)
            out[t] = {"search": env.get_variations_train(), "selection": env.get_variations_dev(),
                      "test": env.get_variations_test()}
    finally:
        env.close()
    json.dump(out, open(p, "w"))
    return out


def instances(subtask, split, limit=None):
    out = [str(v) for v in _variations()[subtask][split]]
    return out[:limit] if limit else out


class SciWorldEnv(TextEnv):
    bench = BENCH

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        self.max_steps = MAX_STEPS_OVERRIDE.get(subtask, MAX_STEPS)
        self._env = ScienceWorldEnv("", envStepLimit=self.max_steps + 10)
        self._env.load(subtask, int(instance), SIMPLIFICATION, generateGoldPath=True)
        obs, self._info = self._env.reset()
        self.instructions = self._env.get_task_description() + "\n\n" + obs
        self.raw_score = 0

    def admissible(self):
        return list(self._info.get("valid", []))

    def _act(self, action):
        obs, _, done, self._info = self._env.step(action.strip())
        self.raw_score = self._info.get("score", self.raw_score)
        if done:
            self.done = True
        if self.done:
            self.close()
        return obs

    def _on_budget_exhausted(self):
        self.close()

    def close(self):
        if getattr(self, "_env", None) is not None:
            self._env.close()
            self._env = None

    def score(self):
        return max(self.raw_score, 0) / 100

    def success(self):
        return self.raw_score == 100

    def result(self):
        return {**super().result(), "raw_score": self.raw_score}

    def _oracle_plan(self):
        return self._env.get_gold_action_sequence()


def make(subtask, instance, responder=None):
    return SciWorldEnv(subtask, instance, responder)
