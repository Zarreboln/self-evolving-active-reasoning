"""ALFWorld (text), one subtask per official task type.

Games are ALFWorld's TextWorld PDDL files, played through ALFWorld's own wrappers (demangler,
infos, handcoded expert). The official splits map onto ours directly: train -> search,
valid_seen -> selection, valid_unseen (the 134 games every paper reports) -> test. The env allows
at most 50 actions. ReAct's own budget is an agent loop of 49 model completions, thoughts included
(alfworld.ipynb, range(1, 50)); its seed harness enforces that, not the env.

score = success = TextWorld's `won`
"""
from __future__ import annotations

import functools
import glob
import os

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredInfos

from mb.core import TextEnv

BENCH = "alfworld"
SUBTASKS = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
            "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
            "pick_two_obj_and_place"]
DIMENSIONS = {"world": "yes", "hypotheses": "structured", "observation": "text percept",
              "cost": "steps", "stop": "task done"}
RESPONDER = None
_SPLIT_DIR = {"search": "train", "selection": "valid_seen", "test": "valid_unseen"}
MAX_STEPS = 50


def _root():
    return os.path.join(os.environ["ALFWORLD_DATA"], "json_2.1.1")


@functools.lru_cache(maxsize=None)
def _games(split):
    return sorted(os.path.relpath(p, _root())
                  for p in glob.glob(f"{_root()}/{_SPLIT_DIR[split]}/*/*/game.tw-pddl"))


def instances(subtask, split, limit=None):
    out = [g for g in _games(split) if g.split("/")[1].split("-")[0] == subtask]
    return out[:limit] if limit else out


class AlfWorldEnv(TextEnv):
    bench = BENCH
    max_steps = MAX_STEPS

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        infos = textworld.EnvInfos(won=True, admissible_commands=True,
                                   extras=["gamefile", "expert_plan"])
        env_id = textworld.gym.register_game(os.path.join(_root(), instance), infos,
                                             max_episode_steps=MAX_STEPS,
                                             wrappers=[AlfredDemangler(), AlfredInfos, AlfredExpert])
        self._env = textworld.gym.make(env_id)
        obs, self._info = self._env.reset()
        self.instructions = obs
        self._won = False

    def admissible(self):
        return list(self._info["admissible_commands"])

    def _act(self, action):
        obs, _, done, self._info = self._env.step(action.strip())
        self._won = bool(self._info["won"])
        if done:
            self.done = True
        return obs

    def score(self):
        return 1.0 if self._won else 0.0

    def success(self):
        return self._won

    def _oracle_next(self):
        plan = self._info.get("extra.expert_plan") or []
        return plan[0] if plan else None


def make(subtask, instance, responder=None):
    return AlfWorldEnv(subtask, instance, responder)
