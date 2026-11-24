"""BabyAI in text, through BALROG's wrapper (vendored in mb/vendor/balrog_babyai, MIT).

Two kinds of subtask:
  balrog/<goal>   BALROG's own five tasks, BabyAI-MixedTrainLocal-v0 filtered by instruction kind
  <level>         an official BabyAI level, wrapped the same way; these carry the level x skill
                  table of the BabyAI paper, which is what the hierarchy can be checked against

BALROG's `make_babyai_env` only builds the MixedTrainLocal tasks (any other name falls through to
an unbound variable), so the official levels are built here with the same wrapper. BALROG passes
`num_dists=0` to every task; the official levels keep their own defaults, as in the BabyAI paper. Observations,
the instruction prompt, the action strings and the invalid-action rule (substitute "go forward",
as BALROG's `check_action_validity` does) are BALROG's.

Instances are seeds. The splits are disjoint seed ranges.

score = success = BALROG's progression: 1 once the mission's reward is positive
"""
from __future__ import annotations

import gymnasium as gym

from mb.core import TextEnv
from mb.vendor.balrog_babyai import get_instruction_prompt
from mb.vendor.balrog_babyai.clean_lang_wrapper import BABYAI_ACTION_SPACE, BabyAITextCleanLangWrapper
import mb.vendor.balrog_babyai.babyai_env  # noqa: F401  (registers the minigrid envs)

BENCH = "babyai"
_BALROG = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]
_LEVELS = ["GoToRedBallGrey", "GoToRedBall", "GoToLocal", "PutNextLocal", "PickupLoc",
           "GoToObjMaze", "GoTo", "Pickup", "UnblockPickup", "Open", "Unlock", "PutNext",
           "Synth", "SynthLoc", "GoToSeq", "SynthSeq", "GoToImpUnlock", "BossLevel", "GoToObj"]
# The BabyAI paper's PutNext level is registered as PutNextS7N4.
_REGISTERED = {"PutNext": "PutNextS7N4"}
SUBTASKS = {**{f"balrog/{g}": ("BabyAI-MixedTrainLocal-v0", g) for g in _BALROG},
            **{lv: (f"BabyAI-{_REGISTERED.get(lv, lv)}-v0", None) for lv in _LEVELS}}
DIMENSIONS = {"world": "yes", "hypotheses": "structured", "observation": "text percept",
              "cost": "steps", "stop": "task done"}
RESPONDER = None
_RANGES = {"search": 0, "selection": 100_000, "test": 1_000_000}


def instances(subtask, split, limit=None):
    start = _RANGES[split]
    return [str(s) for s in range(start, start + (limit or 200))]


class BabyAIEnv(TextEnv):
    bench = BENCH

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        env_id, goal = SUBTASKS[subtask]
        if goal is None:
            base = gym.make(env_id)
        else:
            # MixedTrainLocal draws its instruction kind once, in the constructor, from an
            # entropy-seeded RNG that reset(seed) never touches; that draw (and the room count it
            # implies) is the constructor's only random choice. So, as BALROG does, rebuild until
            # the kind matches -- the env is then fully fixed and reset(seed) decides the episode.
            while True:
                base = gym.make(env_id, num_dists=0)
                if base.unwrapped.action_kinds[0].replace(" ", "_") == goal:
                    break
        self._env = BabyAITextCleanLangWrapper(base)
        obs, _ = self._env.reset(seed=int(instance))
        self.max_steps = self._env.max_steps
        self.instructions = (get_instruction_prompt(self._env, mission=obs["mission"])
                             + "\n\nCurrent Observation:\n" + obs["text"]["long_term_context"])
        self._won = False

    def admissible(self):
        return list(BABYAI_ACTION_SPACE)

    def _act(self, action):
        a = action.strip().lower()
        note = ""
        if a not in BABYAI_ACTION_SPACE:
            note = f"Invalid action {action!r}; 'go forward' was taken instead.\n"
            a = self._env.default_action
        obs, reward, terminated, truncated, _ = self._env.step(a)
        if reward > 0:
            self._won = True
        if terminated or truncated:
            self.done = True
        return note + obs["text"]["long_term_context"]

    def score(self):
        return 1.0 if self._won else 0.0

    def success(self):
        return self._won

    def _oracle_next(self):
        from minigrid.utils.baby_ai_bot import BabyAIBot
        if not hasattr(self, "_bot"):
            self._bot = BabyAIBot(self._env.unwrapped)
        return BABYAI_ACTION_SPACE[int(self._bot.replan())]


def make(subtask, instance, responder=None):
    return BabyAIEnv(subtask, instance, responder)
