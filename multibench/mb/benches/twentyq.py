"""Twenty Questions, TextArena's official implementation, one subtask per (word list, theme).

One deliberate change: TextArena hard-codes GPT-4o through OpenRouter as the gamemaster. Here the
gamemaster is the run's pinned responder: Jev (mb/jev.py), given the secret and the player's
question and answering with TextArena's own options; or a chat model called with TextArena's own
prompt, options and fallback. Everything else -- the word lists, the 20-question limit, the bracketed final guess, and
the substring check that decides a win -- is TextArena's.

score = success = the target word appears in the bracketed guess (TextArena's rule)
"""
from __future__ import annotations

import importlib.resources
import json
import re

import textarena as ta

from mb.core import TextEnv, hash_split, seeded

BENCH = "twentyq"
_LEVELS = {"basic": "TwentyQuestions-v0", "hard": "TwentyQuestions-v0-hardcore"}
_THEMES = ("places", "people", "things")
SUBTASKS = {f"{lv}-{th}": (lv, th) for lv in _LEVELS for th in _THEMES}
DIMENSIONS = {"world": "no", "hypotheses": "enumerable", "observation": "model yes/no",
              "cost": "questions", "stop": "self"}
RESPONDER = "llm"

_DATA = json.load(importlib.resources.files("textarena.envs.TwentyQuestions")
                  .joinpath("twenty_questions_words.json").open())


def _wordlist(level, theme):
    return sorted(set(_DATA["basic" if level == "basic" else "hardcore"][theme]))


def instances(subtask, split, limit=None):
    lv, th = SUBTASKS[subtask]
    out = [w for w in _wordlist(lv, th) if hash_split(f"{BENCH}/{subtask}/{w}") == split]
    return out[:limit] if limit else out


class _NoAgent:
    def __init__(self, *a, **k):
        pass


def _question(prompt):
    """The player's question out of TextArena's gamemaster prompt ("...Q: {action}\\nOptions: ...")."""
    m = re.findall(r"\nQ: (.*?)\nOptions:", prompt, re.S)
    return (m[-1] if m else prompt).strip()


class TwentyQEnv(TextEnv):
    reads_full_reply = True
    bench = BENCH

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        if responder is None:
            raise ValueError("twentyq needs a responder to play the gamemaster")
        lv, th = SUBTASKS[subtask]
        real, ta.agents.OpenRouterAgent = ta.agents.OpenRouterAgent, _NoAgent
        try:
            self._env = ta.make(_LEVELS[lv] + "-raw")
        finally:
            ta.agents.OpenRouterAgent = real
        if getattr(responder, "jev_role", None) == "gamemaster":
            # Jev answers from the secret and the player's question with TextArena's own options
            self._env.gamemaster = lambda prompt: responder.answer(
                "20 Questions", instance, _question(prompt))
        else:
            self._env.gamemaster = lambda prompt: responder(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64)
        self._env.word_list = {th: [instance]}      # reset() draws theme and word from this
        seeded(0, self._env.reset, num_players=1, seed=0)   # reset() reseeds the global RNG
        self.max_steps = self._env.state.max_turns + 1
        _, obs = self._env.get_observation()
        self.instructions = "\n".join(m for _, m, _ in obs)
        self._reward = 0.0

    def _act(self, action):
        done, _ = self._env.step(action)
        _, obs = self._env.get_observation()
        text = "\n".join(m for frm, m, kind in obs if kind != ta.ObservationType.PLAYER_ACTION)
        if done:
            rewards, info = self._env.close()
            self.raw_reward = float(rewards[0])
            self._reward = max(self.raw_reward, 0.0)   # TextArena pays -1 for a loss by invalid moves
            self.done = True
            text = (text + "\n" + info[0].get("reason", "")).strip()
        return text

    def score(self):
        return self._reward

    def result(self):
        return {**super().result(), "raw_reward": getattr(self, "raw_reward", None)}

    def success(self):
        return self._reward == 1.0

    def _oracle_plan(self):
        return ["Is it something people talk about?", f"[{self._instance}]"]


def make(subtask, instance, responder=None):
    return TwentyQEnv(subtask, instance, responder)
