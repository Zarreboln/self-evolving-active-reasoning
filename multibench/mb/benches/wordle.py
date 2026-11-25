"""Wordle, TextArena's official implementation, all four of its registered variants.

The environment, feedback wording, invalid-move rules (one error allowed, the second loses) and
reward are TextArena's. The adapter only fixes the secret word per instance instead of drawing it
at random, so splits are disjoint by word.

score   = TextArena's reward: 1 for a solve, else the green/yellow completion of the last guess;
          floored at 0 (a loss by invalid moves pays -1 in TextArena, kept as `raw_reward`)
success = the secret word was guessed
"""
from __future__ import annotations

import functools
import json
import os

import textarena as ta

from mb.core import TextEnv, hash_split, seeded

BENCH = "wordle"
SUBTASKS = {"basic5": "Wordle-v0", "hard5": "Wordle-v0-hardcore",
            "basic7": "Wordle-v0-long", "hard7": "Wordle-v0-long-hardcore"}
DIMENSIONS = {"world": "no", "hypotheses": "enumerable", "observation": "rule",
              "cost": "guesses", "stop": "self"}
RESPONDER = None


@functools.lru_cache(maxsize=None)
def _words(subtask):
    """TextArena's secret-word list, minus the words its own rules make unwinnable.

    The hardcore lists come from nltk's `words` and keep proper nouns ("Acroa", "Essex"): a guess
    is lowercased before it is compared, so a capitalised secret can never be matched, and several
    also fail TextArena's own dictionary check, so even typing them is an invalid move. 1,134 of
    9,047 (hard5) and 2,175 of 19,045 (hard7) are dropped; the basic lists lose none. Cached,
    because building the list POS-tags every word (~30 s).
    """
    path = os.path.join(os.environ.get("MBENCH_DATA", "."), f"wordle_{subtask}.json")
    if os.path.exists(path):
        return json.load(open(path))
    env = ta.make(SUBTASKS[subtask] + "-raw")
    out = sorted(w for w in set(env.word_list) if w == w.lower() and env._check_word(w))
    json.dump(out, open(path, "w"))
    return out


def instances(subtask, split, limit=None):
    out = [w for w in _words(subtask) if hash_split(f"{BENCH}/{subtask}/{w}") == split]
    return out[:limit] if limit else out


class WordleEnv(TextEnv):
    bench = BENCH

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        self._env = ta.make(SUBTASKS[subtask] + "-raw")
        # reset() reseeds the global RNG (TextArena's state does); keep it inside the shared lock
        seeded(0, self._env.reset, num_players=1, seed=0)
        self._env.state.game_state["secret_word"] = instance
        self.max_steps = self._env.num_guesses + 2     # room for the one allowed invalid move
        _, obs = self._env.get_observation()
        self.instructions = "\n".join(m for _, m, _ in obs)
        self._reward = 0.0
        self._reason = ""

    def _act(self, action):
        done, _ = self._env.step(action)
        _, obs = self._env.get_observation()
        text = "\n".join(m for frm, m, kind in obs if kind != ta.ObservationType.PLAYER_ACTION)
        if done:
            rewards, info = self._env.close()
            self.raw_reward = float(rewards[0])
            self._reward = max(self.raw_reward, 0.0)   # TextArena pays -1 for a loss by invalid moves
            self._reason = info[0].get("reason", "")
            self.done = True
            text = (text + "\n" + self._reason).strip()
        return text

    def score(self):
        return self._reward

    def result(self):
        return {**super().result(), "raw_reward": getattr(self, "raw_reward", None)}

    def success(self):
        hist = self._env.state.game_state.get("guess_history", [])
        return bool(hist) and hist[-1][0] == self._instance

    def _oracle_plan(self):
        return [f"[{self._instance}]"]


def make(subtask, instance, responder=None):
    return WordleEnv(subtask, instance, responder)
