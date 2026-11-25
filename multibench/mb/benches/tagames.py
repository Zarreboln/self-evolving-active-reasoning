"""More pure-information games from TextArena: Mastermind, Hangman, Minesweeper, Guess the Number,
Guess Who. Each is a separate bench; its subtasks are TextArena's own registered variants.

Everything that defines the game is TextArena's: the secret's generation, the feedback text, the
invalid-move rule (one error allowed, the second loses), the turn limits and the reward. The
adapter does three things only:

  * Instances are seeds, which fix the secret through TextArena's own seeding. Seeds whose secret
    repeats an earlier one are skipped, and the split is decided by the secret (not the seed), so
    no secret appears in two splits. Minesweeper places its mines only after the first click, so
    there the key is the seed.
  * The observation after each move is the game's messages plus the current board when the game
    keeps one (`rendered_board`), which is what TextArena's own board wrapper shows; the raw env
    sends the move confirmation alone.
  * Guess Who's gamemaster, hard-coded to GPT-4o through OpenRouter, is the run's pinned responder
    -- Jev given the target's trait record, or a chat model with TextArena's prompt -- answering
    with TextArena's options, the same change as twentyq.

Two TextArena rules worth knowing when reading scores: Minesweeper counts hitting a mine as an
invalid move, so with one error allowed the first mine does not end the game; and every game here
pays partial credit on a loss (its `_get_percentage_completion`).

score   = max(TextArena's reward, 0): 1 on a win, partial credit otherwise; TextArena pays -1 for a
          loss by invalid moves (e.g. two wrong names in Guess Who), kept as `raw_reward`
success = reward == 1
"""
from __future__ import annotations

import functools
import json
import os

import textarena as ta

from mb.core import TextEnv, hash_split, seeded

GAMES = {
    "mastermind": {"basic": "Mastermind-v0", "hard": "Mastermind-v0-hard",
                   "extreme": "Mastermind-v0-extreme"},
    "hangman": {"basic": "Hangman-v0", "hard": "Hangman-v0-hardcore"},
    "minesweeper": {"small": "Minesweeper-v0-small", "basic": "Minesweeper-v0",
                    "medium": "Minesweeper-v0-medium", "hard": "Minesweeper-v0-hard"},
    "guessnumber": {"basic": "GuessTheNumber-v0", "hard": "GuessTheNumber-v0-hardcore"},
    "guesswho": {"all": "GuessWho-v0"},
}
DIMENSIONS = {
    "mastermind": {"world": "no", "hypotheses": "enumerable", "observation": "rule",
                   "cost": "guesses", "stop": "self"},
    "hangman": {"world": "no", "hypotheses": "enumerable", "observation": "rule",
                "cost": "wrong letters", "stop": "self"},
    "minesweeper": {"world": "no", "hypotheses": "structured", "observation": "rule",
                    "cost": "irreversible (mines)", "stop": "board cleared"},
    "guessnumber": {"world": "no", "hypotheses": "enumerable", "observation": "rule",
                    "cost": "guesses", "stop": "self"},
    "guesswho": {"world": "no", "hypotheses": "structured", "observation": "model yes/no",
                 "cost": "questions", "stop": "self"},
}
RESPONDER = {g: ("llm" if g == "guesswho" else None) for g in GAMES}
_SECRET = {
    "mastermind": lambda e: " ".join(map(str, e.state.game_state["secret_code"])),
    "hangman": lambda e: e.state.game_state["target_word"],
    "guessnumber": lambda e: str(e.game_number),
    "guesswho": lambda e: e.target_character["name"],
    "minesweeper": None,
}
_SCAN = 20_000         # seeds examined when listing instances; small secret spaces run out first


class _NoAgent:
    def __init__(self, *a, **k):
        pass


def _build(game, subtask):
    real, ta.agents.OpenRouterAgent = ta.agents.OpenRouterAgent, _NoAgent
    try:
        return ta.make(GAMES[game][subtask] + "-raw")
    finally:
        ta.agents.OpenRouterAgent = real


@functools.lru_cache(maxsize=None)
def _catalogue(game, subtask, per_split=200):
    """{split: [seed, ...]}, at most `per_split` each, every secret in exactly one split. Cached."""
    path = os.path.join(os.environ.get("MBENCH_DATA", "."), f"tagames_{game}_{subtask}.json")
    if os.path.exists(path):
        return json.load(open(path))
    env, key = _build(game, subtask), _SECRET[game]
    out, seen = {"search": [], "selection": [], "test": []}, set()
    for seed in range(_SCAN):
        if all(len(v) >= per_split for v in out.values()):
            break
        seeded(seed, env.reset, num_players=1, seed=seed)
        k = key(env) if key else str(seed)
        if k in seen:
            continue
        seen.add(k)
        split = hash_split(f"{game}/{subtask}/{k}")
        if len(out[split]) < per_split:
            out[split].append(seed)
    json.dump(out, open(path, "w"))
    return out


class TAGameEnv(TextEnv):
    reads_full_reply = True
    def __init__(self, game, subtask, instance, responder=None):
        self.bench = game
        super().__init__(subtask, instance)
        self._game = game
        self._env = _build(game, subtask)
        if RESPONDER[game]:
            if responder is None:
                raise ValueError(f"{game} needs a responder to play the gamemaster")
            if getattr(responder, "jev_role", None) == "gamemaster":
                # Jev answers from the target's trait record (read at call time, after reset)
                from mb.benches.twentyq import _question
                self._env.gamemaster = lambda prompt: responder.answer(
                    "Guess Who", json.dumps(self._env.target_character), _question(prompt))
            else:
                self._env.gamemaster = lambda prompt: responder(
                    [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64)
        self._seed = int(instance)
        seeded(self._seed, self._env.reset, num_players=1, seed=self._seed)
        self.max_steps = (self._env.state.max_turns or 50) + 2   # room for the allowed error
        _, obs = self._env.get_observation()
        self.instructions = "\n".join(m for _, m, _ in obs) + self._board()
        self._reward = 0.0

    def _board(self):
        b = self._env.state.game_state.get("rendered_board")
        return f"\n\nCurrent board:\n{b}" if b else ""

    def _act(self, action):
        # Minesweeper lays its mines on the first click, from the global RNG: seed every step.
        done, _ = seeded(self._seed * 1000 + self.steps, self._env.step, action)
        _, obs = self._env.get_observation()
        text = "\n".join(m for frm, m, kind in obs if kind != ta.ObservationType.PLAYER_ACTION)
        if done:
            rewards, info = self._env.close()
            self.raw_reward = float(rewards[0])
            self._reward = max(self.raw_reward, 0.0)   # TextArena pays -1 for a loss by invalid moves
            self.done = True
            return (text + "\n" + info[0].get("reason", "")).strip()
        return text + self._board()

    def score(self):
        return self._reward

    def result(self):
        return {**super().result(), "raw_reward": getattr(self, "raw_reward", None)}

    def success(self):
        return self._reward == 1.0

    def _oracle_next(self):
        e = self._env
        if self._game == "minesweeper":
            if e.first_move:
                return f"[{e.rows // 2} {e.cols // 2}]"
            for r in range(e.rows):
                for c in range(e.cols):
                    if e.grid[r][c] != -1 and not e.revealed[r][c]:
                        return f"[{r} {c}]"
            return None
        secret = _SECRET[self._game](e)
        return f"[{secret}]"


def instances(game, subtask, split, limit=None):
    out = [str(s) for s in _catalogue(game, subtask)[split]]
    return out[:limit] if limit else out


def make(game, subtask, instance, responder=None):
    return TAGameEnv(game, subtask, instance, responder)
