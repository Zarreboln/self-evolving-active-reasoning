"""Situation Puzzles (AR-Bench, arXiv 2506.08295) as a TTHE environment.

The hidden state is the puzzle's `bottom` (the full story); the player sees only the `surface`.
The REFEREE (AR-Bench: Llama-3.1-405B; here any served model, see --answerer-*) answers each
question with Yes / No / Unknown from surface + bottom using AR-Bench's own referee prompt. After
at most BUDGET questions the player gives a free-text answer; the official outcome metric is the
character-level F1 against `bottom` (utils_sp.f1_score); the process metric counts which of the
puzzle's key questions the player's questions hit (keypoint_hits_prompt, judged by the referee).

Public to a harness: surface, system_prompt (AR-Bench propose_template), turn_prompt(t),
final_prompt (get_answer_prompt), ask(question) -> referee reply, remaining, history.
Private: _bottom, _keypoints; score() is controller-only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "AR-Bench"))
from arbench.reasoner.sp.prompt import (  # noqa: E402
    propose_template, system_prompt_with_2shots, get_answer_prompt, keypoint_hits_prompt)
from arbench.utils.utils_sp import f1_score, place_keypoints  # noqa: E402

from bridge import Solver  # noqa: E402   (same cached OpenAI-compatible client, used for the referee too)

BUDGET = 25
DATA = ROOT / "AR-Bench/data/sp/test.json"
LLAMA3_STOP_IDS = [128001, 128008, 128009]
_HIT_RE = re.compile(r"Hit point:\s*([\d,\s]+)")


def load_instances():
    return json.loads(DATA.read_text())


INSTANCES = load_instances()


class BudgetExhausted(Exception):
    pass


class SPEnv:
    def __init__(self, index, budget=BUDGET, answerer=None):
        inst = INSTANCES[index]
        self.index, self.budget = index, budget
        self.surface = inst["surface"]
        self._bottom = inst["bottom"]
        self._keypoints = inst["key_question"] if isinstance(inst["key_question"], list) else eval(inst["key_question"])
        self._answerer = answerer                      # callable(messages, temperature, max_tokens) -> str
        self.history = []                              # [(question, answer)]
        self.events = []
        self.system_prompt = propose_template.format(turn=budget, question=self.surface)
        self.final_prompt = get_answer_prompt
        self._respond_sys = {"role": "system", "content": system_prompt_with_2shots.format(question=self.surface, answer=self._bottom)}

    @property
    def remaining(self):
        return self.budget - len(self.history)

    @staticmethod
    def turn_prompt(t):
        return f"Turn {t}: please propose your next question."

    def ask(self, question):
        """One question to the referee (AR-Bench wording). Counts against the budget."""
        if self.remaining <= 0:
            raise BudgetExhausted(f"budget of {self.budget} questions used")
        q = (question or "").strip()
        ans = self._answerer([self._respond_sys, {"role": "user", "content": q}], temperature=0.7, max_tokens=16)
        ans = (ans or "").strip().strip(".")
        self.history.append((q, ans))
        self.events.append(("ask", {"q": q, "a": ans}))
        return ans

    # ---- controller-side scoring (never called by a harness) ----
    def score(self, prediction, judge_keypoints=True):
        pred = (prediction or "").strip()
        out = {"parsed_answer": bool(pred), "f1_char": f1_score(pred, self._bottom), "f1_word": f1_score(pred.split(), self._bottom.split()),
               "n_keypoints": len(self._keypoints), "keypoints_hit": None, "prediction": pred[:1000]}
        if judge_keypoints and self.history and self._answerer is not None:
            hit = set()
            for q, _ in self.history:
                txt = self._answerer([{"role": "user", "content": keypoint_hits_prompt.format(
                    question=self.surface, answer=self._bottom, keypoints=place_keypoints(self._keypoints)) + f"\n\nPlayer's question: {q}"}],
                    temperature=0.0, max_tokens=32)
                m = _HIT_RE.search(txt or "")
                if m:
                    hit |= {int(x) for x in re.findall(r"\d+", m.group(1))}
            out["keypoints_hit"] = sorted(hit)
            out["keypoint_recall"] = len(hit) / max(1, len(self._keypoints))
        return out


class SPMockSolver:
    """CPU stand-in for the questioner: a question per turn, a short answer at the end."""
    def __init__(self):
        self.calls = 0; self.tokens_in = self.tokens_out = 0
    def __call__(self, messages, temperature=0.7, max_tokens=512):
        self.calls += 1
        last = messages[-1]["content"]
        if "give your answer" in last.lower() or "A: [" in last:
            return "A: The man faked his own death to escape and wrote from hiding."
        return f"Q: Was something hidden from the family (turn {self.calls})?"


class SPMockAnswerer:
    """CPU stand-in for the referee: cycles Yes / No / Unknown."""
    def __init__(self):
        self.i = 0; self.calls = 0; self.tokens_in = self.tokens_out = 0
    def __call__(self, messages, temperature=0.7, max_tokens=16):
        self.calls += 1; self.i += 1
        if "Hit point" in messages[-1]["content"] or "keypoints" in messages[-1]["content"]:
            return "Hit point: 1"
        return ["Yes", "No", "Unknown"][self.i % 3]


def make_answerer(cfg, cache_dir):
    """cfg: {"base_url", "model", "mock", "seed"}; replies cached like the solver's."""
    if cfg.get("mock"):
        return SPMockAnswerer()
    s = Solver(base_url=cfg["base_url"], model=cfg["model"], cache_dir=cache_dir, seed=int(cfg.get("seed", 0)))
    return s


def make_stream(split="test", split_file=None, seed=0, limit=None):
    import numpy as np
    keys = list(range(len(INSTANCES)))
    keys = [int(k) for k in np.random.default_rng(seed).permutation(keys)]
    return keys[:limit] if limit else keys
