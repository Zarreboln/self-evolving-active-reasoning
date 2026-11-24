"""The blicket detector, after CannyLab's causal_overhypotheses (envs/causal_env_v0.py, NeurIPS 2022
Datasets & Benchmarks), in text.

The semantics are that environment's, re-stated here without its gym/numpy observation code:
  * a hypothesis is a set of blickets and a structure; a disjunctive detector turns on when ANY of
    its blickets is on it, a conjunctive one only when ALL of them are (`test` in the original)
  * the agent puts any subset of objects on the detector per step, for at most `max_steps` = 20
    steps, and may end exploration early (the original's last action bit)
  * then the quiz of the "quiz-type" reward structure: each object, blicket or not (+1 / -1), and
    the structure, conjunctive or disjunctive (+0.5 / -0.5)

Subtasks:
  official   3 objects, the original's default hypothesis list: AB, AC, BC conjunctive; A, B, C
             disjunctive (6 instances in all)
  n4, n5     4 and 5 objects (ours): every disjunctive blicket set of size 1..n-1 and every
             conjunctive set of size 2..n, the same two structures over more objects; 5 objects is
             the size later text versions of the task use. The official list has too few
             instances to split.
Splits: a stable hash of the hypothesis, as everywhere else.

Text protocol (ours; the original is a vector environment): `test: A, C` (or `test: none`) puts
exactly those objects on the detector and reports whether it is ON; `answer: blickets = A, C; rule
= conjunctive` ends the episode. Once the step limit is reached only an answer is accepted.

score   = quiz reward normalised to [0, 1]: (sum of the item rewards + max penalty) / range
success = every quiz item right
"""
from __future__ import annotations

import itertools
import re

from mb.core import TextEnv, hash_split

BENCH = "blicket"
SUBTASKS = {"official": 3, "n4": 4, "n5": 5}
DIMENSIONS = {"world": "intervention (detector contents)", "hypotheses": "structured (set x rule)",
              "observation": "rule", "cost": "tests", "stop": "self"}
RESPONDER = None
MAX_STEPS = 20
_OFFICIAL = [("conjunctive", "AB"), ("conjunctive", "AC"), ("conjunctive", "BC"),
             ("disjunctive", "A"), ("disjunctive", "B"), ("disjunctive", "C")]


def _hypotheses(subtask):
    n = SUBTASKS[subtask]
    if subtask == "official":
        return list(_OFFICIAL)
    objs = "ABCDEFGH"[:n]
    out = [("disjunctive", "".join(c)) for k in range(1, n) for c in itertools.combinations(objs, k)]
    out += [("conjunctive", "".join(c)) for k in range(2, n + 1) for c in itertools.combinations(objs, k)]
    return out


def instances(subtask, split, limit=None):
    if subtask == "official":          # six instances: every split gets all of them, for smoke/eval only
        out = [f"{r}:{b}" for r, b in _OFFICIAL]
    else:
        out = [f"{r}:{b}" for r, b in _hypotheses(subtask) if hash_split(f"{BENCH}/{subtask}/{r}:{b}") == split]
    return out[:limit] if limit else out


class BlicketEnv(TextEnv):
    bench = BENCH

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        self._n = SUBTASKS[subtask]
        self._objs = "ABCDEFGH"[:self._n]
        self._rule, b = instance.split(":")
        self._blickets = set(b)
        self.max_steps = MAX_STEPS + 3
        self._tests = 0
        self._reward = None
        objs = ", ".join(self._objs)
        self.instructions = (
            f"There are {self._n} objects: {objs}. There is also a detector. Some of the objects are "
            "blickets. Whether the detector turns ON depends on which blickets are on it, following a "
            "rule you have to work out: either it turns on when ANY blicket is on it (disjunctive), or "
            "only when ALL of the blickets are on it together (conjunctive). Objects that are not "
            "blickets never affect the detector.\n\n"
            "Each turn, choose exactly which objects to place on the detector, written as\n"
            "test: A, C\n(or `test: none`). You will be told whether the detector is ON or OFF. "
            f"You can run at most {MAX_STEPS} tests. When you are ready, end the game with\n"
            "answer: blickets = A, C; rule = conjunctive\nlisting every object you believe is a "
            "blicket and the rule (conjunctive or disjunctive).")

    def _on(self, placed):
        if self._rule == "disjunctive":
            return bool(self._blickets & placed)
        return self._blickets <= placed

    def _act(self, action):
        a = action or ""
        m = re.search(r"answer\s*:\s*blickets\s*=\s*([A-Za-z,\s]*?)\s*;\s*rule\s*=\s*(\w+)", a, re.I)
        if m:
            said = {c for c in m.group(1).upper() if c in self._objs}
            rule = m.group(2).lower()
            r = sum(1 if ((o in said) == (o in self._blickets)) else -1 for o in self._objs)
            r += 0.5 if rule.startswith(self._rule[:4]) else -0.5
            self._reward = r
            self.done = True
            return "Answer recorded."
        if self._tests >= MAX_STEPS:
            return "No tests left. Give your answer as `answer: blickets = ...; rule = ...`."
        m = re.search(r"test\s*:\s*(.*)", a, re.I)
        if not m:
            return "Invalid action. Write `test: A, C` or `answer: blickets = A, C; rule = conjunctive`."
        body = m.group(1).strip().upper()
        placed = set() if body.startswith("NONE") else {c for c in re.findall(r"\b([A-H])\b", body) if c in self._objs}
        self._tests += 1
        on = ", ".join(sorted(placed)) or "nothing"
        return (f"On the detector: {on}. The detector is {'ON' if self._on(placed) else 'OFF'}. "
                f"{MAX_STEPS - self._tests} tests left.")

    def score(self):
        if self._reward is None:
            return 0.0
        lo, hi = -(self._n + 0.5), self._n + 0.5
        return (self._reward - lo) / (hi - lo)

    def success(self):
        return self._reward == self._n + 0.5

    def _oracle_plan(self):
        b = ", ".join(sorted(self._blickets))
        return [f"test: {self._objs[0]}", f"answer: blickets = {b}; rule = {self._rule}"]


def make(subtask, instance, responder=None):
    return BlicketEnv(subtask, instance, responder)
