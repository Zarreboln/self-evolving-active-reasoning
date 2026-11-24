"""AR-Bench's three tasks (GN, DC, SP) behind the shared interface.

The environments are the ones the Meta-Harness runs already use (~/meta/arbench_mh/{gn,dc,sp}/env.py),
which import AR-Bench's own prompts, role templates and scorers; nothing is restated here. What
this adapter adds is a text action protocol, because those environments expose task-specific
methods (`guess`, `interrogate`, `ask`) and the suite needs one `act(text)`:

  gn   a 4-digit guess
  dc   `ask <suspect name>: <question>`  or  `accuse <letter>`
  sp   `ask: <question>`                 or  `answer: <explanation>`

Each task is one subtask (`all`): GN has no subtask structure, DC has none in the data (every case
has five suspects and no type field), and SP's two flags are balanced only in its test split, so
splitting on them would mean carving search data out of test.

Splits are the Meta-Harness splits: search / selection from ~/arbench_mh_splits, test from
~/sealed, which sits outside any proposer's working directory.

score   gn: exact match (AR-Bench)   dc: correctness (AR-Bench)   sp: character F1 (AR-Bench)
success gn, dc: same as score         sp: None (AR-Bench defines none)
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys

sys.path.insert(0, os.path.expanduser(os.environ.get("ARBENCH_MH_ROOT", "~/meta")))

from arbench_mh.dc.env import DCEnv as _DC  # noqa: E402
from arbench_mh.gn.env import GNEnv as _GN  # noqa: E402
from arbench_mh.sp.env import SPEnv as _SP  # noqa: E402

from mb.core import TextEnv  # noqa: E402

SUBTASKS = ["all"]
_LETTERS = "ABCDE"


@functools.lru_cache(maxsize=None)
def _items(task, split):
    root = "~/sealed" if split == "test" else "~/arbench_mh_splits"
    return json.load(open(os.path.expanduser(f"{root}/{task}/{split}.json")))["instances"]


def _instances(task, split, limit=None):
    out = [f"{split}/{i}" for i in range(len(_items(task, split)))]
    return out[:limit] if limit else out


def _item(task, instance):
    split, i = instance.split("/")
    return _items(task, split)[int(i)]


class GNEnv(TextEnv):
    bench = "gn"

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        self._env = _GN(_item("gn", instance))
        self.max_steps = self._env.budget
        self.instructions = (self._env.system_prompt
                             + "\nEach turn, reply with one 4-digit guess and nothing else.")

    def _act(self, action):
        m = re.search(r"\b(\d{4})\b", action)
        if not m:
            return "Invalid guess: reply with exactly four digits."
        a, b = self._env.guess(m.group(1))
        if self._env.solved:
            self.done = True
        return self._env.record.strip().splitlines()[-1]

    def score(self):
        return 1.0 if self._env.solved else 0.0

    def success(self):
        return self._env.solved

    def _oracle_plan(self):
        return ["1234", "".join(map(str, self._env._target))]


class DCEnv(TextEnv):
    bench = "dc"

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        if responder is None:
            raise ValueError("dc needs a responder to play the suspects")
        self._env = _DC(_item("dc", instance), responder)
        self.max_steps = self._env.budget + 3
        choices = "\n".join(f"{c}. {n}" for c, n in zip(_LETTERS, self._env.suspects))
        self.instructions = (
            self._env.system_prompt
            + f"\n\nSuspects:\n{choices}\n\nEach turn, either question one suspect, written as\n"
            "ask <suspect name>: <question>\nor name the murderer, written as\naccuse <letter>\n"
            f"You may ask at most {self._env.budget} questions in total. Accusing ends the case.")
        self._pred = None

    def _act(self, action):
        m = re.match(r"\s*accuse\s*:?\s*([A-Ea-e])\b", action)
        if m:
            self._pred = _LETTERS.index(m.group(1).upper())
            self.done = True
            return "Accusation recorded."
        m = re.match(r"\s*ask\s+(.+?)\s*:\s*(.+)", action, re.S)
        if not m:
            return "Invalid action: write `ask <suspect name>: <question>` or `accuse <letter>`."
        if self._env.remaining <= 0:
            return "No questions left. Name the murderer with `accuse <letter>`."
        who = m.group(1).strip().lower()
        name = next((n for n in self._env.suspects if n.lower() == who), None) or next(
            (n for n in self._env.suspects if who in n.lower() or n.lower() in who), None)
        if name is None:
            return f"No suspect called {m.group(1)!r}. The suspects are: {', '.join(self._env.suspects)}."
        return f"{name}: {self._env.interrogate(name, m.group(2).strip())}"

    def score(self):
        return 1.0 if self._pred is not None and self._env.score(self._pred)["correctness"] else 0.0

    def success(self):
        return self.score() == 1.0

    def _oracle_plan(self):
        return [f"ask {self._env.suspects[0]}: Where were you that evening?",
                f"accuse {_LETTERS[self._env._label]}"]


class SPEnv(TextEnv):
    bench = "sp"

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        if responder is None:
            raise ValueError("sp needs a responder to play the referee")
        self._env = _SP(_item("sp", instance), responder)
        self.max_steps = self._env.budget + 3
        self.instructions = (
            self._env.system_prompt
            + "\n\nEach turn, either ask the referee one yes/no question, written as\n"
            "ask: <question>\nor give your final explanation, written as\nanswer: <explanation>\n"
            f"You may ask at most {self._env.budget} questions. Answering ends the puzzle.")
        self._answer = None

    def _act(self, action):
        m = re.match(r"\s*answer\s*:\s*(.+)", action, re.S)
        if m:
            self._answer = m.group(1).strip()
            self.done = True
            return "Answer recorded."
        m = re.match(r"\s*ask\s*:\s*(.+)", action, re.S)
        if not m:
            return "Invalid action: write `ask: <question>` or `answer: <explanation>`."
        if self._env.remaining <= 0:
            return "No questions left. Give your explanation with `answer: <explanation>`."
        return self._env.ask(m.group(1).strip())

    def score(self):
        return self._env.score(self._answer)["f1_score_char"] if self._answer else 0.0

    def success(self):
        return None

    def _oracle_plan(self):
        return ["ask: Did someone die?", f"answer: {self._env._bottom}"]


_CLS = {"gn": GNEnv, "dc": DCEnv, "sp": SPEnv}
DIMENSIONS = {
    "gn": {"world": "no", "hypotheses": "enumerable", "observation": "rule",
           "cost": "guesses", "stop": "self"},
    "dc": {"world": "no", "hypotheses": "structured", "observation": "model answer",
           "cost": "questions", "stop": "self"},
    "sp": {"world": "no", "hypotheses": "open", "observation": "model answer",
           "cost": "questions", "stop": "self"},
}
RESPONDER = {"gn": None, "dc": "llm", "sp": "llm"}


def instances(task, subtask, split, limit=None):
    return _instances(task, split, limit)


def make(task, subtask, instance, responder=None):
    return _CLS[task](subtask, instance, responder)
