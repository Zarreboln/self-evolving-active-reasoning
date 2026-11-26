"""AR-Bench's three tasks (GN, DC, SP) behind the shared interface.

The environments are the ones the Meta-Harness runs already use (~/meta/arbench_mh/{gn,dc,sp}/env.py),
which import AR-Bench's own prompts, role templates and scorers; nothing is restated here. GN and DC
follow AR-Bench's zero-shot dialogue exactly (prompts, turn count, parsers, scoring; see each class);
the env is the user side of that conversation. SP still uses this adapter's own text protocol
(`ask: <question>` or `answer: <explanation>`, early answer allowed), which is not AR-Bench's.

Each task is one subtask (`all`): GN has no subtask structure, DC has none in the data (every case
has five suspects and no type field), and SP's two flags are balanced only in its test split, so
splitting on them would mean carving search data out of test.

Splits are the Meta-Harness splits: search / selection from ~/arbench_mh_splits, test from
~/sealed, which sits outside any proposer's working directory.

score   gn: the final guess is exact (AR-Bench)   dc: correctness (AR-Bench)   sp: character F1 (AR-Bench)
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
from arbench_mh.sp.env import SPEnv as _SP  # noqa: E402

from mb.core import TextEnv  # noqa: E402

sys.path.insert(0, os.path.expanduser(os.environ.get("ARBENCH_ROOT", "~/code/AR-Bench")))
from arbench.reasoner.dc import dc_evaluator as _dce  # noqa: E402
from arbench.reasoner.dc import prompt as dc_prompt  # noqa: E402
from arbench.reasoner.dc.dc_evaluator import extract_answer_choice, remove_punctuation_at_ends  # noqa: E402
from arbench.reasoner.gn import prompt as gn_prompt  # noqa: E402
from arbench.utils.utils_gn import compare_guess, extract_and_convert_guess  # noqa: E402

GN_TURNS = DC_TURNS = 25      # AR-Bench's max_turn
# the case-background formatter is renamed between AR-Bench revisions
_dc_background = getattr(_dce, "convert_initial_info_to_string", None) or _dce._format_initial_info

SUBTASKS = ["all"]
_LETTERS = "ABCDE"


@functools.lru_cache(maxsize=None)
def _items(task, split):
    root = (os.environ.get("ARBENCH_SEALED", "~/sealed") if split == "test"
            else os.environ.get("ARBENCH_SPLITS", "~/arbench_mh_splits"))
    return json.load(open(os.path.expanduser(f"{root}/{task}/{split}.json")))["instances"]


def _instances(task, split, limit=None):
    out = [f"{split}/{i}" for i in range(len(_items(task, split)))]
    return out[:limit] if limit else out


def _item(task, instance):
    split, i = instance.split("/")
    return _items(task, split)[int(i)]


class GNEnv(TextEnv):
    """AR-Bench's zero-shot GN dialogue, message for message (gn_evaluator._run_traditional_evaluation).

    The system prompt is `propose_template`; each turn the env asks with `guess_prompt`, the reply is
    parsed by AR-Bench's `extract_and_convert_guess`, and the feedback is `eval_prompt`. All 25 turns
    are played whatever happens; then `final_guess_prompt`, and the episode is correct iff that final
    guess has 4 digits in place (`guess_list[-1][1] == 4`). A reply with no 4-digit number gets
    `refine_prompt` and uses up the turn, which is what AR-Bench's NotNumberError branch intends (its
    extractor returns [] instead of raising, so the original crashes there). Where AR-Bench sends two
    user messages in a row (feedback, then the next turn's prompt) they arrive as one observation.
    """
    bench = "gn"

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        self._target = [int(d) for d in str(_item("gn", instance)).zfill(4)]
        self.max_steps = GN_TURNS + 1
        self.instructions = gn_prompt.propose_template.format(turn=GN_TURNS)
        self.observation = gn_prompt.guess_prompt.format(turn=1)
        self._final = None

    def _act(self, action):
        guess = extract_and_convert_guess(action or "")
        last = self.steps == GN_TURNS + 1
        if len(guess) == 4:
            same, diff, _ = compare_guess(self._target, guess)
        else:
            same, diff = None, None
        if last:
            self._final = same
            self.done = True
            return ""
        if same is None:
            fb = gn_prompt.refine_prompt
        else:
            fb = gn_prompt.eval_prompt.format(same_pos=same, diff_pos=diff)
        nxt = (gn_prompt.final_guess_prompt if self.steps == GN_TURNS
               else gn_prompt.guess_prompt.format(turn=self.steps + 1))
        return fb + "\n" + nxt

    def score(self):
        return 1.0 if self._final == 4 else 0.0

    def success(self):
        return self._final == 4

    def _oracle_plan(self):
        t = "".join(map(str, self._target))
        return [f"Guess: {t}"] * (GN_TURNS + 1)


class DCEnv(TextEnv):
    """AR-Bench's zero-shot DC dialogue, message for message (dc_evaluator._run_traditional_evaluation).

    System prompt `propose_template` over the case background. Each of the 25 turns takes two
    actions: the suspect's name in reply to `select_suspect_template` (one punctuation mark is
    stripped from each end, then it must equal a name exactly; otherwise `refine_select_suspect_prompt`
    and the turn is used up), then the question in reply to `question_propose_prompt`, which the
    suspect, played from AR-Bench's `respond_template` with their own running conversation, answers.
    After 25 turns `select_murderer_template`; the verdict is read by AR-Bench's `extract_answer_choice`.
    """
    bench = "dc"

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        if responder is None:
            raise ValueError("dc needs a responder to play the suspects")
        item = _item("dc", instance)
        self._env = _DC(item, responder)
        self._names = self._env.suspects
        self._label = item["label"]
        self._choices = ", ".join(f"{c}. {n}" for c, n in zip(_LETTERS, self._names))
        self.max_steps = 2 * DC_TURNS + 1
        self.instructions = dc_prompt.propose_template.format(
            background=_dc_background(item["initial_information"]), turn=DC_TURNS)
        self._turn, self._phase, self._who, self._pred = 1, "select", None, None
        self.observation = self._select_prompt()

    def _select_prompt(self):
        if self._turn > DC_TURNS:
            self._phase = "verdict"
            return dc_prompt.select_murderer_template.format(choice=self._choices)
        self._phase = "select"
        return dc_prompt.select_suspect_template.format(turn=self._turn, suspect_names=", ".join(self._names))

    def _act(self, action):
        if self._phase == "verdict":
            letter = extract_answer_choice(action or "").strip()
            self._pred = _LETTERS.index(letter) if letter in _LETTERS else None
            self.done = True
            return ""
        if self._phase == "select":
            name = remove_punctuation_at_ends(action or "")
            if name not in self._names:
                self._turn += 1
                return dc_prompt.refine_select_suspect_prompt + "\n" + self._select_prompt()
            self._who, self._phase = name, "question"
            return dc_prompt.question_propose_prompt.format(turn=self._turn)
        reply = self._env.interrogate(self._who, action)
        self._turn += 1
        return reply + "\n" + self._select_prompt()

    def score(self):
        return 1.0 if self._pred is not None and self._pred == self._label else 0.0

    def success(self):
        return self.score() == 1.0

    def _oracle_next(self):
        if self._phase == "verdict":
            return f"Reason: -\nAnswer: {_LETTERS[self._label]}"
        return self._names[0] if self._phase == "select" else "Where were you that evening?"


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
           "cost": "guesses", "stop": "fixed (25 turns)"},
    "dc": {"world": "no", "hypotheses": "structured", "observation": "model answer",
           "cost": "questions", "stop": "fixed (25 turns)"},
    "sp": {"world": "no", "hypotheses": "open", "observation": "model answer",
           "cost": "questions", "stop": "self"},
}
RESPONDER = {"gn": None, "dc": "llm", "sp": "llm"}


def instances(task, subtask, split, limit=None):
    return _instances(task, split, limit)


def make(task, subtask, instance, responder=None):
    return _CLS[task](subtask, instance, responder)
