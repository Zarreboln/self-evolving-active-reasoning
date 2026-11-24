"""Multi-Turn Puzzles (arXiv 2508.10142, data arianhosseini/mt_puzzles, CC-BY-4.0): Circuit Decoding
and Movie Recommendation, the two of its five tasks where the model gathers information.

Only the data and the instruction files are released, not the environments. Both environments
follow the paper's appendix transcripts, and every choice the release leaves open is marked
(ours) below. The data ships as a single test file per task; splits are a stable hash of the
instance key.

Circuit Decoding -- same protocol and scoring as meta/cd_tthe (bridge.py, agents/bare.py), see
experiments/OFFICIAL_DEFINITIONS.md:
  * system prompt: cd_instruction.txt, placeholders filled as bridge.py fills them
  * action `<query>X(a, b, c)</query>`; reply `X(a, b, c) = bit` + "Please send your next query.
    N queries remaining."; after the 18th, the reply carries cd_last_turn_prompt.txt instead
  * a reply with no valid query spends the turn and gets bare.py's re-ask line (ours)
  * final action `<guess>` + 24 bits
  score   = circuit-wise accuracy (the official metric's name; its normalisation is unspecified,
            so it is reported unnormalised, as in OFFICIAL_DEFINITIONS)
  success = joint truth table correct (our literal reading of a solved episode)

Movie Recommendation -- instruction.txt and last_turn_prompt.txt as released:
  * {last_turn_instruction}: the sentence of the paper's appendix transcript
  * {example_user_preference_function_str}: a weighted sum over the attribute names with fixed
    example weights (ours; elided in the transcript)
  * movie tables: one row per movie, name then the 8 scores (ours; elided in the transcript)
  * action `<question>Would you prefer watching A over B?</question>`; reply "Yes" / "No" /
    "No Preference" from the hidden linear function, then "Please ask the next question. N
    questions remaining." (transcript wording); after the 10th, last_turn_prompt with the 40
    unseen movies; final action `<final_answer>name</final_answer>`
  * a reply with no valid question spends the turn and is told the format (ours)
  score   = normalized rank of the final movie (official name): (N - rank) / (N - 1) over the
            N = 40 unseen movies, rank 1 = the user's best; 0 if no valid pick (normalisation ours)
  success = the pick is the user's best movie (ours)
"""
from __future__ import annotations

import functools
import json
import os
import re

import numpy as np

from mb.core import TextEnv, hash_split

_DIR = os.path.join(os.environ.get("MBENCH_DATA", "."), "mt_puzzles")
SUBTASKS = ["all"]
RESPONDER = {"circuit": None, "movierec": None}
DIMENSIONS = {
    "circuit": {"world": "no", "hypotheses": "structured (gate inventory)", "observation": "rule",
                "cost": "queries", "stop": "fixed budget"},
    "movierec": {"world": "no", "hypotheses": "continuous (linear weights)", "observation": "rule",
                 "cost": "questions", "stop": "fixed budget"},
}


def _read(name):
    return open(os.path.join(_DIR, name)).read()


@functools.lru_cache(maxsize=None)
def _rows(task):
    f = "cd_test.jsonl" if task == "circuit" else "mr_test.jsonl"
    return [json.loads(l) for l in open(os.path.join(_DIR, f))]


def _key(task, i, row):
    return f"{task}/{row['key']}" if task == "circuit" else f"{task}/{i}"


def instances(task, subtask, split, limit=None):
    out = [str(i) for i, r in enumerate(_rows(task)) if hash_split(_key(task, i, r)) == split]
    return out[:limit] if limit else out


# ------------------------------------------------------------------ Circuit Decoding
CD_BUDGET, CD_C, CD_K, CD_LABELS = 18, 3, 3, "ABC"
_QUERY = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")


def _cd_instruction(inventory):
    labels = list(CD_LABELS[:CD_C])
    comp = ", ".join(f"{v} {g} gate{'s' if v != 1 else ''}" for g, v in sorted(inventory.items()) if v)
    order = ", ".join(f"{lb}({','.join(str((r >> (CD_K - 1 - i)) & 1) for i in range(CD_K))})"
                      for lb in labels for r in range(1 << CD_K))
    return (_read("cd_instruction.txt").replace("{components}", comp)
            .replace("{num_circuits}", str(CD_C)).replace("{circuit_labels}", ", ".join(labels))
            .replace("{circuit_labels_str}", ", ".join(labels)).replace("{num_inputs}", str(CD_K))
            .replace("{num_questions_per_episode}", str(CD_BUDGET))
            .replace("{truth_table_description}", order))


class CircuitEnv(TextEnv):
    bench = "circuit"
    max_steps = CD_BUDGET + 1

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        c = _rows("circuit")[int(instance)]["controller"]
        self._truth = np.array([int(ch) for ch in c["circuit_joint_lookup"]], dtype=np.uint8)
        self.instructions = _cd_instruction(c["components"])
        self._queries = 0
        self._result = None

    def _act(self, action):
        if self._queries >= CD_BUDGET:
            bits = re.findall(r"[01]{24}", re.sub(r"\s", "", action or ""))
            guess = np.array([int(ch) for ch in bits[-1]], dtype=np.uint8) if bits else None
            self._result = {
                "parsed_guess": guess is not None,
                "joint_truth_table_correct": guess is not None and bool((guess == self._truth).all()),
                "circuit_wise_accuracy": (sum(int((guess[i * 8:(i + 1) * 8] == self._truth[i * 8:(i + 1) * 8]).all())
                                              for i in range(CD_C)) / CD_C) if guess is not None else 0.0}
            self.done = True
            return "Guess recorded."
        self._queries += 1
        left = CD_BUDGET - self._queries
        m = _QUERY.search(action or "")
        if m:
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            q = CD_LABELS.index(c) * (1 << CD_K) + (a << 2) + (b << 1) + x
            line = f"{c}({a}, {b}, {x}) = {int(self._truth[q])}"
        else:
            line = "Your message did not contain a query in the required format. Reply with exactly one <query>X(a, b, c)</query>."
        if left == 0:
            return line + "\n" + _read("cd_last_turn_prompt.txt")
        return f"{line}\nPlease send your next query. {left} queries remaining."

    def score(self):
        return (self._result or {}).get("circuit_wise_accuracy", 0.0)

    def success(self):
        return bool((self._result or {}).get("joint_truth_table_correct", False))

    def result(self):
        return {**super().result(), **(self._result or {})}

    def _oracle_next(self):
        if self._queries < CD_BUDGET:
            return "<query>A(0, 0, 0)</query>"
        return f"<guess>{''.join(map(str, self._truth))}</guess>"


# ------------------------------------------------------------------ Movie Recommendation
MR_BUDGET = 10
_LAST_TURN_INSTRUCTION = (
    "Only once you're done interacting with the user after asking {n} questions one-by-one, you will be "
    "given a new set of movies and based on whatever you've learnt about the user, your goal would be to "
    "pick the movie which you think the user will enjoy the most.")
_EXAMPLE_WEIGHTS = (0.3, 0.8, 0.1, 0.5, 0.0, 0.6, 0.2, 0.9)          # ours
_ASK = re.compile(r"<question>\s*Would you prefer watching\s+(.+?)\s+over\s+(.+?)\s*\?\s*</question>", re.I | re.S)
_FINAL = re.compile(r"<final_answer>\s*(.+?)\s*</final_answer>", re.S)


def _table(attrs, names, scores):
    head = "| Movie | " + " | ".join(attrs) + " |\n|" + "---|" * (len(attrs) + 1)
    return head + "\n" + "\n".join("| " + n + " | " + " | ".join(str(v) for v in s) + " |"
                                   for n, s in zip(names, scores))


class MovieRecEnv(TextEnv):
    bench = "movierec"
    max_steps = MR_BUDGET + 1

    def __init__(self, subtask, instance, responder=None):
        super().__init__(subtask, instance)
        r = _rows("movierec")[int(instance)]
        self._w = np.array(r["user_preference_function"], dtype=float)
        self._attrs = r["movie_attributes"]
        self._seen = dict(zip(r["seen_movies_names"], r["seen_movies_attribute_scores"]))
        self._unseen_names = r["unseen_movies_names"]
        self._unseen_scores = r["unseen_movies_attribute_scores"]
        example = " + ".join(f"{a}*{w}" for a, w in zip(self._attrs, _EXAMPLE_WEIGHTS))
        self.instructions = (_read("mr_instruction.txt")
                             .replace("{len(movie_attributes)}", str(len(self._attrs)))
                             .replace("{movie_attributes}", ", ".join(self._attrs))
                             .replace("{example_user_preference_function_str}", f"({example})")
                             .replace("{last_turn_instruction}", _LAST_TURN_INSTRUCTION.format(n=MR_BUDGET))
                             .replace("{num_questions_per_episode}", str(MR_BUDGET))
                             .replace("{seen_movie_sample}", _table(self._attrs, list(self._seen),
                                                                    list(self._seen.values()))))
        self._asked = 0
        self._pick = None

    def _utility(self, scores):
        return round(float(np.dot(self._w, scores)), 6)

    def _act(self, action):
        if self._asked >= MR_BUDGET:
            m = _FINAL.search(action or "")
            self._pick = m.group(1).strip() if m else None
            self.done = True
            return "Final answer recorded."
        self._asked += 1
        left = MR_BUDGET - self._asked
        m = _ASK.search(action or "")
        if m and m.group(1).strip() in self._seen and m.group(2).strip() in self._seen:
            u1, u2 = self._utility(self._seen[m.group(1).strip()]), self._utility(self._seen[m.group(2).strip()])
            reply = "Yes" if u1 > u2 else "No" if u1 < u2 else "No Preference"
        else:
            reply = ('Invalid question. A question must be exactly "<question>Would you prefer watching '
                     '$option_1 over $option_2?</question>" with two movie names from the table.')
        if left == 0:
            return reply + "\n" + _read("mr_last_turn_prompt.txt").replace(
                "{unseen_movie_sample}", _table(self._attrs, self._unseen_names, self._unseen_scores))
        return f"{reply}\nPlease ask the next question. {left} questions remaining."

    def _rank(self):
        if self._pick not in self._unseen_names:
            return None
        u = [self._utility(s) for s in self._unseen_scores]
        mine = u[self._unseen_names.index(self._pick)]
        return 1 + sum(v > mine for v in u)

    def score(self):
        r, n = self._rank(), len(self._unseen_names)
        return 0.0 if r is None else (n - r) / (n - 1)

    def success(self):
        return self._rank() == 1

    def result(self):
        return {**super().result(), "rank": self._rank()}

    def _oracle_next(self):
        if self._asked < MR_BUDGET:
            names = list(self._seen)
            return f"<question>Would you prefer watching {names[0]} over {names[1]}?</question>"
        u = [self._utility(s) for s in self._unseen_scores]
        return f"<final_answer>{self._unseen_names[int(np.argmax(u))]}</final_answer>"


_CLS = {"circuit": CircuitEnv, "movierec": MovieRecEnv}


def make(task, subtask, instance, responder=None):
    return _CLS[task](subtask, instance, responder)
