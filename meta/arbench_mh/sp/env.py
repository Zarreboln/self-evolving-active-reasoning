"""Situation Puzzles environment: the referee a harness may reach, and nothing else.

AR-Bench's own prompts and scorer are imported rather than restated, and `ask` reproduces
`_run_traditional_evaluation` exactly: the referee's conversation is rebuilt from scratch every
turn, so it has no memory of earlier questions, and its reply is `.strip().strip(".")`. No
`max_tokens` is sent, because AR-Bench's `inference()` never sends one -- capping it at 512 was
enough to make every one of eight temperature-0 trajectories diverge from theirs. A harness
that does nothing but relay the model's output therefore reproduces AR-Bench's own numbers -- which
is the check that makes every later comparison mean something.

The bottom is private. The only way to learn anything about it is to spend a question.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.sp.prompt import get_answer_prompt  # noqa: E402
from arbench.reasoner.sp.sp_evaluator import METHOD_DICT, RESPONSE_TEMPLATE  # noqa: E402
from arbench.utils.utils_sp import f1_score  # noqa: E402

BUDGET = 25

# See gn/env.py. The referee is the only information channel here and it answers in prose, so
# AREW's (ICML 2026) own intervention applies unchanged: the turn is still spent, the question is
# still asked, and the answer says nothing. A harness reads the reply through env.history and
# env.record alike, and both carry the same string, so there is no second channel to leak through.
ABLATE = os.environ.get("ABLATE_FEEDBACK") == "1"


class BudgetExhausted(Exception):
    pass


class SPEnv:
    def __init__(self, item, answerer, budget=BUDGET, method="zero_shot"):
        """`answerer(messages, temperature, max_tokens) -> str` plays the referee.

        `method` picks which of AR-Bench's system prompts the episode starts from. The benchmark
        reads it off the candidate class (`Harness.METHOD`, default "zero_shot") so that a seed
        wrapping a published baseline gets that baseline's own prompt text, unedited.
        """
        self.budget = budget
        self._answerer = answerer
        self.surface = item["surface"]
        self._bottom = item["bottom"]
        self.history = []            # [(question, answer)]
        self.system_prompt = METHOD_DICT[method].format(turn=budget, question=self.surface)
        self.final_prompt = get_answer_prompt

    @property
    def remaining(self):
        return self.budget - len(self.history)

    @staticmethod
    def turn_prompt(t):
        return f"Turn {t}: please propose your next question."

    @property
    def record(self):
        return "".join("Q: %s\nA: %s\n" % qa for qa in self.history)

    def ask(self, question):
        """One question to the referee. Its conversation is rebuilt each turn, as AR-Bench does."""
        if self.remaining <= 0:
            raise BudgetExhausted()
        msgs = [{"role": "system",
                 "content": RESPONSE_TEMPLATE.format(question=self.surface, answer=self._bottom)},
                {"role": "user", "content": question}]
        reply = ("unknown" if ABLATE else
                 (self._answerer(msgs, temperature=0.7, max_tokens=None) or "").strip().strip("."))
        self.history.append((question, reply))
        return reply

    def score(self, prediction):
        p = prediction or ""
        return {"f1_score_char": f1_score(p, self._bottom),
                "f1_score_word": f1_score(p.split(), self._bottom.split()),
                "pred": p, "answer": self._bottom, "round": len(self.history)}
