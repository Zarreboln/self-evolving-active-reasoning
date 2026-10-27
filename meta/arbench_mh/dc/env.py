"""Detective Cases environment: the suspects a harness may interrogate, and nothing else.

The suspects are played by the same frozen model, from in here, each with AR-Bench's own
`respond_template` over their private story. A harness never holds a suspect's system prompt, so
it cannot read the story, and the case's gold `key_question` list is never constructed at all --
which is the leak AR-Bench's own dc/tot has, where candidate questions are scored by how many
hidden key questions they hit.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.expanduser("~/code/AR-Bench"))

from arbench.reasoner.dc.dc_evaluator import (  # noqa: E402
    ANSWER_CHOICES, CHOICE_TO_INDEX, _format_initial_info, extract_answer_choice,
)
from arbench.reasoner.dc.prompt import (  # noqa: E402
    propose_template, question_propose_prompt, respond_template,
    select_murderer_template_searching,
)

BUDGET = 25


class BudgetExhausted(Exception):
    pass


class DCEnv:
    def __init__(self, item, answerer, budget=BUDGET):
        """`answerer(messages, temperature, max_tokens) -> str` plays the suspects."""
        self.budget = budget
        self._answerer = answerer
        self.init_info = _format_initial_info(item["initial_information"])
        self.suspects = [s["name"] for s in item["initial_information"]["suspect"]]
        self._label = item["label"]
        # Each suspect's private system prompt. Built here and never exposed.
        self._agents = {
            s["name"]: [{"role": "system",
                         "content": respond_template.format(name=s["name"], task=s["task"],
                                                            story=s["story"])}]
            for s in item["suspects"]}
        self.history = []            # [(name, question, reply)]
        self.record = ""
        self.system_prompt = propose_template.format(turn=budget, background=self.init_info)
        self._choice_str = ", ".join(f"{c}. {n}" for c, n in zip(ANSWER_CHOICES, self.suspects))

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def turn_prompt(self, t, suspect=None):
        return question_propose_prompt.format(turn=t, record=self.record,
                                              suspect=suspect or self.suspects[0])

    @property
    def final_prompt(self):
        return (propose_template.format(turn=self.budget, background=self.init_info)
                + select_murderer_template_searching.format(record=self.record,
                                                            choice=self._choice_str))

    def interrogate(self, name, question):
        """Ask one suspect one question. Returns their reply."""
        if self.remaining <= 0:
            raise BudgetExhausted()
        if name not in self._agents:
            raise KeyError("no such suspect: %r; the suspects are %s" % (name, self.suspects))
        self._agents[name].append({"role": "user", "content": question})
        reply = self._answerer(self._agents[name], temperature=0.7, max_tokens=None)
        self._agents[name].append({"role": "assistant", "content": reply})
        self.history.append((name, question, reply))
        self.record += f"Question for {name}: {question} Feedback: {reply}\n"
        return reply

    def parse_choice(self, text):
        """AR-Bench's own letter extraction. Returns the 0-based index, or None.

        The reply has to carry AR-Bench's literal `Answer: <letter>` line -- that is what
        `extract_answer_choice` matches, and `final_prompt` asks for it. None means the reply did
        not follow the format; AR-Bench's own map turns that into `""`, which reads as a valid
        value at a glance and is the easier mistake to make.
        """
        letter = extract_answer_choice(text or "").strip()
        return CHOICE_TO_INDEX.get(letter) if letter else None

    def score(self, prediction):
        return {"correctness": prediction == self._label, "round": len(self.history)}
