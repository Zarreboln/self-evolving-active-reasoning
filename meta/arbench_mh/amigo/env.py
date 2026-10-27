"""The AMIGO environment: the hidden target, the oracle, the protocol detector, and the budget.

Everything the player is not allowed to see lives here. The harness gets the gallery, the rules,
and one method that spends a turn.

The oracle and both detector checks are the paper's own prompts, used verbatim
(`amigo_paper_prompts`, transcribed from arXiv 2603.28662 Appendix B). Our first implementation
reworded them into one request and, replayed over 120 recorded turns, called 53.3% of questions
violations where the paper's prompts call 65.0% -- a detector that lenient would hand a harness a
budget the paper does not give it, and Skip is the paper's headline metric.

Order, as Appendix B gives it: the deterministic P1 check, then Figure 19 (question type), then
Figure 20 (duplicate fine-grained attribute). Figure 19 short-circuits, since a turn already lost
cannot be lost twice.
"""
from __future__ import annotations

import base64
import os
import re
from .paper_prompts import FIG18_ORACLE, FIG19_TYPE, FIG20_DUPLICATE

BUDGET = 20                      # the paper's P11
ANSWER_CRITERIA = {"Yes": "The attribute is present in the dress shown.",
                   "No": "The attribute is absent from the dress shown.",
                   "Unsure": "The image does not show enough to decide."}
IMAGE_DIR = os.path.expanduser("~/amigo/images")

RULES = (
    "One of the dresses in the gallery is my favourite. Find it by asking yes/no questions.\n"
    "- One question per turn, about one observable construction attribute of the dress.\n"
    "- Do not ask about sleeve length, garment length, piece count, colour, prints or patterns,\n"
    "  age group, size, shoes, outerwear, necklace, hat, bag, or the background.\n"
    "- Do not refer to an image by its number or position, and do not ask to inspect images one\n"
    "  at a time.\n"
    "- Do not ask again about an attribute already settled, including with a different value.\n"
    "- A question that breaks a rule is answered Skip and the turn is gone.\n"
    "- You have %d turns. Name your answer as: GUESS: <n>, the image's number in the gallery."
) % BUDGET

WH = re.compile(r"^\s*(what|which|how|why|where|who|when|describe|list|name)\b", re.I)


class BudgetExhausted(Exception):
    pass


def _b64(image_id):
    for ext in (".jpg", ".jpeg", ".png", ""):
        p = os.path.join(IMAGE_DIR, image_id + ext)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    raise FileNotFoundError(image_id)


def p1_violation(text):
    """The paper's deterministic P1 and nothing more: empty turns, non-questions, multi-question
    turns, and questions that are not Yes/No-answerable. Repeats are Figure 20's job here.

    "Not Yes/No-answerable" is the one case Appendix B names without defining; we read it as an
    opening interrogative that demands an open answer.
    """
    q = (text or "").strip()
    if not q:
        return "P1 empty turn"
    if "?" not in q:
        return "P1 not a question"
    if q.count("?") > 1:
        return "P1 more than one question"
    if WH.match(q):
        return "P1 not Yes/No-answerable"
    return None


class AMIGOEnv:
    def __init__(self, example, jev, budget=BUDGET):
        self.gallery = list(example["gallery"])
        self.n = len(self.gallery)
        self.budget = budget
        self.rules = RULES
        self.history = []              # [(question, answer)], Skips included
        self.answered = []             # [(question, answer)], Skips left out
        self._target = example["target_index"]
        self._target_b64 = _b64(self.gallery[self._target])
        self._jev = jev
        self._blocks = None

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def image_block(self, i):
        return {"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + _b64(self.gallery[i])}}

    def gallery_blocks(self):
        if self._blocks is None:       # decoded once per episode, never across episodes
            self._blocks = [self.image_block(i) for i in range(self.n)]
        return list(self._blocks)

    def ask(self, question):
        """One turn. Returns Yes / No / Unsure, or Skip when a protocol rule is broken."""
        if self.remaining <= 0:
            raise BudgetExhausted()
        why = p1_violation(question)
        if why is None:
            why = self._detect(question)
        if why is not None:
            self.history.append((question, "Skip"))
            return "Skip"
        answer = self._oracle(question)
        self.history.append((question, answer))
        self.answered.append((question, answer))
        return answer

    def _detect(self, question):
        a = self._jev.ask({"state": {"question": question},
                           "questions": {"violation": {"type": "noul",
                                                       "instructions": FIG19_TYPE}}})
        if a["violation"]["noul"] > 0.5:
            return "type violation"
        numbered = "\n".join("%d) %s" % (i + 1, q) for i, (q, _) in enumerate(self.answered))
        b = self._jev.ask({"state": {"new_question": question,
                                     "history_questions": numbered or "(none yet)"},
                           "questions": {"duplicate": {"type": "noul",
                                                       "instructions": FIG20_DUPLICATE}}})
        return "duplicate attribute" if b["duplicate"]["noul"] > 0.5 else None

    def _oracle(self, question):
        """Figure 18's prompt, then the question. The image travels as an asset, which is the
        shape the API takes; a backtick in the question would be read as a reference, so it is
        flattened first."""
        ref = "target-image-reference"
        a = self._jev.ask({
            "state": {"dress": ref},
            "assets": {ref: {"type": "image", "format": "jpg", "data_base64": self._target_b64}},
            "questions": {"answer": {
                "type": "choice", "criteria": ANSWER_CRITERIA,
                "instructions": FIG18_ORACLE + "\n\n" + question.replace("`", "'") +
                                "\n\nAnswer about the dress in `dress`."}}})
        return a["answer"]["choice"]

    def score(self, prediction):
        return prediction == self._target
