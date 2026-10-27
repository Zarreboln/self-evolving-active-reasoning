"""The paper's own player loop, as the seed to beat.

The whole gallery goes in once, then one turn at a time: the model writes a question, the
environment answers it or says Skip, and the reply is appended to the conversation. There is no
state outside the transcript, no screening of a draft before it is submitted, and no stopping rule
beyond the model deciding to guess.

Measured on the five difficulty tiers, 100 episodes each: it spends 19.1-19.3 of its 20 turns, and
roughly half of them come back Skip -- two thirds, under the paper's own detector. That is the
headroom the search is being pointed at.
"""
from __future__ import annotations

import re

from ..harness_base import AMIGOHarness

PROHIBITED = ("sleeve length, garment length, piece count, colour, prints or patterns, age group, "
              "size, shoes, necklace, hat, bag, background, or the human model wearing the dress")

SYSTEM = """You are playing Guess My Preferred Dress.

I will upload a gallery of dress photographs. One of them is my favourite; you cannot see which.
Your job is to identify it by asking questions, then name its position in the gallery.

Rules:
1. After I say "End of uploading", ask exactly ONE yes/no question per turn about ONE observable
   attribute of my favourite dress. Ask nothing before that signal.
2. The question must be answerable with Yes, No or Unsure.
3. You must NOT ask about: {prohibited}.
4. You must NOT ask about values of an attribute type you have already asked about. Do not cycle
   through necklines, or through closures, one value at a time.
5. You must NOT refer to any image by its number or position, and must NOT propose inspecting the
   images one at a time.
6. Ask about construction detail: neckline structure, closures, waist shaping, seam placement,
   drape, embellishment, fabric.
7. If I break a rule you will be told "Skip" and learn nothing, but the turn is still spent.
8. Do not guess until the answers leave exactly one possible dress. If the evidence is inconsistent
   or unclear, keep asking.
9. You have at most {budget} questions. When you are ready, reply with exactly this line and
   nothing else:
My guess of your favorite dress: #<number>

Reply with only the question, or only that final line."""

GUESS = re.compile(r"My guess of your favorite dress:\s*#\s*(\d+)", re.I)
FORCE = ("Out of turns. Reply with exactly this line and nothing else:\n"
         "My guess of your favorite dress: #<number>")


class Harness(AMIGOHarness):
    def solve(self) -> int:
        env = self.env
        content = [{"type": "text", "text": "Here is the gallery."}]
        content += env.gallery_blocks()
        content.append({"type": "text", "text": "End of uploading. Ask your first question."})
        messages = [{"role": "system",
                     "content": SYSTEM.format(prohibited=PROHIBITED, budget=env.budget)},
                    {"role": "user", "content": content}]

        while env.remaining > 0:
            out = self.llm(messages) or ""
            m = GUESS.search(out)
            if m:
                return self._clamp(int(m.group(1)) - 1)
            answer = env.ask(out.strip())
            messages = messages + [{"role": "assistant", "content": out},
                                   {"role": "user", "content": answer}]

        out = self.llm(messages + [{"role": "user", "content": FORCE}]) or ""
        m = GUESS.search(out)
        return self._clamp(int(m.group(1)) - 1) if m else 0

    def _clamp(self, i):
        return i if 0 <= i < self.env.n else 0
