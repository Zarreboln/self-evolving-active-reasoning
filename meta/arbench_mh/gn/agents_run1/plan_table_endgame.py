"""The model writes eight guesses in one call before any feedback; after the harness plays them, the
model writes the answer itself from a per-position table of the digits still possible. It is not
handed a list of candidates to copy.

static_cover_plan ends by listing the surviving secrets and asking the model to pick one. With one
survivor, which is the usual case, that pick is a copy. This harness keeps the plan and replaces
the list with the representation the 7B model was measured to use best in iteration 1: the surviving
set projected onto the four positions. Each position shows the digits it can still hold. The model
has to put the number together from that table:

  * State: the set of secrets consistent with every (guess, feedback) pair, recomputed from
    env.history with the harness's own feedback function, so it knows only what the player was told.
  * Plan play: as in static_cover_plan. One call returns eight numbers, played in the model's
    order, and any that would teach nothing are skipped.
  * Endgame: the table under an empty system message, then "write the secret". Each row leads
    with the digit of one survivor, and 0 is called out as an ordinary digit. In prototyping, a
    plain sorted table solved 34/40: the model moved a leading 0 to the end, and with two options
    per row it wrote repeated digits again and again at a fixed temperature. A draft that could
    be the secret is played. A draft that cannot be, but whose feedback is not yet certain, is
    also played instead of being repaired, because a guess costs nothing and a repair call does.
    Only a draft that would teach nothing earns another call, at a higher temperature.

Every guess played, and the final answer, is a number the model wrote.
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
PLAN = 8             # guesses the model writes before any feedback
ORDINAL = ("1st", "2nd", "3rd", "4th")
ROW = "{o}: {ds}"
ASK = ("Digits of my secret code, position by position (0 is a digit like any other):\n{table}\n"
       "Write the code, digits in this order, 4 characters. Reply with the code only.")
MAX_CALLS = 40       # hard cap per episode; a well-behaved episode uses about two
MAX_STALLS = 3       # consecutive turns without a playable pick before giving up

PLAN_PROMPT = (f"Write {PLAN} four-digit numbers, each with 4 different digits, that together use every "
               "digit 0-9 and never put the same digit in the same position twice. "
               f"Reply with the {PLAN} numbers only.")


def feedback(guess, secret):
    """(in place, misplaced) that `guess` would get if `secret` were the answer."""
    bulls = sum(g == s for g, s in zip(guess, secret))
    return bulls, len(set(guess) & set(secret)) - bulls


def numbers(text):
    """Distinct four-digit numbers, in the order written."""
    return list(dict.fromkeys(FOUR.findall(text or "")))


class Harness(GNHarness):
    def __init__(self, env, solver):
        super().__init__(env, solver)
        self._S, self._seen = list(ALL), 0
        self._rng = random.Random()

    def survivors(self):
        h = self.env.history
        while self._seen < len(h):
            g, a, b = h[self._seen]
            self._S = [s for s in self._S if feedback(g, s) == (a, b)]
            self._seen += 1
        return self._S

    def played(self, g):
        return any(g == pg for pg, _, _ in self.env.history)

    def ask(self, content, max_tokens, temperature=0.7):
        """One call on a bare user message. The empty system message replaces the chat template's
        default one, which costs 16 tokens a call and says nothing about the game."""
        return self.llm([{"role": "system", "content": ""}, {"role": "user", "content": content}],
                        temperature=temperature, max_tokens=max_tokens)

    def worth_playing(self, g):
        """A fresh number with four different digits whose feedback is not already certain."""
        if len(set(g)) < 4 or self.played(g):
            return False
        S = self.survivors()
        if g in S:
            return True                    # it could be the secret
        first = feedback(g, S[0])
        return any(feedback(g, s) != first for s in S[1:])

    def play_plan(self):
        out = self.ask(PLAN_PROMPT, max_tokens=6 * PLAN + 8)
        for g in numbers(out)[:PLAN]:
            if self.env.solved or self.env.remaining <= 1:
                break
            if self.worth_playing(g):
                self.env.guess(g)

    def table(self):
        """The surviving secrets projected onto positions: the digits each position can still hold.
        Each row leads with the digit of one survivor (a random one). Read row by row as written,
        the first digits spell a possible secret. The model, left to itself, takes the first
        option per row and writes repeated digits about as often as not."""
        S = self.survivors()
        lead = self._rng.choice(S)
        rows = []
        for i in range(4):
            ds = [lead[i]] + sorted({x[i] for x in S} - {lead[i]})
            rows.append(ROW.format(n=i + 1, o=ORDINAL[i], ds=" or ".join(ds)))
        return "\n".join(rows)

    def draft(self, temperature=0.7):
        """One call. The model's number if it could be the secret, else if it would still teach
        something, else None."""
        S = self.survivors()
        out = self.ask(ASK.format(table=self.table()), max_tokens=16, temperature=temperature)
        drafts = [g for g in numbers(out) if not self.played(g)]
        return next((g for g in drafts if g in S), next((g for g in drafts if self.worth_playing(g)), None))

    def solve(self) -> str:
        env = self.env
        self.play_plan()
        stalls = 0
        while env.remaining > 0 and not env.solved and len(self._trace) < MAX_CALLS:
            g = self.draft(0.7 + 0.3 * stalls)       # a re-ask at the same temperature repeats itself
            if g is None or not self.worth_playing(g):
                stalls += 1
                if stalls >= MAX_STALLS:
                    break
                continue
            stalls = 0
            env.guess(g)
        if env.solved:
            return env.history[-1][0]
        return (self.draft() if len(self._trace) < MAX_CALLS else None) or \
            (env.history[-1][0] if env.history else "")
