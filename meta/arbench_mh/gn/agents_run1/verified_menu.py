"""The harness drafts, the model chooses: each turn the model picks its guess from a short menu of
numbers that are all still possible.

The usual division of labour is inverted. Drafting a number that fits every earlier piece of
feedback is the step a 7B model fails at (asked for three, it gets one right about a quarter of the
time); choosing between numbers it is handed is not. So the harness keeps the set of secrets still
consistent with every (guess, feedback) pair -- recomputed from env.history with its own feedback
function, so it knows only what the player was told -- and each turn offers a handful of them:

  * which numbers: a sample of the surviving set, ranked by the expected number of survivors each
    would leave, best few kept -- so whichever the model picks, the guess is both possible and
    informative;
  * how they are shown: shuffled, so the pick is the model's rather than the ranking's;
  * what the model sees: the record as a short list and the menu; no transcript prose.

A pick that is off the menu is still played if it is a number that could be the secret and has not
been played; otherwise the model is asked once more. Every guess played, and the final answer, is a
number the model wrote.
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
GUESS_RE = re.compile(r"[Gg]uess\s*:?\s*\**\s*(\d{4})(?!\d)")
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
MENU = 5             # numbers offered per turn
POOL = 30            # surviving numbers scored to fill the menu
MAX_CALLS = 60       # hard cap per episode; a well-behaved episode uses ~6
MAX_STALLS = 3       # consecutive turns without a playable pick before giving up


def feedback(guess, secret):
    """(in place, misplaced) that `guess` would get if `secret` were the answer."""
    bulls = sum(g == s for g, s in zip(guess, secret))
    return bulls, len(set(guess) & set(secret)) - bulls


def expected_survivors(guess, S):
    counts = {}
    for s in S:
        k = feedback(guess, s)
        counts[k] = counts.get(k, 0) + 1
    return sum(c * c for c in counts.values()) / len(S)


def parse(text):
    m = GUESS_RE.findall(text or "") or FOUR.findall(text or "")
    return m[-1] if m else None


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

    def menu(self):
        S = self.survivors()
        if len(S) <= MENU:
            return list(S)
        pool = list(S) if len(S) <= POOL else self._rng.sample(S, POOL)
        if len(S) < len(ALL):                  # before any feedback every number splits alike
            pool.sort(key=lambda g: expected_survivors(g, S))
        top = pool[:MENU]
        self._rng.shuffle(top)
        return top

    def prompt(self, options, final=False):
        rec = "\n".join(f"{i}. {g} -> {a} in place, {b} misplaced"
                        for i, (g, a, b) in enumerate(self.env.history, 1)) or "(none yet)"
        n = len(self.survivors())
        head = (f"{self.env.system_prompt}\nGuesses so far:\n{rec}\n\n"
                f"These numbers all fit every piece of feedback so far ({n} such numbers remain in total):\n"
                + "\n".join(options))
        if final:
            return head + "\n\nThe guessing is over. Choose your final answer from the list.\nReply with only: Guess: [number]"
        return head + "\n\nChoose the one you think is most likely to be the secret.\nReply with only: Guess: [number]"

    def choose(self, options, final=False):
        """The model's pick, asked twice at most. Returns a playable pick, else its last answer."""
        S = self.survivors()
        msgs = [{"role": "user", "content": self.prompt(options, final)}]
        last = None
        for attempt in range(2):
            if len(self._trace) >= MAX_CALLS:
                break
            out = self.llm(msgs, temperature=0.7 + 0.3 * attempt, max_tokens=24)
            g = parse(out)
            if g is not None:
                last = g
                if g in S and not self.played(g):
                    return g, True
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": "Pick one of the listed numbers.\nReply with only: Guess: [number]"}]
        return last, False

    def solve(self) -> str:
        env, stalls = self.env, 0
        while env.remaining > 0 and not env.solved and len(self._trace) < MAX_CALLS:
            g, _ = self.choose(self.menu())
            if g is None or self.played(g):
                stalls += 1
                if stalls >= MAX_STALLS:
                    break
                continue
            stalls = 0
            env.guess(g)
        if env.solved:
            return env.history[-1][0]
        S = self.survivors()
        g, _ = self.choose(list(S) if len(S) <= MENU else self.menu(), final=True)
        return g or (env.history[-1][0] if env.history else "0000")
