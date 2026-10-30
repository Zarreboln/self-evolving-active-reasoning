"""Model drafts three guesses from a possibility table; the harness plays the best draft.

AR-Bench's tree-of-thought asks for three numbers and lets the model judge which to play (the
published version judged them against the hidden answer instead). Both halves are replaced here.

  * State: the set of four-digit secrets still consistent with every (guess, feedback) pair so far,
    recomputed from env.history with a feedback function of our own. It reads only what the
    player has been told.
  * Prompt: instead of the prose transcript, the record as a short list plus that set projected
    onto digits and positions ("Position 2 can only be: 1, 5, 7"). A 7B model can follow a
    per-position table; it cannot count its way through "exactly 2 of 0,1,2,3, none in place".
  * Selection: of the drafts that could still be the secret, play the one whose feedback is
    expected to leave the fewest survivors. If no draft qualifies, play the least-violating draft
    that has not been played yet rather than asking again: a guess costs no tokens and the budget
    is several times what is needed, so an imperfect guess is cheaper than a repair call.

Every guess played, and the final answer, is a number the model wrote.
"""
from __future__ import annotations

import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
GUESS_RE = re.compile(r"[Gg]uess\s*:?\s*\**\s*(\d{4})(?!\d)")
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
MAX_CALLS = 60       # hard cap per episode; a well-behaved episode uses ~6
MAX_STALLS = 3       # consecutive turns without a playable draft before giving up


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


def numbers(text):
    seen = []
    for x in FOUR.findall(text or ""):
        if x not in seen:
            seen.append(x)
    return seen


class Harness(GNHarness):
    def __init__(self, env, solver):
        super().__init__(env, solver)
        self._S, self._seen = list(ALL), 0

    def survivors(self):
        h = self.env.history
        while self._seen < len(h):
            g, a, b = h[self._seen]
            self._S = [s for s in self._S if feedback(g, s) == (a, b)]
            self._seen += 1
        return self._S

    def played(self, g):
        return any(g == pg for pg, _, _ in self.env.history)

    def violations(self, g):
        return (len(set(g)) < 4) + sum(feedback(pg, g) != (a, b) for pg, a, b in self.env.history)

    def state(self):
        S = self.survivors()
        rec = "\n".join(f"{i}. {g} -> {a} in place, {b} misplaced"
                        for i, (g, a, b) in enumerate(self.env.history, 1)) or "(none yet)"
        ins = [d for d in "0123456789" if all(d in s for s in S)]
        outs = [d for d in "0123456789" if not any(d in s for s in S)]
        lines = [f"Guesses so far:\n{rec}", "",
                 f"Exact deductions from all feedback ({len(S)} numbers are still possible):",
                 "- Digits certainly in the secret: " + (", ".join(ins) or "none known yet"),
                 "- Digits certainly NOT in the secret: " + (", ".join(outs) or "none known yet")]
        for p in range(4):
            lines.append(f"- Position {p + 1} can only be: " + ", ".join(sorted({s[p] for s in S})))
        if self.env.history:
            lines.append("- Already guessed (none was the answer, do not repeat): "
                         + ", ".join(g for g, _, _ in self.env.history))
        return self.env.system_prompt + "\n" + "\n".join(lines)

    def ask(self, instruction, max_tokens, temperature=0.7):
        if len(self._trace) >= MAX_CALLS:
            return ""
        return self.llm([{"role": "user", "content": self.state() + "\n\n" + instruction}],
                        temperature=temperature, max_tokens=max_tokens)

    def pick(self, drafts):
        S = self.survivors()
        fresh = [d for d in drafts if not self.played(d)]
        good = [d for d in fresh if d in S]
        if good:
            return good[0] if len(good) == 1 else min(good, key=lambda g: expected_survivors(g, S))
        return min(fresh, key=self.violations) if fresh else None

    def solve(self) -> str:
        env, stalls = self.env, 0
        while env.remaining > 0 and not env.solved and len(self._trace) < MAX_CALLS:
            out = self.ask("Propose 3 different numbers, each with 4 different digits, that each fit every "
                           "line above (each must be a number that could still be the secret).\n"
                           "Reply with only: Guess: [num1], [num2], [num3]", max_tokens=64,
                           temperature=0.7 + 0.3 * stalls)     # a stalled turn was all repeats
            g = self.pick(numbers(out))
            if g is None:
                stalls += 1
                if stalls >= MAX_STALLS:
                    break
                continue
            stalls = 0
            env.guess(g)
        if env.solved:
            return env.history[-1][0]
        return self.final_answer()

    def final_answer(self):
        S = self.survivors()
        out = self.ask("The guessing is over. Give your final answer: the number that fits every line above.\n"
                       "Reply with only: Guess: [number]", max_tokens=24)
        cands = GUESS_RE.findall(out or "") or numbers(out)
        for c in cands:
            if c in S:
                return c
        if cands:
            return cands[-1]
        return self.env.history[-1][0] if self.env.history else "0000"
