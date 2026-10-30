"""One guess per turn from a possibility table, checked before it is played and sent back with the
exact reason when it cannot be the secret.

The zero-shot loop plays whatever the model writes; a 7B model mostly writes numbers that earlier
feedback has already ruled out, so most of its 25 guesses teach it nothing. Here the harness keeps
the set of secrets still consistent with every (guess, feedback) pair -- recomputed from
env.history with its own feedback function, so it knows only what the player was told -- and uses
it twice:

  * as the prompt: the record as a short list plus the set projected onto digits and positions
    ("Position 3 can only be: 2, 8"), in place of the prose transcript;
  * as a checker: a draft that is not in the set goes back to the model with every reason it fails
    -- each position the table forbids, or else the earlier guess whose feedback it contradicts and
    by how much -- and the model tries again, a little hotter each time so that it does not
    resubmit the same number.

After the retries the latest consistent draft is played, else the least-violating one that has not
been played before. A repeat is never played: it returns feedback the player already has. Every
guess played, and the final answer, is a number the model wrote.
"""
from __future__ import annotations

import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
GUESS_RE = re.compile(r"[Gg]uess\s*:?\s*\**\s*(\d{4})(?!\d)")
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
RETRIES = 2          # repair rounds per turn after the first draft
MAX_CALLS = 80       # hard cap per episode; a well-behaved episode uses ~12
MAX_STALLS = 3       # consecutive turns without a playable draft before giving up


def feedback(guess, secret):
    """(in place, misplaced) that `guess` would get if `secret` were the answer."""
    bulls = sum(g == s for g, s in zip(guess, secret))
    return bulls, len(set(guess) & set(secret)) - bulls


def parse(text):
    m = GUESS_RE.findall(text or "") or FOUR.findall(text or "")
    return m[-1] if m else None


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

    def why_not(self, g):
        """Every reason `g` cannot be the secret, in terms the prompt already used."""
        if len(set(g)) < 4:
            return f"{g} repeats a digit, but the secret has 4 different digits."
        if self.played(g):
            return f"{g} was already guessed and was not the answer."
        S = self.survivors()
        bad = []
        for p in range(4):
            allowed = sorted({s[p] for s in S})
            if g[p] not in allowed:
                bad.append(f"position {p + 1} cannot be {g[p]} (it can only be {', '.join(allowed)})")
        if bad:
            return f"in {g}, " + "; ".join(bad) + "."
        for i, (pg, a, b) in enumerate(self.env.history, 1):
            x, y = feedback(pg, g)
            if (x, y) != (a, b):
                return (f"if the secret were {g}, guess {i} ({pg}) would have scored {x} in place and "
                        f"{y} misplaced, but it actually scored {a} in place and {b} misplaced.")
        return f"{g} does not fit the feedback."

    def turn(self, stalls):
        """Draft, check, explain, redraft. Returns the drafts in the order written."""
        msgs = [{"role": "user", "content": self.state() + "\n\nPropose your next guess: a number with "
                 "4 different digits that fits every line above (it must be a number that could still "
                 "be the secret).\nReply with only: Guess: [number]"}]
        drafts = []
        for r in range(RETRIES + 1):
            if len(self._trace) >= MAX_CALLS:
                break
            out = self.llm(msgs, temperature=min(1.0, 0.7 + 0.15 * (r + stalls)), max_tokens=40)
            msgs.append({"role": "assistant", "content": out})
            g = parse(out)
            if g is None:
                msgs.append({"role": "user", "content": "Reply with only: Guess: [number]"})
                continue
            drafts.append(g)
            if g in self.survivors() and not self.played(g):
                break
            msgs.append({"role": "user", "content": f"That cannot be the secret: {self.why_not(g)} "
                         "Propose a different number that fits every line above.\n"
                         "Reply with only: Guess: [number]"})
        return drafts

    def solve(self) -> str:
        env, stalls = self.env, 0
        while env.remaining > 0 and not env.solved and len(self._trace) < MAX_CALLS:
            fresh = [d for d in self.turn(stalls) if not self.played(d)]
            if not fresh:
                stalls += 1
                if stalls >= MAX_STALLS:
                    break
                continue
            stalls = 0
            S = self.survivors()
            good = [d for d in fresh if d in S]
            env.guess(good[-1] if good else min(fresh, key=self.violations))
        if env.solved:
            return env.history[-1][0]
        return self.final_answer()

    def final_answer(self):
        S = self.survivors()
        msgs = [{"role": "user", "content": self.state() + "\n\nThe guessing is over. Give your final "
                 "answer: the number that fits every line above.\nReply with only: Guess: [number]"}]
        last = None
        for _ in range(2):
            if len(self._trace) >= MAX_CALLS + 2:
                break
            out = self.llm(msgs, max_tokens=24)
            g = parse(out)
            if g is not None:
                last = g
                if g in S:
                    return g
                msgs += [{"role": "assistant", "content": out},
                         {"role": "user", "content": f"That cannot be the secret: {self.why_not(g)} "
                          "Reply with only: Guess: [number]"}]
        return last or (self.env.history[-1][0] if self.env.history else "0000")
