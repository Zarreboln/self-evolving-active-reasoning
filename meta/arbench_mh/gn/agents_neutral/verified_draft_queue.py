"""The model writes every guess; code only checks it. Drafts are verified, queued and repaired.

Mechanism (axes A guess generation + C choosing what to play + D record shape). The zero-shot
loop asks for three numbers a turn, plays one, and never checks any of them against the record,
so most of what it plays could not be the secret. Here:

  * The record is shown as a compact table, followed by a digest of what it implies, computed by
    enumerating the 5040 possible secrets: how many remain, which digits are certainly in or out,
    which digits each position can still hold, and the full list once it is five or fewer.
  * The model drafts three guesses (AR-Bench's own format). Each draft is checked against every
    record line: would it, if it were the secret, have produced exactly the feedback observed?
  * Consistent drafts go into a queue. The first is played; after its feedback the rest are
    re-checked, and any still consistent are played without asking again -- one call can buy
    several guesses.
  * If no draft is consistent, one repair call shows, for each draft, the earlier guess it
    contradicts and the score that guess would have got. If the redrafts still all fail, the
    draft that contradicts the fewest record lines is played (never one already played).

Probe (Qwen2.5-7B, 8 random secrets drawn outside every split): 8/8 solved in 6.1 guesses,
8.4 calls and ~2.9k tokens per episode (2.7k in, 0.17k out). With the digest the model's drafts
were often consistent outright, and it sometimes wrote the answer among its three drafts.

The harness never writes a number itself: a round where the model produces nothing playable
spends no guess, and the number of rounds is bounded. The final answer is the solving guess, or,
if the budget runs out, the model's answer to the final prompt over the same table and digest.
"""
from __future__ import annotations

import re
from itertools import permutations

from ..harness_base import GNHarness

FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
SECRETS = ["".join(p) for p in permutations("0123456789", 4)]
POP = [bin(i).count("1") for i in range(1024)]


def mask(num):
    m = 0
    for c in num:
        m |= 1 << int(c)
    return m


MASK = {s: mask(s) for s in SECRETS}


def score(guess, secret):
    """(right place, misplaced) for `guess` if `secret` were the answer; AR-Bench's counting."""
    a = sum(g == s for g, s in zip(guess, secret))
    return a, POP[mask(guess) & MASK[secret]] - a


SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

ASK = ("Suggest 3 different guesses, best first. Each should agree with every record line: if it "
       "were the secret, every earlier guess would have scored exactly what it did.\n"
       "Answer in this format only:\nGuess: [num1], [num2], [num3]")

FIRST = ("No guesses yet. Suggest 3 different first guesses, each with four different digits.\n"
         "Answer in this format only:\nGuess: [num1], [num2], [num3]")

FINAL = ("The guesses are used up. Give your final answer: the number you believe is the secret. "
         "Reply with the number only.")

ORD = ["1st", "2nd", "3rd", "4th"]


class Harness(GNHarness):
    MAX_ROUNDS = 40          # model rounds, so a model that never writes a number cannot spin

    def solve(self) -> str:
        env = self.env
        self.alive = list(SECRETS)            # secrets consistent with every feedback so far
        queue, last = [], None
        for _ in range(self.MAX_ROUNDS):
            if env.solved or env.remaining <= 0:
                break
            queue = [d for d in queue if self.consistent(d) and not self.played(d)]
            if not queue:
                queue = self.draft()
            if not queue:
                continue                      # nothing playable this round; no guess spent
            last = queue.pop(0)
            a, b = env.guess(last)
            self.alive = [s for s in self.alive if score(last, s) == (a, b)]
        if env.solved:
            return env.history[-1][0]
        return self.final(last)

    # ---- state ------------------------------------------------------------------------------
    def played(self, num):
        return any(g == num for g, _, _ in self.env.history)

    def violations(self, num):
        """Record lines this number contradicts as a secret, with the score it would have given."""
        out = []
        if len(set(num)) < 4:
            return [(-1, None, None)]          # repeated digits: cannot be the secret at all
        for i, (g, a, b) in enumerate(self.env.history):
            got = score(g, num)
            if got != (a, b):
                out.append((i, got, (a, b)))
        return out

    def consistent(self, num):
        return not self.violations(num)

    # ---- prompts ----------------------------------------------------------------------------
    def table(self):
        rows = ["Record (guess: right place, misplaced):"]
        for i, (g, a, b) in enumerate(self.env.history, 1):
            rows.append("%d. %s: %d, %d" % (i, g, a, b))
        return "\n".join(rows)

    def digest(self):
        alive = self.alive
        n = len(alive)
        lines = ["What the record implies:", "- %d possible secrets remain." % n]
        if n == 0:
            return "\n".join(lines)
        sure_in = [d for d in "0123456789" if all(d in s for s in alive)]
        sure_out = [d for d in "0123456789" if not any(d in s for s in alive)]
        if sure_in:
            lines.append("- Surely in the secret: " + " ".join(sure_in))
        if sure_out:
            lines.append("- Surely not in the secret: " + " ".join(sure_out))
        pos = []
        for p in range(4):
            ds = sorted({s[p] for s in alive})
            pos.append("%s: %s" % (ORD[p], " ".join(ds)))
        lines.append("- Digits each position can still hold: " + " | ".join(pos))
        if n <= 5:
            lines.append("- The secret is one of: " + ", ".join(alive))
        return "\n".join(lines)

    def state_prompt(self, tail):
        if not self.env.history:
            return FIRST
        return "%s\n\n%s\n\n%s" % (self.table(), self.digest(), tail)

    # ---- drafting ---------------------------------------------------------------------------
    @staticmethod
    def parse(text):
        seen, out = set(), []
        for m in FOUR.findall(text or ""):
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def draft(self):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": self.state_prompt(ASK)}]
        out = self.llm(msgs, temperature=0.7, max_tokens=48)
        drafts = [d for d in self.parse(out) if not self.played(d)]
        good = [d for d in drafts if self.consistent(d)]
        if good or not self.env.history:
            # On the first turn every number with four different digits is consistent.
            return good or drafts[:1]
        # Repair: say exactly which record line each draft contradicts, and redraft once.
        notes = []
        for d in drafts[:3]:
            v = self.violations(d)
            i, got, want = v[0]
            if i < 0:
                notes.append("- %s repeats a digit, so it cannot be the secret." % d)
            else:
                g = self.env.history[i][0]
                notes.append("- %s cannot be the secret: guess %d (%s) would have scored %d, %d "
                             "against it, but it scored %d, %d." % (d, i + 1, g, got[0], got[1],
                                                                   want[0], want[1]))
        fix = ("None of those agree with the record:\n%s\n\n%s"
               % ("\n".join(notes) or "- no four-digit number found.", ASK))
        msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": fix}]
        out2 = self.llm(msgs, temperature=0.7, max_tokens=48)
        drafts2 = [d for d in self.parse(out2) if not self.played(d)]
        good = [d for d in drafts2 if self.consistent(d)]
        if good:
            return good
        pool = drafts2 + [d for d in drafts if d not in drafts2]
        if not pool:
            return []
        # Least-contradicting draft: fewest record lines broken, then smallest score error.
        def err(d):
            v = self.violations(d)
            if v and v[0][0] < 0:
                return (99, 99)
            return (len(v), sum(abs(g[0] - w[0]) + abs(g[1] - w[1]) for _, g, w in v))
        return [min(pool, key=err)]

    def final(self, last):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": self.state_prompt(FINAL)}]
        out = self.llm(msgs, temperature=0.0, max_tokens=16)
        cands = self.parse(out)
        for c in cands:
            if self.consistent(c):
                return c
        return cands[0] if cands else (last or "")
