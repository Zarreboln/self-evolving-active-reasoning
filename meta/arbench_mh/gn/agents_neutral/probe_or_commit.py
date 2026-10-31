"""The model decides when to stop probing and start committing; code may play non-secrets.

Mechanism (axis A, where guesses come from, with E, when to commit, and D, what the model is
shown). Every earlier harness played only possible secrets, so the model's only decision was
which of several equally likely secrets to name, and its answers there looked like a habit of
position. The real strategic decision in this game is a different one. Late in an episode a
possible secret can win now, while a number that cannot be the secret (a probe) can split the
remaining ones better. Which to play is a trade between a chance of winning this turn and a
smaller worst case afterwards, and code cannot settle it from information alone.

So belief stays in code (the secrets consistent with every feedback). While more than PROBE_CAP
remain, code plays the possible secret with the fewest expected survivors. Once few remain, code
finds the best commit (the possible secret whose miss leaves the fewest at worst) and the best
probe (any other number with distinct digits, by the same rule). When the probe's worst case is
strictly smaller, the model is asked one abstract question: N secrets remain; A) guess one,
right with chance 1 in N, else at most X remain; B) guess a number that cannot be right, after
which at most Y remain. The prompt names only those three numbers, so it is the same question in
every episode that meets the same (N, X, Y). It is asked at temperature 0 and cached for the
whole process. The opening is still the model's own free choice. A lone survivor, a commit that
no probe beats, or a budget of two or fewer guesses left is played without asking. An
unreadable reply or a failed call falls back to the commit.

Stub (100 random secrets, 16 threads; tokens estimated as one per digit, word or mark, which
puts pairwise_canon_tourney at 16.4 on the same stub against its measured 20.1): 100% solved,
5.36-5.42 guesses (max 7) whether the stub always commits, always probes or answers at random.
Variants compared on probe search thresholds: below 30 secrets, 28 distinct dilemmas, 29 calls,
32.0 tokens/ep; below 12, 10 dilemmas, 11.8; below 6 (kept), 6 dilemmas, 7 calls, 7.4. Large N
gives N, X and Y too many values to repeat, and the chance of winning now is only a real stake
when few remain. Garbage replies: 16 calls, every dilemma committed, 100%. An endpoint that
raises: 0 calls, 100%. No real-model probe: the cluster login was down this session.

Nothing is learned about any secret: the cache holds the model's replies to prompts, starts
empty in every process, and the first episode is played exactly as it would be without it.
"""
from __future__ import annotations

import re
import threading
from itertools import permutations

from ..harness_base import GNHarness

FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
LETTER = re.compile(r"\b([AB])\b")
SECRETS = ["".join(p) for p in permutations("0123456789", 4)]
POP = [bin(i).count("1") for i in range(1024)]
WIN = (4, 0)


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


def cells(g, pool):
    """How the secrets in pool split by the feedback g would get: {feedback: count}."""
    mg = MASK[g]
    out = {}
    for s in pool:
        a = (g[0] == s[0]) + (g[1] == s[1]) + (g[2] == s[2]) + (g[3] == s[3])
        k = (a, POP[mg & MASK[s]] - a)
        out[k] = out.get(k, 0) + 1
    return out


def spread(pool, n):
    """At most n items of pool, evenly spaced, so a large set is sampled without randomness."""
    if len(pool) <= n:
        return list(pool)
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def rank(alive, pool_cap=200, eval_cap=500):
    """Survivors ordered by expected number of survivors after playing them (fewest first)."""
    evals = spread(alive, eval_cap)
    scored = []
    for g in spread(alive, pool_cap):
        c = cells(g, evals)
        scored.append((sum(v * v for v in c.values()), g))
    scored.sort()
    return [g for _, g in scored]


def minimax(pool, live, can_win):
    """(worst case after a miss, guess) for the best guess in pool, ties by expected survivors."""
    best = None
    for g in pool:
        c = cells(g, live)
        if can_win:
            c.pop(WIN, None)
        worst = max(c.values(), default=0)
        if best is not None and worst > best[0]:
            continue
        key = (worst, sum(v * v for v in c.values()), g)
        if best is None or key < best:
            best = key
    return best[0], best[2]


SYSTEM = ("We are playing a number-guessing game. The secret is four different digits (it may "
          "start with 0). After each guess you are told how many digits are in the right place "
          "and how many are in the secret but in a different place.")

FIRST = "No guesses yet. Make your first guess: any four different digits. Reply with the number only."

DILEMMA = ("%d secrets are still possible.\n"
           "A) Guess one of them: it is right with chance 1 in %d. If not, at most %d stay possible.\n"
           "B) Guess a number that cannot be right. Afterwards at most %d stay possible.\n"
           "Which do you choose? Reply with the letter only.")

PROBE_CAP = 6       # probes are searched (all 5040 numbers) only once this few secrets remain


class Harness(GNHarness):
    # Shared by every instance in the process: prompt -> {"done": Event, "out": reply or None}.
    _memo = {}
    _memo_lock = threading.Lock()

    def solve(self) -> str:
        env = self.env
        alive = list(SECRETS)
        while not env.solved and env.remaining > 0:
            g = self.choose(alive)
            a, b = env.guess(g)
            alive = [s for s in alive if score(g, s) == (a, b)]
            if not alive:                     # cannot happen with a truthful oracle
                alive = list(SECRETS)
        if env.solved:
            return env.history[-1][0]
        return minimax(alive, alive, True)[1]

    def ask(self, messages):
        """The model at temperature 0, asked at most once per distinct prompt across episodes.

        Returns None if the call fails, so the caller falls back to code's choice."""
        key = tuple((m["role"], m["content"]) for m in messages)
        cls = type(self)
        with cls._memo_lock:
            slot = cls._memo.get(key)
            owner = slot is None
            if owner:
                slot = cls._memo[key] = {"done": threading.Event(), "out": None}
        if owner:
            try:
                slot["out"] = self.llm(messages, temperature=0.0, max_tokens=8)
            except Exception:
                slot["out"] = None
            finally:
                if slot["out"] is None:       # the call failed: let the next asker retry
                    with cls._memo_lock:
                        cls._memo.pop(key, None)
                slot["done"].set()
            return slot["out"]
        slot["done"].wait()
        if slot["out"] is not None:
            self._trace.append({"step": "llm_memo", "messages": [dict(m) for m in messages],
                                "response": slot["out"]})
        return slot["out"]

    def first(self):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
        out = self.ask(msgs)
        for n in FOUR.findall(out or ""):
            if len(set(n)) == 4:
                return n
        if out is not None:
            out = self.ask(msgs + [{"role": "assistant", "content": out},
                                   {"role": "user", "content": "Four different digits, number only."}])
            for n in FOUR.findall(out or ""):
                if len(set(n)) == 4:
                    return n
        return SECRETS[0]

    def dilemma(self, n, x, y):
        """The model's choice between committing (A) and probing (B); A if it gives none."""
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": DILEMMA % (n, n, x, y)}]
        out = self.ask(msgs)
        m = LETTER.search(out or "")
        if m is None and out is not None:
            out = self.ask(msgs + [{"role": "assistant", "content": out},
                                   {"role": "user", "content": "Reply with one letter, A or B."}])
            m = LETTER.search(out or "")
        return m.group(1) if m else "A"

    def choose(self, alive):
        if not self.env.history:
            return self.first()
        if len(alive) == 1:                   # nothing to choose between
            return alive[0]
        if len(alive) > PROBE_CAP:
            return rank(alive)[0]
        x, commit = minimax(alive, alive, True)
        if x <= 1 or self.env.remaining <= 2:  # no probe can beat a perfect split
            return commit
        live = set(alive)
        y, probe = minimax((g for g in SECRETS if g not in live), alive, False)
        if y >= x:
            return commit
        return probe if self.dilemma(len(alive), x, y) == "B" else commit
