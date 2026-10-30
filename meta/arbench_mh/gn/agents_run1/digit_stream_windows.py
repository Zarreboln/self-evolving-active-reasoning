"""The model writes one stream of random digits, and the harness plays the 4-digit runs inside it as
probes. Nothing is kept or shared between episodes.

scored_plan_dictation is the cheapest design so far with nothing kept or shared between episodes
(94 tok/ep). Its plan call writes eight four-digit numbers: about 40 output tokens for 8 probes,
because Qwen spends a token on every digit and one on every separator. These eight pin down 84.6%
of secrets. Changes, each measured live first:

  * Guess generation from a digit stream: one call, "Write 20 random digits, no spaces." (18 in,
    about 20 out). Every four-digit run with four different digits inside the stream is a probe,
    so each extra digit the model writes adds one probe for one token. Over 16 live streams this
    gave 15.8 probes on average, pinning down 99.7% of secrets (offline, all 5040). Asked for
    several lines at once, the model counts cyclically instead (85%), so it is one stream per
    episode. The stream is scored offline over all 5040 secrets as an expected endgame cost.
    It is re-asked once, hotter, only if that cost exceeds the cost of asking again.
  * Adaptive play of the windows: fewest expected survivors first, stop at one secret, never play
    a window whose feedback is already certain. About 6 guesses instead of 7.4.
  * Capped dictation: "Repeat: N" with max_tokens=4 is 19 tokens (64/64 live, 24/24 with a leading
    0), against 22. The proven "Reply with N only." is the re-ask. 2-5 survivors are ranked in
    one call, as before.

Every guess played and the final answer are numbers the model wrote. A probe is a contiguous run
of digits in the model's stream.
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
RUN = re.compile(r"\d{4,}")
REPLY = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")    # a leading 0 is sometimes dropped in a reply
MENU = 5             # most numbers the model orders in one call
POOL = 30            # surviving numbers scored to fill a menu when more than MENU survive
MAX_CALLS = 12       # hard cap per episode; a well-behaved episode makes two
STREAM_TRIES = 2     # stream calls, at most
MAX_STALLS = 3       # consecutive calls without a playable number before giving up

STREAM_PROMPT = "Write 20 random digits, no spaces."
REPEAT = "Repeat: {n}"          # with max_tokens=4: the four digits and nothing else
DICTATE = "Reply with {n} only."
RANK = "My secret number is one of these:\n{opts}\n{tail}"

# Measured token costs (input + output) of the endgame calls above, used to price a pool.
DICTATE_TOKENS = 19
RANK_TOKENS = (34, 10)         # base + per listed number
STREAM_TOKENS = 38             # the stream call itself
FORCED_TOKENS = 20             # endgame cost of a stream that pins down ~99% of secrets


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


def endgame_tokens(k):
    """Expected tokens to finish once k secrets are left: a dictation, one ranking call, or a menu
    round that leaves about a MENU-th of them."""
    if k <= 1:
        return DICTATE_TOKENS
    if k <= MENU:
        return RANK_TOKENS[0] + RANK_TOKENS[1] * k
    return RANK_TOKENS[0] + RANK_TOKENS[1] * MENU + endgame_tokens(-(-k // MENU))


def plan_cost(plan):
    """Expected endgame tokens after playing `plan`, over every possible secret. The harness's own
    computation: no guess is spent and the oracle is not consulted."""
    if not plan:
        return float("inf")
    sizes = {}
    for s in ALL:
        key = tuple(feedback(g, s) for g in plan)
        sizes[key] = sizes.get(key, 0) + 1
    return sum(c * endgame_tokens(c) for c in sizes.values()) / len(ALL)


def windows(text):
    """Every four-digit run with four different digits inside the digit runs of the model's text,
    in the order written."""
    out = []
    for run in RUN.findall(text or ""):
        out += [run[i:i + 4] for i in range(len(run) - 3) if len(set(run[i:i + 4])) == 4]
    return list(dict.fromkeys(out))


def replies(text):
    """Numbers in a reply, in order; a three-digit one is read with its leading 0 restored."""
    return list(dict.fromkeys(g.zfill(4) for g in REPLY.findall(text or "")))


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
        """One call; the whole prompt is the system message and there is no user turn (8 template tokens)."""
        return self.llm([{"role": "system", "content": content}], temperature=temperature, max_tokens=max_tokens)

    def worth_playing(self, g):
        """A fresh number with four different digits whose feedback is not already certain."""
        if len(set(g)) < 4 or self.played(g):
            return False
        S = self.survivors()
        if g in S:
            return True                    # it could be the secret
        first = feedback(g, S[0])
        return any(feedback(g, s) != first for s in S[1:])

    def stream(self):
        """This episode's probes: the windows of one model-written digit stream, re-asked once if the
        endgame it leaves is priced above the cost of asking again."""
        best, best_cost = [], float("inf")
        for t in range(STREAM_TRIES):
            plan = windows(self.ask(STREAM_PROMPT, max_tokens=26, temperature=0.7 + 0.3 * t))
            cost = plan_cost(plan)
            if cost < best_cost:
                best, best_cost = plan, cost
            if best_cost <= STREAM_TOKENS + FORCED_TOKENS:
                break
        return best

    def play(self, pool):
        """Adaptive: the unplayed pool number that leaves the fewest secrets expected, until one is left."""
        env = self.env
        while not env.solved and env.remaining > 1 and len(self.survivors()) > 1:
            S = self.survivors()
            fresh = [g for g in pool if self.worth_playing(g)]
            if not fresh:
                break
            if len(S) == len(ALL):             # before any feedback every number splits alike
                env.guess(fresh[0])
                continue
            env.guess(min(fresh, key=lambda g: expected_survivors(g, S)))

    def menu(self):
        S = self.survivors()
        if len(S) <= MENU:
            out = list(S)
            self._rng.shuffle(out)
            return out
        pool = list(S) if len(S) <= POOL else self._rng.sample(S, POOL)
        pool.sort(key=lambda g: expected_survivors(g, S))
        top = pool[:MENU]
        self._rng.shuffle(top)
        return top

    def picks(self, temperature, first):
        """One call: the forced answer repeated (capped at its four digits; the longer wording on a
        re-ask), or the model's order over the survivors."""
        S = self.survivors()
        if len(S) == 1:
            if first:
                return replies(self.ask(REPEAT.format(n=S[0]), max_tokens=4, temperature=temperature))
            return replies(self.ask(DICTATE.format(n=S[0]), max_tokens=12, temperature=temperature))
        opts = self.menu()
        k = len(opts)
        tail = (f"Guess {k} of them, most likely first. Reply with the {k} numbers only." if k > 1
                else "Guess which one. Reply with the number only.")
        return replies(self.ask(RANK.format(opts="\n".join(opts), tail=tail),
                                max_tokens=6 * k + 8, temperature=temperature))

    def endgame(self):
        env, stalls, last = self.env, 0, None
        while env.remaining > 0 and not env.solved and len(self._trace) < MAX_CALLS:
            n = len(env.history)
            for g in self.picks(0.7 + 0.3 * stalls, stalls == 0):   # a re-ask at the same temperature repeats
                last = g
                if env.solved or env.remaining <= 0:
                    break
                if g in self.survivors() and not self.played(g):
                    env.guess(g)
            if len(env.history) > n:
                stalls = 0
                continue
            stalls += 1
            if stalls >= MAX_STALLS:
                break
        return last

    def solve(self) -> str:
        self.play(self.stream())
        last = self.endgame()
        if self.env.solved:
            return self.env.history[-1][0]
        S = self.survivors()
        return last if last in S else (self.env.history[-1][0] if self.env.history else "")
