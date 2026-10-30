"""The model writes an eight-number plan, the harness scores that plan against every possible secret
before playing it, and the endgame prompt depends on how many secrets the plan leaves.

static_cover_plan (141 tok/ep) spends ~97 tokens on its plan call and ~41 on a pick from a list
that, 83% of the time, holds one number. Three changes, each measured live before use:

  * Prompt architecture: every call is a single system message with no user turn. The chat template
    then costs 8 tokens instead of 13. The plan prompt is shorter (30 tokens in, against 56). In this
    role it gets the full cyclic covering plan 24/24 times. The same words as a user message got it
    0/24, and the old prompt in this role drifted to a worse plan, so the wording is not free.
  * Plan scoring: the plan is checked before any of it is played. The harness runs its own
    feedback function over all 5040 possible secrets and gets the share the plan would pin down
    alone, and the expected token cost of the endgame it leaves. Only if that cost exceeds the
    cost of asking again does the model get one more plan call (the older, longer prompt), and the
    better-scoring plan is played. No guess is spent and the oracle is not consulted to do this.
  * Endgame by residual size: one secret left (85% on the canonical plan) means the answer is
    forced, so the model is asked to say it: "Reply with N only.", 22 tokens, 64/64 correct
    including 24/24 with a leading 0. Two to five left: the model orders them all in one call
    (the wording that was 100% reliable), and the harness plays down the order, so it is one call
    however many it takes. More than five: the most informative five, the same way.
    A three-digit reply (a leading 0 dropped) is read with the 0 restored, as env.score reads it.

  * State: the set of secrets consistent with every (guess, feedback) pair, recomputed from
    env.history with the harness's own feedback function, so it knows only what the player was told.

Every guess played, and the final answer, is a number the model wrote.
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
REPLY = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")    # a leading 0 is sometimes dropped in a reply
PLAN = 8             # guesses the model writes before any feedback
MENU = 5             # most numbers the model orders in one call
POOL = 30            # surviving numbers scored to fill a menu when more than MENU survive
MAX_CALLS = 12       # hard cap per episode; a well-behaved episode uses two
MAX_STALLS = 3       # consecutive calls without a playable number before giving up

PLAN_PROMPT = f"Write {PLAN} four-digit numbers, no repeated digits, covering all digits 0-9. Numbers only."
REPLAN_PROMPT = (f"Write {PLAN} four-digit numbers, each with 4 different digits, that together use every "
                 "digit 0-9 and never put the same digit in the same position twice. "
                 f"Reply with the {PLAN} numbers only.")
DICTATE = "Reply with {n} only."
RANK = "My secret number is one of these:\n{opts}\n{tail}"

# Measured token costs (input + output) of the endgame calls above, used to price a plan.
DICTATE_TOKENS = 22
RANK_TOKENS = (34, 10)         # base + per listed number
REPLAN_TOKENS = 96             # the re-plan call itself
COVERED_TOKENS = 27            # endgame cost a full covering plan leaves (85% of secrets pinned down)


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


def plan_numbers(text):
    """Distinct four-digit numbers with four different digits, in the order written."""
    return [g for g in dict.fromkeys(FOUR.findall(text or "")) if len(set(g)) == 4][:PLAN]


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

    def ask(self, content, max_tokens, temperature=0.7, role="system"):
        """One call. By default the whole prompt is the system message and there is no user turn:
        the chat template then costs 8 tokens, against 13 with an empty system message."""
        msgs = ([{"role": "system", "content": content}] if role == "system" else
                [{"role": "system", "content": ""}, {"role": "user", "content": content}])
        return self.llm(msgs, temperature=temperature, max_tokens=max_tokens)

    def worth_playing(self, g):
        """A fresh number with four different digits whose feedback is not already certain."""
        if len(set(g)) < 4 or self.played(g):
            return False
        S = self.survivors()
        if g in S:
            return True                    # it could be the secret
        first = feedback(g, S[0])
        return any(feedback(g, s) != first for s in S[1:])

    def write_plan(self):
        """The model's plan, re-asked once (older wording, as a user turn) only when the endgame it
        would leave is priced above a fresh plan call plus that plan's endgame."""
        plan = plan_numbers(self.ask(PLAN_PROMPT, max_tokens=6 * PLAN + 8))
        cost = plan_cost(plan)
        if cost > REPLAN_TOKENS + COVERED_TOKENS:
            other = plan_numbers(self.ask(REPLAN_PROMPT, max_tokens=6 * PLAN + 8, role="user"))
            if plan_cost(other) < cost:
                plan = other
        return plan

    def play(self, plan):
        for g in plan:
            if self.env.solved or self.env.remaining <= 1:
                break
            if self.worth_playing(g):
                self.env.guess(g)

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

    def picks(self, temperature):
        """One call: the forced answer dictated, or the model's order over the survivors."""
        S = self.survivors()
        if len(S) == 1:
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
            for g in self.picks(0.7 + 0.3 * stalls):     # a re-ask at the same temperature repeats
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
        self.play(self.write_plan())
        last = self.endgame()
        if self.env.solved:
            return self.env.history[-1][0]
        S = self.survivors()
        return last if last in S else (self.env.history[-1][0] if self.env.history else "")
