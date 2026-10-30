"""A pool of probe numbers, written by the model once and remembered across episodes. The harness
keeps a written number only where its own scoring shows it helps, and plays the pool adaptively.

opening_memory remembers the model's eight-number covering plan, which pins down the secret alone
for 85% of secrets. The rest are left in pairs and triples that cyclic numbers cannot split, and
each of those episodes pays for a ranking call (54-64 tokens) instead of a dictation (22). This
harness grows the remembered opening until nearly every answer is forced:

  * Memory: one module-level pool, guarded by a lock, built during the first episode. It is the
    model's covering plan plus numbers from one more call, "4 numbers with no pattern" (45 tokens,
    once). Only model-written numbers, chosen before any feedback; nothing about any secret is kept.
  * Verified acceptance: each extra number is scored with the harness's own feedback function over
    all 5040 possible secrets. It joins the pool only if it lowers the expected endgame cost the
    pool leaves. The extension is re-asked once if the pool still leaves more than a few percent of
    secrets unforced. In prototyping, four such numbers took the canonical plan from 84.6% of
    secrets pinned down to 98.9%. A bad first plan is re-asked as in opening_memory.
  * Adaptive play: each turn the harness plays the unplayed pool number with the fewest expected
    survivors given the feedback so far, and stops once one secret is left. Numbers whose feedback
    is already certain are never played. The final split is the same as playing the whole pool, in
    fewer guesses.
  * Endgame, per episode, as in scored_plan_dictation: one secret left means the model is asked to
    say it (22 tokens, 64/64 correct including leading zeros). Otherwise it orders the survivors in
    one call. Single system message, no user turn.

Every guess played, and the final answer, is a number the model wrote. The pool's numbers were
written in earlier calls, possibly during another episode.
"""
from __future__ import annotations

import random
import re
import threading
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
REPLY = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")    # a leading 0 is sometimes dropped in a reply
PLAN = 8             # guesses the model writes before any feedback
MENU = 5             # most numbers the model orders in one call
POOL = 30            # surviving numbers scored to fill a menu when more than MENU survive
MAX_CALLS = 12       # hard cap per episode; a well-behaved episode uses one
PLAN_TRIES = 3       # plan calls, at most, before the best one is remembered
EXTEND = 4           # numbers asked for per extension call
EXTEND_TRIES = 2     # extension calls, at most
MAX_STALLS = 3       # consecutive calls without a playable number before giving up

PLAN_PROMPT = f"Write {PLAN} four-digit numbers, no repeated digits, covering all digits 0-9. Numbers only."
REPLAN_PROMPT = (f"Write {PLAN} four-digit numbers, each with 4 different digits, that together use every "
                 "digit 0-9 and never put the same digit in the same position twice. "
                 f"Reply with the {PLAN} numbers only.")
EXTEND_PROMPT = f"Write {EXTEND} four-digit numbers with no repeated digits and no pattern. Numbers only."
DICTATE = "Reply with {n} only."
RANK = "My secret number is one of these:\n{opts}\n{tail}"

# Measured token costs (input + output) of the endgame calls above, used to price a plan.
DICTATE_TOKENS = 22
RANK_TOKENS = (34, 10)         # base + per listed number
REPLAN_TOKENS = 96             # the re-plan call itself
COVERED_TOKENS = 27            # endgame cost a full covering plan leaves (85% of secrets pinned down)
FORCED_TOKENS = 23             # endgame cost once ~97% of secrets are pinned down: stop extending


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


_OPENING = None                # the remembered pool: model-written, written before any feedback
_OPENING_LOCK = threading.Lock()


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

    def opening(self):
        """The remembered pool, built now if this is the first episode: the best-scoring of up to
        PLAN_TRIES plans, then extension numbers kept one by one where they lower the endgame cost."""
        global _OPENING
        with _OPENING_LOCK:
            if _OPENING is None:
                best, best_cost = [], float("inf")
                for t in range(PLAN_TRIES):
                    prompt, role = (PLAN_PROMPT, "system") if t % 2 == 0 else (REPLAN_PROMPT, "user")
                    plan = plan_numbers(self.ask(prompt, max_tokens=6 * PLAN + 8,
                                                 temperature=0.7 + 0.15 * t, role=role))
                    cost = plan_cost(plan)
                    if cost < best_cost:
                        best, best_cost = plan, cost
                    if best_cost <= REPLAN_TOKENS + COVERED_TOKENS:
                        break
                if not best:
                    return []              # nothing readable: this episode plays without an opening
                for t in range(EXTEND_TRIES):
                    if best_cost <= FORCED_TOKENS:
                        break
                    out = self.ask(EXTEND_PROMPT, max_tokens=6 * EXTEND + 8, temperature=0.7 + 0.3 * t)
                    for g in plan_numbers(out)[:EXTEND]:
                        if g in best:
                            continue
                        cost = plan_cost(best + [g])
                        if cost < best_cost:
                            best, best_cost = best + [g], cost
                _OPENING = best
            return list(_OPENING)

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
        self.play(self.opening())
        last = self.endgame()
        if self.env.solved:
            return self.env.history[-1][0]
        S = self.survivors()
        return last if last in S else (self.env.history[-1][0] if self.env.history else "")
