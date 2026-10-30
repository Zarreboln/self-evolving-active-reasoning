"""Batch prompting across episodes: the forced answers of episodes running at the same time are
dictated together, in one call, instead of one call each.

verified_pool_memory costs 22 tokens per episode, and all of it is one dictation call:
"Reply with N only." is 8 template tokens, about 5 instruction tokens, 4 digits in and 5 out. Every
episode pays the template and the instruction again. benchmark.py runs a split's episodes in one
process on a thread pool, so up to 16 episodes reach that step together:

  * Coalesced dictation: an episode whose answer is forced (one secret left) submits it to a
    module-level coalescer and waits. The group closes when every episode currently registered is
    waiting, after MIN_GATHER, or after MAX_GATHER at the latest, so an episode running alone still
    finishes. The episode that closes it makes one call, "Reply with these numbers only:" plus
    one number per line, with its own self.llm. Measured live: 192/192 numbers repeated correctly
    in groups of 16, 96/96 in 8s and 48/48 in 4s. That is 10.8 tokens per number in a group of 16,
    13.3 in a group of 4. Each episode plays its own number only if the reply contains it.
    Otherwise it dictates alone, as before. Tokens are billed to the episode that made the call.
    The cost axis averages over episodes, so the mean is unaffected by who pays.
  * Memory, as in verified_pool_memory, but cheaper to build: one call, "Write 20 random digits, no
    spaces." (about 38 tokens). The harness reads every 4-digit run with distinct digits in it as a
    probe. A stream of about 20 digits gives about 16 such windows and pins down 99.7% of secrets,
    against 84.6% for the eight-number covering plan. The stream is scored offline over all 5040
    secrets, as before. Another stream is asked for, and its windows kept one by one, only if the
    pool leaves too many secrets unforced.
  * Play and endgame as in verified_pool_memory: adaptive play of the remembered windows (fewest
    expected survivors first, stop at one secret), then one ranking call if 2-5 are left.

Each episode stays correct on its own: no request waits longer than MAX_GATHER, and a failed or
missing group reply falls back to that episode's own call. Every guess played and every final
answer is a number the model wrote. A probe is a contiguous run of digits in the model's stream.
Nothing about any secret is kept between episodes.
"""
from __future__ import annotations

import random
import re
import threading
import time
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
RUN = re.compile(r"\d{4,}")
REPLY = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")    # a leading 0 is sometimes dropped in a reply
MENU = 5             # most numbers the model orders in one call
POOL = 30            # surviving numbers scored to fill a menu when more than MENU survive
MAX_CALLS = 12       # hard cap per episode; a well-behaved episode makes at most one
STREAM_TRIES = 3     # stream calls, at most, while building the pool
MAX_STALLS = 3       # consecutive calls without a playable number before giving up
MIN_GATHER = 0.05    # seconds a complete group still waits for episodes that are just starting
MAX_GATHER = 1.5     # longest a request waits for other episodes before its group is sent

STREAM_PROMPT = "Write 20 random digits, no spaces."
GROUP = "Reply with these numbers only:\n{nums}"
DICTATE = "Reply with {n} only."
RANK = "My secret number is one of these:\n{opts}\n{tail}"

# Measured token costs (input + output) of the endgame calls above, used to price a pool.
DICTATE_TOKENS = 22
RANK_TOKENS = (34, 10)         # base + per listed number
FORCED_TOKENS = 23             # endgame cost once ~99% of secrets are pinned down: stop extending


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


class _Request:
    __slots__ = ("payload", "done", "result", "since")

    def __init__(self, payload):
        self.payload, self.done, self.result, self.since = payload, False, None, time.monotonic()


class Coalescer:
    """Requests from episodes running at the same time, answered by one call.

    An episode joins on entering solve() and leaves once it will not submit again. A request waits
    until every member is waiting too, or until MAX_GATHER. The episode whose wait closes the group
    makes the call with its own `run`, and every member of the group gets its own result back.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._members = 0
        self._pending = []

    def join(self):
        with self._cv:
            self._members += 1

    def leave(self):
        with self._cv:
            self._members -= 1
            self._cv.notify_all()

    def submit(self, payload, run):
        req, group = _Request(payload), None
        with self._cv:
            self._pending.append(req)
            self._cv.notify_all()
            while not req.done:
                if req in self._pending:
                    waited = time.monotonic() - self._pending[0].since
                    full = len(self._pending) >= self._members
                    if (full and waited >= MIN_GATHER) or waited >= MAX_GATHER:
                        group, self._pending = self._pending, []
                        break
                    timeout = (MIN_GATHER if full else MAX_GATHER) - waited
                else:
                    timeout = MAX_GATHER               # another episode is making the call
                self._cv.wait(max(timeout, 0.005))
            if group is None:
                return req.result
        results = [None] * len(group)
        try:
            results = run([r.payload for r in group])
        except Exception:                              # every member then falls back on its own
            pass
        with self._cv:
            for r, res in zip(group, results):
                r.result, r.done = res, True
            self._cv.notify_all()
        return req.result


_OPENING = None                # the remembered pool: model-written, written before any feedback
_OPENING_LOCK = threading.Lock()
_ECHO = Coalescer()


class Harness(GNHarness):
    def __init__(self, env, solver):
        super().__init__(env, solver)
        self._S, self._seen = list(ALL), 0
        self._rng = random.Random()
        self._member = False

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

    def opening(self):
        """The remembered pool, built now if this is the first episode: the windows of one model-written
        digit stream, extended from further streams one window at a time while that lowers the cost."""
        global _OPENING
        with _OPENING_LOCK:
            if _OPENING is None:
                best, best_cost = [], float("inf")
                for t in range(STREAM_TRIES):
                    if best_cost <= FORCED_TOKENS:
                        break
                    for g in windows(self.ask(STREAM_PROMPT, max_tokens=26, temperature=0.7 + 0.15 * t)):
                        if g in best:
                            continue
                        cost = plan_cost(best + [g])
                        if cost < best_cost:
                            best, best_cost = best + [g], cost
                if not best:
                    return []              # nothing readable: this episode plays without an opening
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

    def leave(self):
        if self._member:
            self._member = False
            _ECHO.leave()

    def group_echo(self, numbers):
        """Run by whichever episode closes a group: one call repeating every number in it."""
        if len(numbers) == 1:
            prompt, max_tokens = DICTATE.format(n=numbers[0]), 12
        else:
            prompt, max_tokens = GROUP.format(nums="\n".join(numbers)), 5 * len(numbers) + 4
        got = set(replies(self.ask(prompt, max_tokens=max_tokens)))
        return [n in got for n in numbers]

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
        """One call of this episode's own: the forced answer dictated, or the model's order over the survivors."""
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

    def finish(self):
        """A forced answer goes to the group; anything else, or a number the group reply missed, is
        finished with this episode's own calls."""
        S = self.survivors()
        if not self.env.solved and self.env.remaining > 0 and len(S) == 1 and not self.played(S[0]):
            n = S[0]
            ok = _ECHO.submit(n, self.group_echo)
            self.leave()
            if ok:
                self.env.guess(n)
                return n
        self.leave()
        return self.endgame()

    def solve(self) -> str:
        _ECHO.join()
        self._member = True
        try:
            self.play(self.opening())
            last = self.finish()
        finally:
            self.leave()
        if self.env.solved:
            return self.env.history[-1][0]
        S = self.survivors()
        return last if last in S else (self.env.history[-1][0] if self.env.history else "")
