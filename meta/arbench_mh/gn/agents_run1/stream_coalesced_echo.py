"""Batch prompting with no memory: each episode writes its own probes in its own call, as in
digit_stream_windows. Only the final dictation is shared with the other episodes running at the time.

This separates the two kinds of cross-episode saving. batched_echo_memory keeps a pool of probes
across episodes and also batches the dictation. This harness keeps nothing: no state survives an
episode, and every probe an episode plays comes from its own stream call. The only coupling is
that concurrent episodes send their forced answers in one call. That is batch prompting proper,
with one prompt carrying several instances and each instance getting its own output line. Pooling
outputs is what memory adds on top. It is the cheapest design if model outputs must not be reused
across episodes but call batching is allowed.

  * Probes: one call, "Write 20 random digits, no spaces." Every four-digit run with four different
    digits inside the stream is played adaptively (fewest expected survivors first, stop at one
    secret). It is scored offline over all 5040 secrets and re-asked once only if the endgame it
    leaves costs more than asking again. Live: 80/80, every secret pinned down, 57 tok/ep with a
    capped solo dictation.
  * Coalesced dictation: a forced answer is submitted to a module-level coalescer. The group closes
    when every registered episode is waiting, or after MAX_GATHER at the latest, so a lone episode
    still finishes. The episode that closes it makes one call, "Reply with these numbers only:" with
    one number per line, using its own self.llm (192/192 numbers repeated correctly in groups of
    16, 10.8 tokens per number). An episode plays its number only if the group reply contains it.
    Otherwise it dictates alone. 2-5 survivors are ranked in the episode's own call.

Every guess played and the final answer are numbers the model wrote.
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
MAX_CALLS = 12       # hard cap per episode; a well-behaved episode makes one, plus its group's
STREAM_TRIES = 2     # stream calls, at most
MAX_STALLS = 3       # consecutive calls without a playable number before giving up
MIN_GATHER = 0.05    # seconds a complete group still waits for episodes that are just starting
MAX_GATHER = 1.5     # longest a request waits for other episodes before its group is sent

STREAM_PROMPT = "Write 20 random digits, no spaces."
GROUP = "Reply with these numbers only:\n{nums}"
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

    def leave(self):
        if self._member:
            self._member = False
            _ECHO.leave()

    def group_echo(self, numbers):
        """Run by whichever episode closes a group: one call repeating every number in it."""
        if len(numbers) == 1:
            prompt, max_tokens = REPEAT.format(n=numbers[0]), 4
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
            self.play(self.stream())
            last = self.finish()
        finally:
            self.leave()
        if self.env.solved:
            return self.env.history[-1][0]
        S = self.survivors()
        return last if last in S else (self.env.history[-1][0] if self.env.history else "")
