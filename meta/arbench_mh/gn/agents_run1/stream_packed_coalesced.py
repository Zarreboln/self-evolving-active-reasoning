"""Batching without memory, re-priced: each episode writes its own SHORT stream, and the whole surviving
set of every concurrent episode goes out in one packed digit string.

stream_coalesced_echo (50.2 tok/ep) keeps nothing between episodes: each writes "Write 20 random
digits, no spaces." (38 tokens) and only forced answers are coalesced, one number per line. Two
things change, and they only pay together:

  * Packed group echo (as in packed_group_echo): the closing episode merges every number the group
    needs into one short digit string and makes one "Repeat: <string>" call capped at its length.
    Each episode accepts a number only if it is a contiguous run of the reply. About 7.3 tokens per
    number in a group of 16, against about 10.8 one per line.
  * A shorter stream, because a survivor is now cheap: an episode sends ALL its survivors (up to
    MAX_ECHO) and plays them down itself (fewest expected survivors first). An extra survivor costs
    ~7 tokens, and a stream digit costs 1, so the cost model moves the stream from 20 to 10 digits.
    Priced offline on 16 live streams per length (each stream's windows over all 5040 secrets,
    stream tokens plus 7.3 per expected survivor): 20 digits 45.5, 16 42.8, 14 40.8, 12 39.3,
    10 38.5, 8 56.0 (the model writes "12345678" at 8). At 10 digits the model writes a shuffled 0-9
    (7 windows): 75% of secrets forced, 20% in pairs, 4% in triples.
  * The prompt is "10 random digits, no spaces:" capped at 10 tokens, so no end-of-turn token is
    billed: 64/64 shuffled 0-9 at temperature 0.7 and at 1.0, 26 tokens (priced: 35.5). "Write 10
    random digits, no spaces." gave the same windows at 28. Live prototype, 2x100 episodes in one
    process on 16 threads (seeds 7-8): 200/200, 1.08 calls/ep, 35.6 / 35.4 tok/ep. With the
    28-token prompt and the cap at 12: 100/100, 38.3.
  * A stream priced above REASK_ABOVE is asked for once more at a higher temperature. Nothing is
    remembered between episodes. Each episode's probes come from its own call, and its answers
    are runs of a reply that its request was part of.

Every guess played and every final answer is a contiguous run of digits the model wrote. A failed or
missing group reply falls back to this episode's own packed echo, then to the ranking endgame.
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
MENU = 5             # most numbers the model orders in one call (last-resort endgame)
POOL = 30            # surviving numbers scored to fill a menu when more than MENU survive
MAX_ECHO = 6         # most survivors an episode sends to be repeated; past this it plays the menu
MAX_DIGITS = 72      # longest string asked for in one call; a bigger group is split
MAX_CALLS = 12       # hard cap per episode; a well-behaved episode makes one, plus its group's
OWN_ECHO_TRIES = 2   # this episode's own packed echoes after a group reply missed a number
STREAM_TRIES = 2     # stream calls, at most
MAX_STALLS = 3       # consecutive calls without a playable number before giving up
MIN_GATHER = 0.05    # seconds a complete group still waits for episodes that are just starting
MAX_GATHER = 1.5     # longest a request waits for other episodes before its group is sent

STREAM_PROMPT = "10 random digits, no spaces:"
REPEAT = "Repeat: {s}"
DICTATE = "Reply with {n} only."
RANK = "My secret number is one of these:\n{opts}\n{tail}"

# Measured token costs (input + output) used to price a pool. In a group of 16 one packed number
# costs about 7.3 tokens.
GROUP_TOKENS = 7.3
RANK_TOKENS = (34, 10)         # base + per listed number
REASK_ABOVE = 35               # a second stream (26) plus a typical endgame (9.5): past this, ask again


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
    """Expected tokens to finish once k secrets are left: all of them packed into the group echo,
    or a menu round that leaves about a MENU-th of them."""
    if k <= MAX_ECHO:
        return GROUP_TOKENS * max(k, 1)
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


def superstring(numbers):
    """A short digit string with every number in it: greedy merging of the pair that overlaps most.
    A leading 0 is kept off the front where another order allows it, because a reply can drop it."""
    S = list(dict.fromkeys(numbers))
    while len(S) > 1:
        best = None
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i != j:
                    k = next((k for k in (3, 2, 1) if a[-k:] == b[:k]), 0)
                    if best is None or k > best[0]:
                        best = (k, i, j)
        k, i, j = best
        S = [s for t, s in enumerate(S) if t not in (i, j)] + [S[i] + S[j][k:]]
    s = S[0] if S else ""
    if s.startswith("0") and len(numbers) > 1:
        lead = [n for n in dict.fromkeys(numbers) if not n.startswith("0")]
        if lead:
            alt = lead[0] + superstring([n for n in numbers if n != lead[0]])
            s = alt if len(alt) <= len(s) + 1 else s
    return s


def echoed(text, asked):
    """Every four-digit run inside the digit runs of a reply. If the string asked for starts with 0,
    a run is also read with that 0 restored, as env.score reads a three-digit answer."""
    out = set()
    for run in re.findall(r"\d{3,}", text or ""):
        for r in ([run, "0" + run] if asked.startswith("0") else [run]):
            out |= {r[i:i + 4] for i in range(len(r) - 3)}
    return out


def echo_tokens(k):
    """Measured cost of 'Repeat: <superstring>' for k numbers: 8 template + 3 instruction tokens,
    then the string once in and once out, about 3.3 digits per number once k is past a few."""
    if k <= 0:
        return 0
    return 11 + 2 * (4 + 3.3 * (k - 1))

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


def chunks(numbers):
    """The numbers split, in order, into runs whose packed string fits in one call."""
    out, cur = [], []
    for n in numbers:
        if cur and len(superstring(cur + [n])) > MAX_DIGITS:
            out.append(cur)
            cur = []
        cur.append(n)
    return out + [cur] if cur else out


class Harness(GNHarness):
    def __init__(self, env, solver):
        super().__init__(env, solver)
        self._S, self._seen = list(ALL), 0
        self._rng = random.Random()
        self._member = False
        self._written = set()      # numbers this episode has seen the model write, still unplayed

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
        """This episode's probes: the windows of its own short digit stream, asked once more only if the
        endgame it leaves is priced above REASK_ABOVE."""
        best, best_cost = [], float("inf")
        for t in range(STREAM_TRIES):
            plan = windows(self.ask(STREAM_PROMPT, max_tokens=10, temperature=0.7 + 0.3 * t))
            cost = plan_cost(plan)
            if cost < best_cost:
                best, best_cost = plan, cost
            if best_cost <= REASK_ABOVE:
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

    def packed_echo(self, numbers):
        """Calls repeating `numbers` packed into as few digits as possible; the ones the model wrote."""
        got = set()
        for part in chunks(list(numbers)):
            s = superstring(part)
            got |= echoed(self.ask(REPEAT.format(s=s), max_tokens=len(s)), s) & set(part)
        return got

    def group_echo(self, payloads):
        """Run by whichever episode closes a group: every member's numbers in one packed string."""
        got = self.packed_echo(dict.fromkeys(n for p in payloads for n in p))
        return [set(p) & got for p in payloads]

    def play_down(self):
        """Play the model-written survivors, the one that leaves the fewest secrets expected first."""
        env = self.env
        while not env.solved and env.remaining > 0:
            S = self.survivors()
            ready = [g for g in S if g in self._written and not self.played(g)]
            if not ready:
                return
            env.guess(min(ready, key=lambda g: expected_survivors(g, S)))

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
        """Last resort, one call of this episode's own: the forced answer dictated, or the model's
        order over the survivors."""
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
        """Up to MAX_ECHO survivors go to the group as one payload. Whatever the group reply missed
        is repeated by this episode's own packed calls; a bigger set goes to the menu."""
        S = self.survivors()
        if not self.env.solved and self.env.remaining > 0 and 0 < len(S) <= MAX_ECHO:
            self._written |= _ECHO.submit(tuple(S), self.group_echo) or set()
            self.leave()
            self.play_down()
            for _ in range(OWN_ECHO_TRIES):
                S = self.survivors()
                if self.env.solved or self.env.remaining <= 0 or not S or len(S) > MAX_ECHO:
                    break
                self._written |= self.packed_echo([g for g in S if not self.played(g)])
                self.play_down()
        self.leave()
        if self.env.solved:
            return self.env.history[-1][0]
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
