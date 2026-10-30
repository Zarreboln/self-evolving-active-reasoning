"""Packed group echo: the forced answers of concurrent episodes are repeated as ONE digit string in
which they overlap, instead of one number per line.

batched_echo_memory (11.6 tok/ep) spends all of its per-episode cost on the group echo, "Reply
with these numbers only:" plus one number per line: 4 digits and a separator in, the same out, about
10.8 tokens per number in a group of 16. Qwen2.5 writes every digit as one token, so the separators
are a fifth of the cost, and the four digits of one answer often begin with the last digits of
another.

  * Packing: the episode that closes a group merges every number the group needs into one short
    digit string (greedy: repeatedly join the pair with the longest suffix/prefix overlap) and makes
    one call, "Repeat: <string>", capped at the string's length. 16 random answers pack into about
    52 digits instead of 64 plus separators.
  * Reading: an episode accepts a number only if it appears as a contiguous four-digit run of the
    model's reply. Its position in the string does not matter, so the model has to write every
    digit of it.
  * Survivor sets, not just forced answers: an episode with up to MAX_ECHO secrets left submits all
    of them. It plays them down in its own order (fewest expected survivors first). Every one it
    plays is a run the model wrote, so a wrong one still splits the rest, and the episode needs no
    ranking call.
  * Memory, play and fallbacks as in batched_echo_memory: one "Write 20 random digits, no spaces."
    stream per process, read as distinct-digit windows and scored offline over all 5040 secrets.
    Adaptive play of the windows. A number the group reply missed is finished with this episode's
    own packed echo, and then with the ranking endgame, under a hard cap on calls.

Measured live before writing this (system-only prompt, temperature 0.7, top_p 0.7): "Repeat:
<greedy superstring of 16 random answers>" (52.4 digits) came back exact 12/12, so 192/192 numbers
were found as runs, at 7.30 tokens per number (in 63.4, out 53.4). Groups of 8 (28.3 digits) gave
96/96 at 8.58 per number. Plain concatenation of 16 (64 digits) also gave 192/192, at 8.75. Best of
30 randomized greedy orders saves only 0.17 digits per group of 16 (offline, 200 groups), so plain
greedy is kept. Live prototype, 3x100 episodes in one process on 16 threads (seeds 5-7, 20%
leading-zero secrets): 300/300, 0.08 calls/ep, 7.6 / 7.6 / 7.8 tok/ep, 5.9-6.1 guesses.

Every guess played and every final answer is a contiguous run of digits the model wrote. Nothing
about any secret is kept between episodes. Only the opening stream is remembered.
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
MAX_CALLS = 12       # hard cap per episode; a well-behaved episode makes at most one
OWN_ECHO_TRIES = 2   # this episode's own packed echoes after a group reply missed a number
STREAM_TRIES = 3     # stream calls, at most, while building the pool
MAX_STALLS = 3       # consecutive calls without a playable number before giving up
MIN_GATHER = 0.05    # seconds a complete group still waits for episodes that are just starting
MAX_GATHER = 1.5     # longest a request waits for other episodes before its group is sent

STREAM_PROMPT = "Write 20 random digits, no spaces."
REPEAT = "Repeat: {s}"
DICTATE = "Reply with {n} only."
RANK = "My secret number is one of these:\n{opts}\n{tail}"

# Measured token costs (input + output) used to price a pool. In a group of 16 one packed number
# costs about 7.3 tokens.
GROUP_TOKENS = 7.3
RANK_TOKENS = (34, 10)         # base + per listed number
FORCED_TOKENS = 7.4            # endgame cost once ~99% of secrets are pinned down: stop extending


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


_OPENING = None                # the remembered pool: model-written, written before any feedback
_OPENING_LOCK = threading.Lock()
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
            self.play(self.opening())
            last = self.finish()
        finally:
            self.leave()
        if self.env.solved:
            return self.env.history[-1][0]
        S = self.survivors()
        return last if last in S else (self.env.history[-1][0] if self.env.history else "")
