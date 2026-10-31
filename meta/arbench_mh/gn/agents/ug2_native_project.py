"""AR-Bench's own loop, with every proposal projected onto the numbers the feedback allows.

The model sees exactly what the zero-shot loop shows it -- AR-Bench's rule text and turn prompt
over the record so far -- and answers in AR-Bench's format, three proposals per turn. The harness
keeps the exact belief state (P(10,4) filtered by each (guess, in place, misplaced) line, with the
oracle's counting rule applied to hypotheses, never to the target) and plays:

  * the first proposal that agrees with every feedback line, if there is one;
  * otherwise the surviving number NEAREST to the proposals: most digits in the same place, then
    most digits shared, earlier proposals first. Among equally near survivors it prefers the one
    that leaves the fewest expected survivors.

No second call, no retry: where iteration 1's gate re-asked the model (an extra full-prompt call
per rejection), this keeps the model's aim and repairs it in place.

Hypothesis (iteration 2): the zero-shot loop fails because its guesses contradict earlier
feedback, not because its aim is bad. Projection alone, with the prompt and the model's output
untouched, solves nearly every episode at the zero-shot loop's cost per turn (about one native
prompt per guess). Falsified if accuracy stays far below a menu harness's, or if mean guess
rounds are well above a consistent-guess strategy's (the model's aim then drags the search).

Prototyped against stub models only (the cluster login had expired; no real-model probe was
possible), 200 random secrets per stub: random proposals, proposals with repeated digits, and
replies with no number all solved 100% in 5.3-5.4 guesses (max 8), ~2.0k tokens/ep (rough
estimate; real Qwen replies may be longer, since the native loop does not cap them).
"""
from __future__ import annotations

import random
import re
from itertools import permutations

from ..harness_base import GNHarness

ALL = ["".join(p) for p in permutations("0123456789", 4)]
_MASK = {n: sum(1 << int(c) for c in n) for n in ALL}
_POP = [bin(i).count("1") for i in range(1024)]
FOUR = re.compile(r"(?<!\d)(\d{4})(?!\d)")

N_TIE = 60        # how many equally near survivors are scored for the tie-break
N_REF = 300       # against how many survivors each is scored


def _mask(n):
    m = _MASK.get(n)
    return m if m is not None else sum(1 << int(c) for c in set(n))


def score(guess, secret):
    """(in place, misplaced) if `secret` were the answer -- the oracle's rule, on a hypothesis."""
    a = sum(1 for x, y in zip(guess, secret) if x == y)
    return a, _POP[_mask(guess) & _mask(secret)] - a


def consistent(history):
    s = ALL
    for g, a, b in history:
        s = [n for n in s if score(g, n) == (a, b)]
    return s


def expected_left(g, ref):
    cells = {}
    for s in ref:
        k = score(g, s)
        cells[k] = cells.get(k, 0) + 1
    cells.pop((4, 0), None)
    return sum(c * c for c in cells.values())


def project(proposals, survivors, rng):
    """The survivor nearest the model's proposals (see the module docstring)."""
    if len(survivors) == 1:
        return survivors[0]
    best, near = None, []
    for rank, p in enumerate(proposals or [""]):
        pm = _mask(p) if p else 0
        for n in survivors:
            same = sum(1 for x, y in zip(p, n) if x == y)
            key = (2 * same + _POP[pm & _MASK[n]], -rank)
            if best is None or key > best:
                best, near = key, [n]
            elif key == best:
                near.append(n)
    if len(near) == 1:
        return near[0]
    cands = near if len(near) <= N_TIE else rng.sample(near, N_TIE)
    ref = survivors if len(survivors) <= N_REF else rng.sample(survivors, N_REF)
    return min(cands, key=lambda g: expected_left(g, ref))


class Harness(GNHarness):
    def solve(self) -> str:
        env = self.env
        last = None
        while env.remaining > 0 and not env.solved:
            out = self.llm([{"role": "user", "content": env.turn_prompt()}])
            proposals = FOUR.findall(out or "")
            survivors = consistent(env.history)
            alive = set(survivors)
            last = next((p for p in proposals if p in alive), None) or \
                project(proposals, survivors, random.Random(repr(env.history)))
            env.guess(last)
        return last
