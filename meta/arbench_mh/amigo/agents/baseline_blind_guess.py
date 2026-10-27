"""The anchor: name a dress without asking anything.

It scores the chance rate, which differs per tier because the galleries differ in size -- 4.1% at
tau=0.3 where a gallery holds 26 dresses, 14.7% at tau=0.8 where it holds 7. Having it on the
frontier as a real point rather than a dashed line is what makes "above chance" a thing the figure
shows instead of a thing the caption asserts. It is also the cheapest possible episode, so it
fixes the left end of the cost axis.

The model still names the dress: the harness never picks for it.
"""
from __future__ import annotations

import re

from ..harness_base import AMIGOHarness

NUM = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")


class Harness(AMIGOHarness):
    def solve(self) -> int:
        env = self.env
        content = [{"type": "text", "text": "Here is a gallery of dresses."}]
        content += env.gallery_blocks()
        content.append({"type": "text", "text":
                        "One of them is my favourite. Name which one, by its number from 1 to %d. "
                        "Reply with the number only." % env.n})
        out = self.llm([{"role": "user", "content": content}], max_tokens=8) or ""
        m = NUM.search(out)
        i = int(m.group(1)) - 1 if m else 0
        return i if 0 <= i < env.n else 0
