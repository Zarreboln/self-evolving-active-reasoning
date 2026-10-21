"""The context-evolution arm: the harness is fixed, only the text evolves.

In the code arm a candidate is an arbitrary Python harness. Here the loop is frozen to the
benchmark's official zero-shot loop and the *only* thing a candidate may change is one string,
`PLAYBOOK`, which is injected into the system message every turn. Everything else — the stream,
label isolation, the session audit, the label-free judge, the frozen evaluation — is unchanged, so
the two arms differ in exactly one variable: whether the evolving artifact is code or context.

This is the setting of the context-evolution literature: Dynamic Cheatsheet (arXiv 2504.07952)
curates a persistent memory at test time with no labels; ACE (arXiv 2510.04618) argues such a
context should be a detailed structured playbook updated by deltas rather than rewritten whole,
because monolithic rewriting collapses it into ever shorter summaries. Our proposer rewrites the
whole artifact each round, which is precisely the regime ACE warns about — so running this arm
tests that warning on our tasks.

A candidate file looks exactly like this and nothing more:

    \"\"\"One line saying what this playbook changes.\"\"\"
    from ctx_harness import SPCtxHarness

    class Harness(SPCtxHarness):
        PLAYBOOK = \"\"\"
        - ...
        \"\"\"

`ctx_audit.text_only` rejects anything else: no function definitions, no control flow, no second
assignment. That is what makes this a controlled comparison rather than a second code arm.
"""

from __future__ import annotations

import re

from harness_base import CDHarness
from sp_harness_base import SPHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")

HEADER = ("\n\n# Playbook\n"
          "Notes collected from earlier episodes of this task. They are guidance, not facts about\n"
          "this instance. Follow them unless this instance contradicts them.\n")


def _system(base: str, playbook: str) -> str:
    p = (playbook or "").strip()
    return base if not p else base + HEADER + p + "\n"


class SPCtxHarness(SPHarness):
    """AR-Bench's official Situation Puzzles loop, with the playbook in the system message."""

    PLAYBOOK = ""

    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": _system(env.system_prompt, self.PLAYBOOK)}]
        for t in range(1, env.budget + 1):
            msgs.append({"role": "user", "content": env.turn_prompt(t)})
            q = self.llm(msgs) or ""
            msgs.append({"role": "assistant", "content": q})
            msgs.append({"role": "user", "content": env.ask(q)})
        msgs.append({"role": "user", "content": env.final_prompt})
        return self.llm(msgs) or ""


class CDCtxHarness(CDHarness):
    """Circuit Decoding's own protocol (identical to agents/bare.py), with the playbook in the
    system message. Nothing else about the loop may differ, or the two arms stop being comparable."""

    PLAYBOOK = ""

    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": _system(env.system_prompt, self.PLAYBOOK)},
                {"role": "user", "content": "Please start with your first query."}]
        last_line, B = None, env.budget
        for t in range(1, B + 1):
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            m = QUERY_RE.search(reply or "")
            if not m:
                msgs.append({"role": "user", "content": "Your message did not contain a query in the "
                             "required format. Reply with exactly one <query>X(a, b, c)</query>."})
                if t == B:
                    last_line = None
                continue
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            last_line = env.format_answer(c, a, b, x, bit)
            if t < B:
                msgs.append({"role": "user",
                             "content": f"{last_line}\nPlease send your next query. {B - t} queries remaining."})
        msgs.append({"role": "user", "content": (f"{last_line}\n" if last_line else "") + env.last_turn_prompt})
        return self.llm(msgs)
