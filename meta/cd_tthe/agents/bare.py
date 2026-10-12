"""Seed harness: the benchmark's own protocol, nothing else. The frozen model is the only
decision maker; the harness just relays the environment's lines (paper wording)."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")


class Harness(CDHarness):
    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        last_line = None
        for t in range(1, 19):
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            m = QUERY_RE.search(reply or "")
            if not m:
                # the benchmark gives no second chance: the turn is spent
                msgs.append({"role": "user", "content": "Your message did not contain a query in the "
                             "required format. Reply with exactly one <query>X(a, b, c)</query>."})
                if t == 18:
                    last_line = None      # the previous answer was already delivered with its own prompt
                continue
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            last_line = env.format_answer(c, a, b, x, bit)
            if t < 18:
                msgs.append({"role": "user", "content": f"{last_line}\nPlease send your next query. {18 - t} queries remaining."})
        final = (f"{last_line}\n" if last_line else "") + env.last_turn_prompt
        msgs.append({"role": "user", "content": final})
        return self.llm(msgs)
