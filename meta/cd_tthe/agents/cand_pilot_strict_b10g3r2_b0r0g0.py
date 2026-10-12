"""Conservative repair of `bare`: same protocol and wording, plus three trace-driven fixes.
1. A reply without a query no longer burns a turn: the model is re-asked (bounded) until it queries.
2. Each answer line also carries how many queries each circuit has had so far.
3. The final prompt recaps every answer received, row by row in guess order; a guess that is not
   24 bits or that contradicts a received answer is sent back to the model with the conflicts
   listed (bounded), and the least-conflicting reply is returned."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"<guess>\s*([01\s]*?)\s*</guess>")
MAX_BAD_REPLIES = 6        # total non-query replies tolerated per episode before giving up on queries
MAX_FINAL_RETRIES = 3


class Harness(CDHarness):
    def _counts(self):
        n = {c: 0 for c in "ABC"}
        for (c, _, _, _), _ in self.env.history:
            n[c] += 1
        return "Queries so far: " + ", ".join(f"{c}: {n[c]}" for c in "ABC") + "."

    def _recap(self):
        seen = {q: bit for q, bit in self.env.history}
        lines = []
        for c in "ABC":
            rows = []
            for i in range(8):
                a, b, x = (i >> 2) & 1, (i >> 1) & 1, i & 1
                v = seen.get((c, a, b, x))
                rows.append(f"{c}({a},{b},{x})={'?' if v is None else v}")
            lines.append("  ".join(rows))
        return ("All answers you received (? = not queried, you must infer it):\n" + "\n".join(lines))

    def _check(self, reply):
        m = GUESS_RE.findall(reply or "")
        bits = re.sub(r"\s", "", m[-1]) if m else ""
        if len(bits) != 24:
            return None, [f"your <guess> has {len(bits)} bits; it must have exactly 24 (8 for A, then 8 for B, then 8 for C)"]
        bad = []
        for (c, a, b, x), bit in self.env.history:
            i = "ABC".index(c) * 8 + a * 4 + b * 2 + x
            if int(bits[i]) != bit:
                bad.append(f"your guess says {c}({a},{b},{x})={bits[i]} but the answer you received was {bit}")
        return bits, bad

    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        last_line = None
        bad_replies = 0
        while env.remaining > 0:
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            m = QUERY_RE.search(reply or "")
            if not m:
                bad_replies += 1
                if bad_replies > MAX_BAD_REPLIES:
                    break
                msgs.append({"role": "user", "content": "Your message did not contain a query in the "
                             f"required format. You still have {env.remaining} queries; do not give a guess yet. "
                             "Reply with exactly one <query>X(a, b, c)</query>."})
                continue
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            last_line = env.format_answer(c, a, b, x, bit)
            if env.remaining > 0:
                msgs.append({"role": "user", "content": f"{last_line}\n{self._counts()}\n"
                             f"Please send your next query. {env.remaining} queries remaining."})
        final = (f"{last_line}\n" if last_line else "") + self._recap() + "\n" + env.last_turn_prompt
        msgs.append({"role": "user", "content": final})
        reply = self.llm(msgs)
        best, best_score = reply, None
        for attempt in range(MAX_FINAL_RETRIES + 1):
            bits, problems = self._check(reply)
            score = len(problems) + (100 if bits is None else 0)
            if best_score is None or score < best_score:
                best, best_score = reply, score
            if not problems or attempt == MAX_FINAL_RETRIES:
                break
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": "Your guess conflicts with the answers you received:\n- "
                         + "\n- ".join(problems) + "\n" + self._recap()
                         + "\nFix these and give the full 24-bit table again inside <guess> </guess>."})
            reply = self.llm(msgs)
        return best
