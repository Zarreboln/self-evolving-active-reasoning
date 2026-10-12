"""Conservative repair of `bare`: same relay loop and wording, plus two trace-driven fixes.

1. Query spread. In the bare traces the model walks A's 8 rows, then B's, and reaches C with 0-2
   queries left (and sometimes repeats a row). The harness now vetoes a query on a circuit that has
   already had its share (ceil(budget / 3)) or on a row already asked, and re-asks the model; a vetoed
   reply does not spend a query.
2. Final transcription. Every bare final guess contradicted rows the model had itself observed.
   The final prompt now shows the observed table ('?' for unasked rows); the model fills the
   '?'s, and the harness keeps the observed bits for asked rows and the model's bits elsewhere
   (its 24-bit guess, else its per-row statements, else a short re-ask for the guess).
"""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"<guess>\s*([01\s]{24,})\s*</guess>")
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
ROW_RE = re.compile(r"([A-Ca-c])\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*=\s*([01])\b")
MAX_REASKS = 3
FINAL_TOKENS = 2400
FINAL_REASKS = 2


class Harness(CDHarness):
    def known(self):
        return {q: bit for q, bit in self.env.history}

    def table_text(self):
        kn = self.known()
        lines = []
        for c in "ABC":
            cells = [f"{c}({a},{b},{x})={kn.get((c, a, b, x), '?')}" for a, b, x in ROWS]
            lines.append("  ".join(cells))
        return "\n".join(lines)

    def veto_reason(self, c, a, b, x, cap):
        kn = self.known()
        if (c, a, b, x) in kn:
            return f"You already asked {c}({a}, {b}, {x}) and got {kn[(c, a, b, x)]}."
        if sum(1 for q in kn if q[0] == c) >= cap:
            others = [o for o in "ABC" if sum(1 for q in kn if q[0] == o) < cap]
            return (f"Circuit {c} has already had its share of {cap} queries; the rest of its rows must be "
                    f"deduced. Query circuit {' or '.join(others)} instead.")
        return None

    def solve(self):
        env = self.env
        B = env.budget
        cap = -(-B // 3)
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        last_line = None
        t = 0
        while t < B and env.remaining > 0:
            t += 1
            accepted = None
            for attempt in range(MAX_REASKS + 1):
                reply = self.llm(msgs)
                msgs.append({"role": "assistant", "content": reply})
                m = QUERY_RE.search(reply or "")
                if not m:
                    msgs.append({"role": "user", "content": "Your message did not contain a query in the "
                                 "required format. Reply with exactly one <query>X(a, b, c)</query>."})
                    continue
                c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
                why = self.veto_reason(c, a, b, x, cap)
                if why:
                    msgs.append({"role": "user", "content": f"{why}\nObserved so far:\n{self.table_text()}\n"
                                 "Send a different query."})
                    continue
                accepted = (c, a, b, x)
                break
            if accepted is None:
                continue          # turn spent, as in the benchmark
            c, a, b, x = accepted
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            last_line = env.format_answer(c, a, b, x, bit)
            if env.remaining > 0:
                msgs.append({"role": "user", "content": f"{last_line}\nPlease send your next query. "
                             f"{env.remaining} queries remaining."})
        final = ((f"{last_line}\n" if last_line else "") +
                 "All observed outputs ('?' = not queried; deduce those from the circuit rules and your "
                 f"inventory):\n{self.table_text()}\n" + env.last_turn_prompt)
        msgs.append({"role": "user", "content": final})
        reply = self.llm(msgs, max_tokens=FINAL_TOKENS)
        out = self.assemble(reply)
        for _ in range(FINAL_REASKS):
            if out:
                break
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": "Your answer did not contain a 24-bit guess. Fill in each '?' "
                         f"below with your deduced value:\n{self.table_text()}\nReply with only "
                         "<guess>XXXXXXXXXXXXXXXXXXXXXXXX</guess> (exactly 24 bits: A rows, then B, then C)."})
            reply = self.llm(msgs, max_tokens=FINAL_TOKENS)
            out = self.assemble(reply)
        return out or reply

    def assemble(self, reply):
        """Observed bits for asked rows; the model's bits for the rest -- from a 24-bit guess if it
        wrote one, else from its explicit 'X(a,b,c) = v' row statements. None if any row is missing."""
        m = GUESS_RE.search(reply or "")
        bits = re.sub(r"\s", "", m.group(1)) if m else None
        if not bits or len(bits) != 24:
            found = re.findall(r"[01]{24}", reply or "")
            bits = found[-1] if found else None
        stated = {}
        for r in ROW_RE.finditer(reply or ""):
            stated[(r.group(1).upper(), int(r.group(2)), int(r.group(3)), int(r.group(4)))] = r.group(5)
        kn = self.known()
        out = []
        for i, c in enumerate("ABC"):
            for j, row in enumerate(ROWS):
                q = (c,) + row
                if q in kn:
                    out.append(str(kn[q]))
                elif bits:
                    out.append(bits[i * 8 + j])
                elif q in stated:
                    out.append(stated[q])
                else:
                    return None
        return f"<guess>{''.join(out)}</guess>"
