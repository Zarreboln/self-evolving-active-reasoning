"""Ledger harness: the harness, not the chat, holds the episode state. Every turn the model sees
the benchmark rules plus a fresh 24-row grid of everything observed so far ('?' = not queried yet),
instead of a growing transcript it has to re-read. The final guess is asked per circuit against
the same grid; rows the environment already answered are taken from those answers, the model
only fills the '?' rows."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
# the model often writes ```query\nA(0, 1, 1)\n``` -- still its own query, just a different wrapper
FENCED_RE = re.compile(r"```\s*query\s*\n\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)")
BITS_RE = re.compile(r"<bits>\s*([01?](?:\s*[01?]){7})\s*</bits>")
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]


class Harness(CDHarness):
    def ledger(self):
        seen = {q: bit for q, bit in self.env.history}
        lines = []
        for c in "ABC":
            n = sum(1 for r in ROWS if (c,) + r in seen)
            lines.append(f"Circuit {c} ({n}/8 rows known):")
            for r in ROWS:
                v = seen.get((c,) + r)
                lines.append(f"  {c}({r[0]}, {r[1]}, {r[2]}) = {'?' if v is None else v}")
        return "\n".join(lines), seen

    def ask_query(self, t, B):
        grid, seen = self.ledger()
        msgs = [{"role": "system", "content": self.env.system_prompt},
                {"role": "user", "content": f"Everything observed so far ('?' = not queried yet):\n{grid}\n\n"
                 f"This is query {t} of {B} ({B - t + 1} remaining, including this one). Think about which "
                 f"circuit structures are still possible, then send exactly one <query>X(a, b, c)</query> "
                 f"for a row that is still '?'."}]
        for _ in range(3):
            reply = self.llm(msgs)
            m = QUERY_RE.search(reply or "") or FENCED_RE.search(reply or "")
            if not m:
                note = "Your message did not contain a query in the required format."
            else:
                q = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                if q not in seen:
                    return q
                note = f"{self.env.format_answer(*q, seen[q])} is already known (see the grid)."
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"{note} Reply with exactly one <query>X(a, b, c)</query> "
                      f"for a row that is still '?'."}]
        return None

    def ask_circuit(self, c):
        grid, seen = self.ledger()
        msgs = [{"role": "system", "content": self.env.system_prompt},
                {"role": "user", "content": f"The queries are over. Everything observed ('?' = never queried):\n"
                 f"{grid}\n\nNow give the 8 outputs of circuit {c} only, in the order {c}(0,0,0) ... {c}(1,1,1). "
                 f"Copy the known rows exactly; for each '?' row, work out which circuit built from the "
                 f"inventory fits the known rows and use its output. End with <bits>XXXXXXXX</bits> "
                 f"(8 digits, 0 or 1)."}]
        bits = None
        for _ in range(2):
            m = None
            for m in BITS_RE.finditer(self.llm(msgs, temperature=0.3) or ""):
                pass
            if m:
                bits = re.sub(r"\s", "", m.group(1))
                if "?" not in bits:
                    break
        bits = bits or "?" * 8
        out = []
        for r, b in zip(ROWS, bits):
            v = seen.get((c,) + r)
            out.append(str(v) if v is not None else (b if b in "01" else "0"))
        return "".join(out)

    def solve(self):
        env = self.env
        B = env.budget
        for t in range(1, B + 1):
            q = self.ask_query(t, B)
            if q is None:
                continue          # turn spent, as in the benchmark
            try:
                env.query(*q)
            except Exception:
                break
        guess = "".join(self.ask_circuit(c) for c in "ABC")
        return f"<guess>{guess}</guess>"
