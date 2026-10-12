"""Stateless-ledger harness. Hypothesis: the 7B loses track of its own observations inside the
growing chat (36+ messages by the end); the bare traces show final guesses contradicting rows it
had itself queried, and query choices made without a view of what is still unknown.

One mechanism: no running conversation. Every model call is a fresh two-message prompt -- the
benchmark's system prompt plus a ledger the harness rebuilds from env.history (every circuit's
8 rows, observed bit or '?', and the count of unasked rows per circuit). The model still writes
every query and every bit of the final guess; the harness never vetoes or overrides its content.
A reply without a parseable query / guess is re-asked (costs no query).
"""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"[01]{24}")
LINE_RE = {c: re.compile(rf"\b{c}\s*:\s*([01]{{8}})\b") for c in "ABC"}
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
MAX_REASKS = 2


class Harness(CDHarness):
    def ledger(self):
        known = {q: bit for q, bit in self.env.history}
        done = [self.env.format_answer(c, a, b, x, known[(c, a, b, x)])
                for c in "ABC" for a, b, x in ROWS if (c, a, b, x) in known]
        lines = [f"You are in the MIDDLE of the game. You have ALREADY made {len(self.env.history)} "
                 f"queries; their answers are recorded here, so do not start over."]
        lines.append("Answers already received:" + ("\n- " + "\n- ".join(done) if done else " none yet."))
        lines.append("Tables so far (rows 000..111, '?' = not queried yet):")
        for c in "ABC":
            lines.append(f"  {c}: " + "".join(str(known.get((c, a, b, x), "?")) for a, b, x in ROWS))
        todo = {c: [f"{c}({a}, {b}, {x})" for a, b, x in ROWS if (c, a, b, x) not in known] for c in "ABC"}
        lines.append("Rows still unknown:")
        for c in "ABC":
            lines.append(f"  {c} ({len(todo[c])} unknown): " + (", ".join(todo[c]) or "none"))
        return "\n".join(lines)

    def ask(self, text):
        return self.llm([{"role": "system", "content": self.env.system_prompt},
                         {"role": "user", "content": text}])

    def solve(self):
        env = self.env
        while env.remaining > 0:
            known = {q for q, _ in env.history}
            prompt = (f"{self.ledger()}\nQueries remaining: {env.remaining} of {env.budget}.\n"
                      "Pick ONE row that is still unknown and reply with just that query, "
                      "e.g. <query>B(1, 0, 1)</query>.")
            m, note = None, ""
            for _ in range(1 + MAX_REASKS):
                reply = self.ask(prompt + note) or ""
                m = QUERY_RE.search(reply)
                if not m:
                    note = "\n(Your last reply had no <query>X(a, b, c)</query>. Send exactly one.)"
                    continue
                q = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                if q not in known:
                    break
                note = (f"\n(Your last reply asked {q[0]}({q[1]}, {q[2]}, {q[3]}), which is already "
                        "answered above. Choose a row from 'Rows still unknown'.)")
            if not m:
                break
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                env.query(c, a, b, x)
            except Exception:
                break
        final = (f"{self.ledger()}\nAll {env.budget} queries are used; no more queries.\n"
                 "Fill in every '?' using the task rules (gate inventory, how circuits are built). "
                 "Rows already answered must keep their observed value. Be brief: write one line per "
                 "circuit as 'A: <8 bits>', 'B: <8 bits>', 'C: <8 bits>', then the final answer.\n"
                 f"{env.last_turn_prompt}")
        reply, note = "", ""
        for _ in range(1 + MAX_REASKS):
            reply = self.ask(final + note) or ""
            lines = [LINE_RE[c].findall(reply) for c in "ABC"]
            if all(lines):      # the model's own 'A: / B: / C:' lines, joined in benchmark order
                return f"{reply}\n<guess>{''.join(l[-1] for l in lines)}</guess>"
            if GUESS_RE.search(reply.replace(" ", "")):
                break
            note = "\n(Your last reply had no 24-bit <guess>. Keep it short and end with <guess>...</guess>.)"
        return reply
