"""Ledger harness: the harness, not the 7B's context window, holds the episode state.

Hypothesis: bare episodes fail at bookkeeping, not deduction -- the model loses track of which
rows it has seen, re-spends queries, piles them on one circuit, and finally cannot transcribe its
own observations into one 24-bit string (wrong lengths, observed rows flipped). So:
  * every turn the model sees only the rules + a per-circuit table of known rows and '?' rows;
  * a query for an already-known row (or an unparseable reply) is vetoed and re-asked, no budget spent;
  * the final guess is assembled per circuit: observed rows are locked, and the model fills the '?'
    rows (it writes them as `X(a, b, c) = bit` lines, which the harness reads back).
The model still writes every query and every unobserved bit."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
ROW_RE = re.compile(r"([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*(?:=|:|->|is)\s*\**\s*([01])\b")
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
MAX_RETRIES = 4


class Harness(CDHarness):
    def _known(self):
        known = {}
        for (c, a, b, x), bit in self.env.history:
            known[(c, a, b, x)] = bit
        return known

    def _table(self, known, circuits="ABC"):
        lines = []
        for c in circuits:
            n = sum(1 for r in ROWS if (c,) + r in known)
            lines.append(f"Circuit {c} ({n} of 8 rows known):")
            for r in ROWS:
                v = known.get((c,) + r)
                lines.append(f"  {c}({r[0]}, {r[1]}, {r[2]}) = {'?' if v is None else v}")
        return "\n".join(lines)

    def _turn_prompt(self, known, note=""):
        env = self.env
        unknown = {c: sum(1 for r in ROWS if (c,) + r not in known) for c in "ABC"}
        last = ""
        if env.history:
            (c, a, b, x), bit = env.history[-1]
            last = f"Last answer: {env.format_answer(c, a, b, x, bit)}\n\n"
        return (f"{last}Everything observed so far ('?' = not yet queried):\n{self._table(known)}\n\n"
                f"Queries remaining: {env.remaining}. Unknown rows: A {unknown['A']}, B {unknown['B']}, "
                f"C {unknown['C']} ({sum(unknown.values())} in total) -- you cannot query them all, so "
                "spread your queries over the three circuits and use the circuit-building rules and the "
                "gate inventory to infer the rows you do not query. Querying a row that is already known "
                "gives no new information.\n"
                f"{note}Think briefly, then send exactly one query for a '?' row as <query>X(a, b, c)</query>.")

    def solve(self):
        env = self.env
        sys_msg = {"role": "system", "content": env.system_prompt}
        while env.remaining > 0:
            known = self._known()
            note, pick = "", None
            for _ in range(MAX_RETRIES):
                reply = self.llm([sys_msg, {"role": "user", "content": self._turn_prompt(known, note)}])
                m = QUERY_RE.search(reply or "")
                if not m:
                    note = "(Your previous reply had no query in the required format.) "
                    continue
                q = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                pick = q
                if q in known:
                    note = (f"(You asked for {q[0]}({q[1]}, {q[2]}, {q[3]}), but it is already known: "
                            f"{known[q]}. Pick a row marked '?'.) ")
                    continue
                break
            if pick is None:
                break                     # the model never produced a query: stop asking
            try:
                env.query(*pick)
            except Exception:
                break
        return self._final()

    def _final(self):
        env = self.env
        known = self._known()
        sys_msg = {"role": "system", "content": env.system_prompt}
        bits = {}
        for c in "ABC":
            missing = [r for r in ROWS if (c,) + r not in known]
            filled = {}
            note = ""
            for _ in range(MAX_RETRIES if missing else 0):
                todo = [r for r in missing if r not in filled]
                if not todo:
                    break
                ask = "\n".join(f"  {c}({r[0]}, {r[1]}, {r[2]}) = ?" for r in todo)
                prompt = (f"The queries are over. Everything observed ('?' = never queried):\n{self._table(known)}\n\n"
                          f"Now complete circuit {c}. Its observed rows are fixed. Using the circuit-building "
                          f"rules and the gate inventory, decide which gates circuit {c} most likely uses and how "
                          f"they are wired so that every observed row of {c} is reproduced, then give its "
                          f"output on these rows:\n{ask}\n{note}"
                          f"Think briefly, then end with one line per row above, exactly in the form "
                          f"{c}(a, b, c) = 0 or {c}(a, b, c) = 1.")
                reply = self.llm([sys_msg, {"role": "user", "content": prompt}])
                for m in ROW_RE.finditer(reply or ""):
                    r = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                    if m.group(1).upper() == c and r in todo:
                        filled[r] = int(m.group(5))     # the last statement for a row wins
                note = "(Your previous reply did not state every row in the required form.)\n"
            for r in ROWS:
                v = known.get((c,) + r)
                bits[(c,) + r] = v if v is not None else filled.get(r, 0)
        guess = "".join(str(bits[(c,) + r]) for c in "ABC" for r in ROWS)
        return f"<guess>{guess}</guess>"
