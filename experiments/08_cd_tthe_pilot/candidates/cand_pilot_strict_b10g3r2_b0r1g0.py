"""Conservative repair of `cand_pilot_strict_b10g3r2_b0r0g0`: same protocol and wording, plus two trace-driven fixes.
Kept from the parent: non-query replies are re-asked (bounded); each answer line carries per-circuit
query counts; the final prompt recaps every answer received, row by row in guess order.
1. Query spread: the parent's model swept A and B row by row and left C with 2 queries. A query on a
   circuit that already has its fair share (budget // 3) while another circuit is below it, or a query
   that was already answered, is vetoed and the model re-asks (bounded; it does not burn budget).
2. Final assembly: the parent's model never wrote a 24-bit string consistent with its own answers
   (wrong length or mis-copied bits, 10/10 episodes). Now the model states a value for each unqueried
   row as `X(a,b,c) = v`; the harness assembles the 24 bits from the received answers plus those
   statements (falling back to the model's <guess> bits), re-asking for rows it left out or for a
   circuit that came out constant (forbidden by the rules)."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"<guess>\s*([01\s]*?)\s*</guess>")
CELL_RE = re.compile(r"\b([ABC])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*=\s*([01])\b")
MAX_BAD_REPLIES = 6        # total non-query replies tolerated per episode before giving up on queries
MAX_VETOES = 12            # total vetoed queries per episode; after that every well-formed query is sent
MAX_FINAL_RETRIES = 3


def _cell(c, a, b, x):
    return "ABC".index(c) * 8 + a * 4 + b * 2 + x


def _name(i):
    return f"{'ABC'[i // 8]}({(i >> 2) & 1},{(i >> 1) & 1},{i & 1})"


class Harness(CDHarness):
    def _n(self):
        n = {c: 0 for c in "ABC"}
        for (c, _, _, _), _ in self.env.history:
            n[c] += 1
        return n

    def _counts(self):
        n = self._n()
        return ("Queries so far: " + ", ".join(f"{c}: {n[c]}" for c in "ABC")
                + f". Spread them: at most {self._share} per circuit.")

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

    def _unasked(self, c):
        seen = {q for q, _ in self.env.history}
        return [_name(_cell(c, a, b, x)) for a in (0, 1) for b in (0, 1) for x in (0, 1)
                if (c, a, b, x) not in seen]

    def _veto(self, c, a, b, x):
        seen = {q: bit for q, bit in self.env.history}
        if (c, a, b, x) in seen:
            return (f"You already asked {c}({a}, {b}, {x}); the answer was {seen[(c, a, b, x)]}. "
                    "Ask a row you have not asked yet.")
        n = self._n()
        low = [d for d in "ABC" if d != c and n[d] < self._share]
        if n[c] >= self._share and low:
            return (f"Circuit {c} already has {n[c]} queries, its share of the budget. "
                    + " ".join(f"Circuit {d} has only {n[d]}; its unasked rows: {', '.join(self._unasked(d))}."
                               for d in low)
                    + f" Query one of those circuits instead. {self.env.remaining} queries remaining.")
        return None

    def _read(self, reply, cells):
        """Fill unknown cells (None) from the model's `X(a,b,c) = v` statements, then from its <guess>."""
        stated = {}
        for c, a, b, x, v in CELL_RE.findall(reply or ""):
            stated[_cell(c, int(a), int(b), int(x))] = int(v)      # last statement wins
        m = GUESS_RE.findall(reply or "")
        bits = re.sub(r"\s", "", m[-1]) if m else ""
        for i in range(24):
            if cells[i] is None:
                if i in stated:
                    cells[i] = stated[i]
                elif len(bits) == 24:
                    cells[i] = int(bits[i])

    def solve(self):
        env = self.env
        self._share = max(1, env.budget // 3)
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        last_line = None
        bad_replies = vetoes = 0
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
            why = self._veto(c, a, b, x) if vetoes < MAX_VETOES else None
            if why:
                vetoes += 1
                msgs.append({"role": "user", "content": why + " Reply with exactly one <query>X(a, b, c)</query>."})
                continue
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            last_line = env.format_answer(c, a, b, x, bit)
            if env.remaining > 0:
                msgs.append({"role": "user", "content": f"{last_line}\n{self._counts()}\n"
                             f"Please send your next query. {env.remaining} queries remaining."})

        known = [None] * 24
        for (c, a, b, x), bit in env.history:
            known[_cell(c, a, b, x)] = bit
        final = ((f"{last_line}\n" if last_line else "") + self._recap() + "\n"
                 "First, for EVERY row marked ?, write one line `X(a,b,c) = v` with the value you infer "
                 "for it (for example `C(0,1,0) = 1`). " + env.last_turn_prompt)
        msgs.append({"role": "user", "content": final})
        cells = list(known)
        for attempt in range(MAX_FINAL_RETRIES + 1):
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            self._read(reply, cells)
            missing = [i for i in range(24) if cells[i] is None]
            constant = [c for k, c in enumerate("ABC")
                        if None not in cells[8 * k:8 * k + 8] and len(set(cells[8 * k:8 * k + 8])) == 1]
            if (not missing and not constant) or attempt == MAX_FINAL_RETRIES:
                break
            problems = []
            if missing:
                problems.append("You did not give a value for: " + ", ".join(_name(i) for i in missing) + ".")
            for c in constant:
                k = "ABC".index(c)
                problems.append(f"Your table makes circuit {c} output {cells[8 * k]} on every row, but the rules "
                                "say no circuit is constant. Reconsider its ? rows: "
                                + ", ".join(self._unasked(c)) + ".")
                for i in range(8 * k, 8 * k + 8):
                    cells[i] = known[i]          # let the model restate this circuit's inferred rows
            msgs.append({"role": "user", "content": "\n".join(problems) + "\n" + self._recap()
                         + "\nWrite one line `X(a,b,c) = v` for each of these rows, then the full 24-bit "
                         "table inside <guess> </guess>."})
        bits = "".join(str(0 if v is None else v) for v in cells)
        return f"<guess>{bits}</guess>"
