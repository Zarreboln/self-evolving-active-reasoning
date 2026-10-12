"""Conservative repair of `bare`: same relay loop and wording, plus two trace-driven fixes.

1. Query spread. In the bare traces the model walks A's 8 rows, then B's, and reaches C with 0-2
   queries left (and sometimes repeats a row). The harness now vetoes a query on a circuit that has
   already had its share (ceil(budget / 3)) or on a row already asked, and re-asks the model; a vetoed
   reply does not spend a query.
2. Final transcription. Every bare final guess contradicted rows the model had itself observed.
   The final prompt now shows the observed table ('?' for unasked rows); the model fills the
   '?'s, and the harness keeps the observed bits for asked rows and the model's bits elsewhere
   (its 24-bit guess, else its per-row statements, else a short re-ask for the guess).
3. Rule-consistent completion (b0r1). In the parent's traces the 18 answers already pin most circuits
   down under the task rules, yet the model's '?' fills contradicted them. The harness now derives,
   from the system-prompt rules alone (two 2-input gates per circuit, NOTs anywhere, no constant
   circuit, inventory used exactly), which tables remain consistent with the answers; the final table
   shows rule-implied rows instead of '?', lists the candidates for any still-open circuit, and each
   circuit of the assembled guess is snapped to the consistent table nearest the model's. Queries
   whose answer the rules already force are vetoed like repeats (the parent spent ~40% of its queries
   on them, leaving circuits ambiguous); the share cap no longer blocks a circuit once no other
   circuit has open rows; querying stops when every row is forced.
"""

import itertools
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"<guess>\s*([01\s]{24,})\s*</guess>")
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
ROW_RE = re.compile(r"([A-Ca-c])\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*=\s*([01])\b")
MAX_REASKS = 3
FINAL_TOKENS = 2400
FINAL_REASKS = 2


def _tables(n_and, n_or, n_not):
    """8-bit tables of a 3-input circuit with exactly these gates: g = op1(xi, xj), out = op2(g, xk),
    with NOTs (stackable) on any of the five signals."""
    if n_and + n_or != 2:
        return set()
    out = set()
    pars = [p for p in itertools.product((0, 1), repeat=5) if sum(p) <= n_not and (n_not - sum(p)) % 2 == 0]
    for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        for op1, op2 in set(itertools.permutations(["AND"] * n_and + ["OR"] * n_or)):
            for p in pars:
                tab = []
                for row in ROWS:
                    u, v = row[i] ^ p[0], row[j] ^ p[1]
                    g = ((u & v) if op1 == "AND" else (u | v)) ^ p[3]
                    w = row[k] ^ p[2]
                    tab.append(((g & w) if op2 == "AND" else (g | w)) ^ p[4])
                if 0 < sum(tab) < 8:
                    out.add(tuple(tab))
    return out


class Harness(CDHarness):
    def consistent(self):
        """Per circuit, the tables consistent with every answer under some split of the inventory
        that also fits the other circuits' answers. Empty dict if the rules admit nothing."""
        inv = self.env.inventory
        n_and, n_or, n_not = inv.get("AND", 0), inv.get("OR", 0), inv.get("NOT", 0)
        kn = self.known()
        cache, marg = {}, {c: set() for c in "ABC"}
        for ands in itertools.product(range(3), repeat=3):
            if sum(ands) != n_and or sum(2 - a for a in ands) != n_or:
                continue
            for nots in itertools.product(range(n_not + 1), repeat=3):
                if sum(nots) != n_not:
                    continue
                filt = []
                for c, a, n in zip("ABC", ands, nots):
                    if (a, n) not in cache:
                        cache[(a, n)] = _tables(a, 2 - a, n)
                    filt.append({t for t in cache[(a, n)]
                                 if all(t[j] == kn[(c,) + r] for j, r in enumerate(ROWS) if (c,) + r in kn)})
                if all(filt):
                    for c, f in zip("ABC", filt):
                        marg[c] |= f
        return marg if all(marg.values()) else {}

    def known(self):
        return {q: bit for q, bit in self.env.history}

    def table_text(self):
        kn = self.known()
        lines = []
        for c in "ABC":
            cells = [f"{c}({a},{b},{x})={kn.get((c, a, b, x), '?')}" for a, b, x in ROWS]
            lines.append("  ".join(cells))
        return "\n".join(lines)

    def deduced_text(self):
        """Rows the rules force, and the remaining candidate tables of any circuit still open."""
        if not self._cons:
            return ""
        kn, lines = self.known(), []
        for c in "ABC":
            cands = sorted(self._cons[c])
            if len(cands) == 1:
                lines.append(f"Circuit {c} is fully determined by these answers and the rules: "
                             + "".join(map(str, cands[0])))
                continue
            forced = [f"{c}({a},{b},{x})={cands[0][j]}" for j, (a, b, x) in enumerate(ROWS)
                      if (c, a, b, x) not in kn and len({t[j] for t in cands}) == 1]
            lines.append(f"Circuit {c} is one of these tables (rows 000..111): "
                         + ", ".join("".join(map(str, t)) for t in cands[:12])
                         + (f"; forced rows: {', '.join(forced)}" if forced else ""))
        return "Deduced from the rules:\n" + "\n".join(lines) + "\n"

    def open_rows(self, cons):
        """Per circuit, the unasked rows whose value the rules do not force yet."""
        kn = self.known()
        return {c: [r for j, r in enumerate(ROWS) if (c,) + r not in kn and len({t[j] for t in cons[c]}) > 1]
                for c in "ABC"}

    def veto_reason(self, c, a, b, x, cap):
        kn = self.known()
        if (c, a, b, x) in kn:
            return f"You already asked {c}({a}, {b}, {x}) and got {kn[(c, a, b, x)]}."
        cons = self.consistent()
        if cons:
            op = self.open_rows(cons)
            if (a, b, x) not in op[c]:
                v = next(iter(cons[c]))[ROWS.index((a, b, x))]
                still = "; ".join(f"{o}: " + ", ".join(f"{o}({p}, {q}, {r})" for p, q, r in op[o])
                                  for o in "ABC" if op[o])
                return (f"The rules and the answers so far already force {c}({a}, {b}, {x}) = {v}, so that query "
                        f"gives no information. Rows still uncertain -- {still}.")
            others = [o for o in "ABC" if o != c and op[o] and sum(1 for q in kn if q[0] == o) < cap]
        else:
            others = [o for o in "ABC" if o != c and sum(1 for q in kn if q[0] == o) < cap]
        if sum(1 for q in kn if q[0] == c) >= cap and others:
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
            cons = self.consistent()
            if cons and not any(self.open_rows(cons).values()):
                break             # every row is forced: further queries carry no information
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
        self._cons = self.consistent()
        final = ((f"{last_line}\n" if last_line else "") +
                 "All observed outputs ('?' = not queried; deduce those from the circuit rules and your "
                 f"inventory):\n{self.table_text()}\n" + self.deduced_text() + env.last_turn_prompt)
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
        if not out and self._cons:        # never return a malformed guess when the rules leave candidates
            out = "<guess>" + "".join("".join(map(str, sorted(self._cons[c])[0])) for c in "ABC") + "</guess>"
        return out or reply

    def assemble(self, reply):
        """Observed bits for asked rows; the model's bits for the rest -- from a 24-bit guess if it
        wrote one, else from its per-circuit 8-digit table ('Circuit A: 00010101' -- the 7B drops digits
        when concatenating), else from its explicit 'X(a,b,c) = v' row statements. None if any row is missing."""
        m = GUESS_RE.search(reply or "")
        bits = re.sub(r"\s", "", m.group(1)) if m else None
        if not bits or len(bits) != 24:
            found = re.findall(r"[01]{24}", reply or "")
            bits = found[-1] if found else None
        per = {}
        for c in "ABC":
            lines = re.findall(rf"\b{c}\b[^\n01]{{0,12}}([01]{{8}})(?![01])", reply or "")
            if lines:
                per[c] = lines[-1]
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
                elif c in per:
                    out.append(per[c][j])
                elif q in stated:
                    out.append(stated[q])
                else:
                    return None
        cons = getattr(self, "_cons", {})
        if cons:
            for i, c in enumerate("ABC"):
                said = tuple(int(ch) for ch in out[i * 8:i * 8 + 8])
                best = min(sorted(cons[c]), key=lambda t: sum(u != v for u, v in zip(t, said)))
                out[i * 8:i * 8 + 8] = [str(v) for v in best]
        return f"<guess>{''.join(out)}</guess>"
