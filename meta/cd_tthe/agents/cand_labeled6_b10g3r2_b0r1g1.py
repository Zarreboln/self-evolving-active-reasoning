"""Formula-commitment harness: the model names each circuit's wiring, the harness computes it.

Hypothesis (from the parent's traces): the ledger fixed bookkeeping -- no guess contradicted an
observed row -- yet every circuit the model queried on all 8 rows was right and every circuit it
left partially queried was wrong. The 7B cannot extrapolate a table row by row; it fills '?'
rows by pattern-guessing. It can, however, write a small gate formula, and the task rules fix what
AND / OR / NOT compute. So:
  * query phase = the parent's ledger, plus a per-circuit share (budget // 3 rows): a query for a
    circuit that already holds its share is vetoed (free re-ask) while another circuit is below it,
    so no circuit is left with 2-3 rows;
  * final phase, per circuit: the model must write `X = <formula over a, b, c>` with exactly two
    AND/OR gates, each input once, NOTs anywhere. The harness computes the formula on all 8 rows;
    a formula that misreads the rules or contradicts an observed row is vetoed, and the model is
    re-asked with the rows where its formula and the answers disagree. The unqueried rows are the
    accepted formula's outputs (observed rows stay locked).
The model still writes every query and every formula; the harness only computes and vetoes."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
ROW_RE = re.compile(r"([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*(?:=|:|->|is)\s*\**\s*([01])\b")
TOKEN_RE = re.compile(r"\s*(AND|OR|NOT|&&|\|\||[abc()&|~!¬∧∨·+'])", re.I)
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
MAX_RETRIES = 4
FORMULA_TRIES = 8


class FormulaError(Exception):
    pass


def parse_formula(text):
    """AND/OR/NOT formula over a, b, c -> (function row -> bit, gate and input counts)."""
    toks, pos, text = [], 0, text.strip().rstrip(".;`*")
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise FormulaError(f"cannot read '{text[pos:pos + 12].strip()}'")
        t = m.group(1)
        t = {"&&": "AND", "&": "AND", "∧": "AND", "·": "AND", "||": "OR", "|": "OR", "∨": "OR",
             "+": "OR", "~": "NOT", "!": "NOT", "¬": "NOT"}.get(t, t)
        toks.append(t.upper() if t.upper() in ("AND", "OR", "NOT") else t.lower())
        pos = m.end()
    stats = {"AND": 0, "OR": 0, "a": 0, "b": 0, "c": 0}
    i = 0

    def peek():
        return toks[i] if i < len(toks) else None

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def factor():
        t = peek()
        if t == "NOT":
            take()
            f = factor()
            node = lambda r, f=f: 1 - f(r)
        elif t == "(":
            take()
            node = expr()
            if peek() != ")":
                raise FormulaError("unbalanced parentheses")
            take()
        elif t in ("a", "b", "c"):
            take()
            stats[t] += 1
            k = "abc".index(t)
            node = lambda r, k=k: r[k]
        else:
            raise FormulaError(f"unexpected '{t}'" if t else "formula ends too early")
        while peek() == "'":                     # postfix negation a'
            take()
            node = lambda r, f=node: 1 - f(r)
        return node

    def term():
        node = factor()
        while peek() == "AND":
            take()
            stats["AND"] += 1
            rhs = factor()
            node = lambda r, x=node, y=rhs: x(r) & y(r)
        return node

    def expr():
        node = term()
        while peek() == "OR":
            take()
            stats["OR"] += 1
            rhs = term()
            node = lambda r, x=node, y=rhs: x(r) | y(r)
        return node

    if not toks:
        raise FormulaError("empty formula")
    f = expr()
    if i != len(toks):
        raise FormulaError(f"unexpected '{toks[i]}'")
    return f, stats


class Harness(CDHarness):
    def _known(self):
        return {q: bit for q, bit in self.env.history}

    def _table(self, known, circuits="ABC"):
        lines = []
        for c in circuits:
            n = sum(1 for r in ROWS if (c,) + r in known)
            lines.append(f"Circuit {c} ({n} of 8 rows known):")
            for r in ROWS:
                v = known.get((c,) + r)
                lines.append(f"  {c}({r[0]}, {r[1]}, {r[2]}) = {'?' if v is None else v}")
        return "\n".join(lines)

    def _counts(self, known):
        return {c: sum(1 for r in ROWS if (c,) + r in known) for c in "ABC"}

    def _turn_prompt(self, known, note=""):
        env = self.env
        unknown = {c: 8 - n for c, n in self._counts(known).items()}
        last = ""
        if env.history:
            (c, a, b, x), bit = env.history[-1]
            last = f"Last answer: {env.format_answer(c, a, b, x, bit)}\n\n"
        return (f"{last}Everything observed so far ('?' = not yet queried):\n{self._table(known)}\n\n"
                f"Queries remaining: {env.remaining}. Unknown rows: A {unknown['A']}, B {unknown['B']}, "
                f"C {unknown['C']} ({sum(unknown.values())} in total) -- you cannot query them all, so "
                f"each circuit gets at most {env.budget // 3} queries; use the circuit-building rules and "
                "the gate inventory to infer the rows you do not query. Querying a row that is already "
                "known gives no new information.\n"
                f"{note}Think briefly, then send exactly one query for a '?' row as <query>X(a, b, c)</query>.")

    def solve(self):
        env = self.env
        share = max(1, env.budget // 3)
        sys_msg = {"role": "system", "content": env.system_prompt}
        while env.remaining > 0:
            known = self._known()
            counts = self._counts(known)
            note, pick = "", None
            for _ in range(MAX_RETRIES):
                reply = self.llm([sys_msg, {"role": "user", "content": self._turn_prompt(known, note)}])
                m = QUERY_RE.search(reply or "")
                if not m:
                    note = "(Your previous reply had no query in the required format.) "
                    continue
                q = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                if q in known:
                    note = (f"(You asked for {q[0]}({q[1]}, {q[2]}, {q[3]}), but it is already known: "
                            f"{known[q]}. Pick a row marked '?'.) ")
                    continue
                pick = q
                under = [c for c in "ABC" if counts[c] < share]
                if counts[q[0]] >= share and under:
                    note = (f"(Circuit {q[0]} already has its {share} queries; query circuit "
                            f"{' or '.join(under)} instead.) ")
                    continue
                break
            if pick is None:
                break                     # the model never produced a new query: stop asking
            try:
                env.query(*pick)
            except Exception:
                break
        return self._final()

    # ---- final phase: one formula per circuit, computed by the harness ----

    def _formula_prompt(self, c, known, feedback):
        rows = "\n".join(f"  {c}({r[0]}, {r[1]}, {r[2]}) = {known.get((c,) + r, '?')}" for r in ROWS)
        inv = self.env.inventory or {}
        inv_txt = ", ".join(f"{v} {k}" for k, v in inv.items())
        return (f"The queries are over. Observed rows of circuit {c} ('?' = never queried):\n{rows}\n\n"
                f"Gate inventory for all three circuits together: {inv_txt}. Circuit {c} takes inputs "
                f"(a, b, c) and uses exactly two two-input gates (AND / OR), each input exactly once, and "
                f"NOT gates anywhere (on an input, between the gates, or on the output).\n"
                f"Write circuit {c} as a formula over a, b, c, using AND, OR, NOT and parentheses, that "
                f"reproduces EVERY observed row above. Examples of the form: `(a AND NOT b) OR c`, "
                f"`NOT (a OR c) AND b`, `NOT ((b OR c) AND NOT a)`.\n{feedback}"
                f"Check your formula on each observed row, then end with one final line exactly of the form\n"
                f"{c} = <formula>")

    def _extract_formula(self, c, reply):
        found = None
        for line in (reply or "").splitlines():
            s = line.strip().strip("`*").strip()
            m = re.match(rf"^(?:circuit\s+)?{c}\s*(?:\(\s*a\s*,\s*b\s*,\s*c\s*\))?\s*=\s*(.+)$", s, re.I)
            if m and not re.match(r"^\s*[01?]\s*$", m.group(1)):
                found = m.group(1).strip().strip("`*$").strip()
        return found

    def _fit_formula(self, c, known):
        sys_msg = {"role": "system", "content": self.env.system_prompt}
        feedback, best = "", None          # best = (mismatches, table)
        for _ in range(FORMULA_TRIES):
            reply = self.llm([sys_msg, {"role": "user", "content": self._formula_prompt(c, known, feedback)}],
                             max_tokens=900)
            text = self._extract_formula(c, reply)
            if text is None:
                feedback = (f"(Your previous reply did not end with a line `{c} = <formula>`.)\n")
                continue
            try:
                f, st = parse_formula(text)
            except FormulaError as e:
                feedback = (f"(Your previous formula `{text}` could not be read: {e}. Use only a, b, c, "
                            f"AND, OR, NOT and parentheses.)\n")
                continue
            if st["AND"] + st["OR"] != 2 or any(st[v] != 1 for v in "abc"):
                feedback = (f"(Your previous formula `{text}` uses {st['AND'] + st['OR']} AND/OR gates and "
                            f"inputs a x{st['a']}, b x{st['b']}, c x{st['c']}; it must use exactly two "
                            f"AND/OR gates and each of a, b, c exactly once.)\n")
                continue
            table = [f(r) for r in ROWS]
            if len(set(table)) == 1:
                feedback = f"(Your previous formula `{text}` is constant; a circuit is never constant.)\n"
                continue
            bad = [r for r in ROWS if (c,) + r in known and table[ROWS.index(r)] != known[(c,) + r]]
            if best is None or len(bad) < best[0]:
                best = (len(bad), table)
            if not bad:
                return table
            feedback = (f"(Your previous formula `{text}` is wrong on observed rows: " +
                        "; ".join(f"it gives {c}({r[0]}, {r[1]}, {r[2]}) = {table[ROWS.index(r)]} but the "
                                  f"answer was {known[(c,) + r]}" for r in bad) +
                        ". Try a different wiring, different gates or different NOT placement.)\n")
        return best[1] if best else None

    def _fill_rows(self, c, known, missing):
        """The parent's fallback: the model states the '?' rows directly."""
        sys_msg = {"role": "system", "content": self.env.system_prompt}
        filled, note = {}, ""
        for _ in range(MAX_RETRIES):
            todo = [r for r in missing if r not in filled]
            if not todo:
                break
            ask = "\n".join(f"  {c}({r[0]}, {r[1]}, {r[2]}) = ?" for r in todo)
            prompt = (f"The queries are over. Everything observed ('?' = never queried):\n{self._table(known)}\n\n"
                      f"Now complete circuit {c}. Give its output on these rows:\n{ask}\n{note}"
                      f"Think briefly, then end with one line per row above, exactly in the form "
                      f"{c}(a, b, c) = 0 or {c}(a, b, c) = 1.")
            reply = self.llm([sys_msg, {"role": "user", "content": prompt}])
            for m in ROW_RE.finditer(reply or ""):
                r = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                if m.group(1).upper() == c and r in todo:
                    filled[r] = int(m.group(5))
            note = "(Your previous reply did not state every row in the required form.)\n"
        return filled

    def _final(self):
        known = self._known()
        bits = {}
        for c in "ABC":
            missing = [r for r in ROWS if (c,) + r not in known]
            filled = {}
            if missing:
                table = self._fit_formula(c, known)
                if table is not None:
                    filled = {r: table[ROWS.index(r)] for r in missing}
                else:
                    filled = self._fill_rows(c, known, missing)
            for r in ROWS:
                v = known.get((c,) + r)
                bits[(c,) + r] = v if v is not None else filled.get(r, 0)
        guess = "".join(str(bits[(c,) + r]) for c in "ABC" for r in ROWS)
        return f"<guess>{guess}</guess>"
