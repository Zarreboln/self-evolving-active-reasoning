"""Fair-share ledger harness (parent: stateless ledger). Hypothesis: the episodes fail on ALLOCATION,
not memory. 18 queries cannot cover 3 x 8 = 24 rows, and the 7B always scans rows in order: A gets
8, B gets 7-8, C gets 2-3, so the third circuit is a near-blind guess while A and B waste rows that
the task rules already pin down.

Query phase, on top of the parent's ledger: each circuit gets an equal share of the budget
(ceil(budget / 3) queries). A reply that queries a circuit whose share is used up is re-asked, and
the ledger stops listing that circuit's unknown rows as options.

Final phase: with 6 rows per circuit the 7B fills the missing bits badly (non-monotone fills with
AND/OR-only inventories, constant circuits, contradicted rows). The rules say each circuit is a
read-once formula: every input consumed exactly once, exactly two 2-input gates, any NOTs. So the
model is asked, per circuit, for a FORMULA; the harness evaluates it on all 8 rows, and re-asks
(with the mismatches listed) if it breaks those rules or disagrees with an observed row. The
circuit's 8 bits are the model's formula evaluated; if no formula passes, the model's own bit line.
"""

import math
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"[01]{24}")
LINE_RE = {c: re.compile(rf"\b{c}\s*:\s*([01]{{8}})\b") for c in "ABC"}
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
MAX_REASKS = 4
MAX_FORMULA_TRIES = 6
VAR = {"x1": 0, "x2": 1, "x3": 2, "input1": 0, "input2": 1, "input3": 2, "in1": 0, "in2": 1, "in3": 2}
TOK_RE = re.compile(r"\s*(input[123]|in[123]|x[123]|AND|OR|NOT|&&|\|\||[&|~!()¬∧∨'])", re.I)
FORMULA_RE = re.compile(r"<formula>(.*?)</formula>", re.S | re.I)


def parse_formula(text):
    """Read-once check + evaluator. Returns (table 8-char str, problem str or None)."""
    toks, pos, text = [], 0, re.sub(r"^\s*[A-C]\s*(\(\s*x1\s*,\s*x2\s*,\s*x3\s*\))?\s*=", "", text).strip().rstrip(".")
    while pos < len(text):
        m = TOK_RE.match(text, pos)
        if not m:
            return None, f"could not read '{text[pos:pos + 8]}'"
        toks.append(m.group(1).lower())
        pos = m.end()
    i, uses, gates = 0, [0, 0, 0], [0]

    def atom():
        nonlocal i
        if i >= len(toks):
            raise ValueError("formula ends too early")
        t = toks[i]
        i += 1
        if t in ("not", "~", "!", "¬"):
            f = atom()
            return lambda r: 1 - f(r)
        if t == "(":
            f = expr()
            if i >= len(toks) or toks[i] != ")":
                raise ValueError("missing ')'")
            i += 1
            return postfix(f)
        if t in VAR:
            uses[VAR[t]] += 1
            v = VAR[t]
            return postfix(lambda r: r[v])
        raise ValueError(f"unexpected '{t}'")

    def postfix(f):            # x1' = NOT x1
        nonlocal i
        while i < len(toks) and toks[i] == "'":
            i += 1
            f = (lambda g: lambda r: 1 - g(r))(f)
        return f

    def chain(sub, ops, combine):
        nonlocal i
        f = sub()
        while i < len(toks) and toks[i] in ops:
            i += 1
            g = sub()
            gates[0] += 1
            f = (lambda p, q: lambda r: combine(p(r), q(r)))(f, g)
        return f

    def conj():                # AND binds tighter than OR
        return chain(atom, ("and", "&", "&&", "∧"), lambda u, v: u & v)

    def expr():
        return chain(conj, ("or", "|", "||", "∨"), lambda u, v: u | v)

    try:
        f = expr()
        if i != len(toks):
            raise ValueError(f"unexpected '{toks[i]}'")
    except ValueError as e:
        return None, str(e)
    table = "".join(str(f(r)) for r in ROWS)
    if uses != [1, 1, 1]:
        bad = [f"x{k + 1} {n} times" for k, n in enumerate(uses) if n != 1]
        return table, "every input must be used exactly once, but it uses " + ", ".join(bad)
    if gates[0] != 2:
        return table, f"it has {gates[0]} AND/OR gates; each circuit has exactly 2"
    if table in ("0" * 8, "1" * 8):
        return table, "it is constant, which the rules forbid"
    return table, None


class Harness(CDHarness):
    def share(self):
        return math.ceil(self.env.budget / 3)

    def used(self):
        n = {c: 0 for c in "ABC"}
        for (c, _, _, _), _ in self.env.history:
            n[c] += 1
        return n

    def open_circuits(self):
        used, known = self.used(), {q for q, _ in self.env.history}
        return [c for c in "ABC" if used[c] < self.share()
                and any((c, a, b, x) not in known for a, b, x in ROWS)]

    def ledger(self, querying=True):
        known = {q: bit for q, bit in self.env.history}
        used, share = self.used(), self.share()
        done = [self.env.format_answer(c, a, b, x, known[(c, a, b, x)])
                for c in "ABC" for a, b, x in ROWS if (c, a, b, x) in known]
        lines = [f"You are in the MIDDLE of the game. You have ALREADY made {len(self.env.history)} "
                 f"queries; their answers are recorded here, so do not start over."]
        lines.append("Answers already received:" + ("\n- " + "\n- ".join(done) if done else " none yet."))
        lines.append("Tables so far (rows 000..111, '?' = not queried yet):")
        for c in "ABC":
            lines.append(f"  {c}: " + "".join(str(known.get((c, a, b, x), "?")) for a, b, x in ROWS))
        if not querying:
            return "\n".join(lines)
        lines.append(f"There are {self.env.budget} queries for 24 rows, so each circuit gets at most "
                     f"{share} queries; the rows you do not query are deduced from the task rules at the end.")
        lines.append("Queries per circuit: " + ", ".join(f"{c} {used[c]}/{share}" for c in "ABC") + ".")
        opened = self.open_circuits()
        lines.append("Rows you may query now:")
        for c in "ABC":
            if c in opened:
                todo = [f"{c}({a}, {b}, {x})" for a, b, x in ROWS if (c, a, b, x) not in known]
                lines.append(f"  {c}: " + ", ".join(todo))
            else:
                lines.append(f"  {c}: none ({c} has used its {share} queries)")
        return "\n".join(lines)

    def ask(self, text):
        return self.llm([{"role": "system", "content": self.env.system_prompt},
                         {"role": "user", "content": text}])

    def solve(self):
        env = self.env
        while env.remaining > 0:
            known = {q for q, _ in env.history}
            opened = self.open_circuits()
            if not opened:
                break
            prompt = (f"{self.ledger()}\nQueries remaining: {env.remaining} of {env.budget}.\n"
                      "Pick ONE row from 'Rows you may query now' and reply with just that query, "
                      f"e.g. <query>{opened[-1]}(1, 0, 1)</query>.")
            q, fallback, note = None, None, ""
            for k in range(1 + MAX_REASKS):
                reply = self.ask(prompt + note) or ""
                m = QUERY_RE.search(reply)
                if not m:
                    note = f"\n(Reply {k + 1} had no <query>X(a, b, c)</query>. Send exactly one.)"
                    continue
                cand = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                label = f"{cand[0]}({cand[1]}, {cand[2]}, {cand[3]})"
                if cand in known:
                    note = (f"\n(Reply {k + 1} asked {label}, which is already answered above. "
                            "Choose a row from 'Rows you may query now'.)")
                    continue
                if cand[0] not in opened:
                    fallback = fallback or cand
                    note = (f"\n(Reply {k + 1} asked {label}, but circuit {cand[0]} has used its "
                            f"{self.share()} queries. Query circuit {' or '.join(opened)} instead.)")
                    continue
                q = cand
                break
            q = q or fallback      # the model would not leave a full circuit: take its own query anyway
            if q is None:
                break
            try:
                env.query(*q)
            except Exception:
                break
        final = (f"{self.ledger(querying=False)}\nAll queries are used; no more queries.\n"
                 "Fill in every '?' using the task rules (gate inventory, how circuits are built). "
                 "Rows already answered must keep their observed value. Be brief: write one line per "
                 "circuit as 'A: <8 bits>', 'B: <8 bits>', 'C: <8 bits>', then the final answer.\n"
                 f"{env.last_turn_prompt}")
        reply, note, lines = "", "", None
        for k in range(1 + MAX_REASKS):
            reply = self.ask(final + note) or ""
            found = [LINE_RE[c].findall(reply) for c in "ABC"]
            if all(found):
                lines = [l[-1] for l in found]
                break
            if GUESS_RE.search(reply.replace(" ", "")):
                break
            note = f"\n(Reply {k + 1} had no 24-bit <guess>. Keep it short and end with <guess>...</guess>.)"
        if lines is None:
            m = GUESS_RE.findall(reply.replace(" ", ""))
            lines = [m[-1][8 * j:8 * j + 8] for j in range(3)] if m else None
        tables = [self.formula_table(c) for c in "ABC"]
        if lines is None and not all(tables):
            return reply
        out = [t or lines[j] for j, t in enumerate(tables)]
        return f"{reply}\n<guess>{''.join(out)}</guess>"

    def formula_table(self, c):
        """Ask the model for circuit c as a read-once formula; return its 8-row table or None."""
        known = {q: bit for q, bit in self.env.history}
        obs = [(r, known[(c,) + r]) for r in ROWS if (c,) + r in known]
        rows = "\n".join(f"- {c}({a}, {b}, {x}) = {bit}" for (a, b, x), bit in obs)
        prompt = (f"All queries are used. Now describe circuit {c} as a formula.\n"
                  f"Answers received for {c} (inputs x1, x2, x3):\n{rows}\n"
                  "By the task rules, each circuit uses every input exactly once and exactly two "
                  "2-input gates (AND / OR), plus NOT gates if it has any (see the inventory). "
                  f"Write a formula for {c} over x1, x2, x3 with AND, OR, NOT and parentheses that "
                  "matches EVERY answer above. Check each answer, then end with the formula, e.g. "
                  "<formula>(x1 AND NOT x2) OR x3</formula>.")
        note = ""
        for k in range(MAX_FORMULA_TRIES):
            reply = self.ask(prompt + note) or ""
            m = FORMULA_RE.findall(reply)
            if not m:
                note = f"\n(Attempt {k + 1} had no <formula>...</formula>. End with exactly one.)"
                continue
            table, problem = parse_formula(m[-1])
            if problem is None:
                wrong = [f"{c}({a}, {b}, {x}) would be {table[ROWS.index((a, b, x))]} but the answer was {bit}"
                         for (a, b, x), bit in obs if table[ROWS.index((a, b, x))] != str(bit)]
                if not wrong:
                    return table
                problem = "it does not match the answers: " + "; ".join(wrong)
            note = (f"\n(Attempt {k + 1}: <formula>{m[-1].strip()}</formula> is not possible because "
                    f"{problem}. Try a different formula.)")
        return None
