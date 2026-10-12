"""Formula checker: the harness runs the model's own circuit hypotheses and reports counterexamples.

In the ledger traces the model fills unqueried rows from structural claims it never checks (XOR
circuits with only AND/OR gates, formulas that contradict rows on the grid it was just shown), and it
spends its queries reading circuit A row by row, so C ends with 1-3 known rows. Here the model writes
every hypothesis as a formula over x1, x2, x3 with AND(p, q), OR(p, q), NOT(p) -- the gates the rules
allow, each input used once -- and the harness only *evaluates* what it wrote:
  * query turns: the model writes two different formulas for one circuit that both fit its known
    rows, then queries an input where they disagree. A formula that misfits a known row, breaks the
    gate rules, or a query that does not separate the two formulas is sent back with the concrete
    counterexample; after a few failed tries any query on an unknown row is accepted.
  * final: per circuit the model writes one formula; the harness re-asks with counterexamples until it
    fits every observed row and the gate totals over A, B, C match the inventory. The final table is
    the evaluated formula (observed rows always kept); without a fitting formula the model's own bits
    fill the '?' rows, as in the parent.
The harness never proposes a formula, a table, or a query itself.
"""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
FENCED_RE = re.compile(r"```\s*query\s*\n\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)")
FORMULA_RE = re.compile(r"<formula>\s*([A-Ca-c])\s*=\s*(.*?)</formula>", re.S)
BITS_RE = re.compile(r"<bits>\s*([01?](?:\s*[01?]){7})\s*</bits>")
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
TOKEN_RE = re.compile(r"\s*(AND|OR|NOT|x[123]|\(|\)|,)", re.I)
TURN_TRIES = 4
FINAL_TRIES = 5

SYNTAX = ("Write a circuit as a formula over its inputs x1, x2, x3 (so A(1, 0, 1) means x1=1, x2=0, x3=1) "
          "using only AND(p, q), OR(p, q) and NOT(p). Each of x1, x2, x3 appears exactly once (every input "
          "signal is used once), so a formula has exactly two AND/OR gates plus any NOTs, "
          "e.g. OR(AND(x1, NOT(x2)), x3). There is no XOR gate.")


class BadFormula(Exception):
    pass


def parse_formula(text):
    """Formula text -> nested tuples ('AND', l, r) / ('OR', l, r) / ('NOT', p) / ('x', i)."""
    toks, pos, text = [], 0, text.strip().rstrip(".;")
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise BadFormula(f"cannot read '{text[pos:pos + 12].strip()}'")
        toks.append(m.group(1).upper() if m.group(1)[0] != "x" else m.group(1).lower())
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    toks = [t.lower() if t.startswith("X") else t for t in toks]
    i = 0

    def take(expected=None):
        nonlocal i
        if i >= len(toks) or (expected and toks[i] != expected):
            raise BadFormula(f"expected '{expected or 'more'}'")
        i += 1
        return toks[i - 1]

    def node():
        t = take()
        if t in ("x1", "x2", "x3"):
            return ("x", int(t[1]) - 1)
        if t in ("AND", "OR"):
            take("(")
            left = node()
            take(",")
            right = node()
            take(")")
            return (t, left, right)
        if t == "NOT":
            take("(")
            inner = node()
            take(")")
            return ("NOT", inner)
        raise BadFormula(f"unexpected '{t}'")

    tree = node()
    if i != len(toks):
        raise BadFormula("extra text after the formula")
    return tree


def count(tree):
    """{'AND': n, 'OR': n, 'NOT': n, 'inputs': [i, ...]}"""
    if tree[0] == "x":
        return {"AND": 0, "OR": 0, "NOT": 0, "inputs": [tree[1]]}
    subs = [count(t) for t in tree[1:]]
    out = {g: sum(s[g] for s in subs) for g in ("AND", "OR", "NOT")}
    out["inputs"] = [v for s in subs for v in s["inputs"]]
    out[tree[0]] += 1
    return out


def evaluate(tree, row):
    if tree[0] == "x":
        return row[tree[1]]
    if tree[0] == "NOT":
        return 1 - evaluate(tree[1], row)
    u, v = evaluate(tree[1], row), evaluate(tree[2], row)
    return (u & v) if tree[0] == "AND" else (u | v)


def table(tree):
    return "".join(str(evaluate(tree, r)) for r in ROWS)


class Harness(CDHarness):
    def ledger(self):
        seen = {q: bit for q, bit in self.env.history}
        lines = []
        for c in "ABC":
            n = sum(1 for r in ROWS if (c,) + r in seen)
            lines.append(f"Circuit {c} ({n}/8 known):")
            for r in ROWS:
                v = seen.get((c,) + r)
                lines.append(f"  {c}({r[0]}, {r[1]}, {r[2]}) = {'?' if v is None else v}")
        return "\n".join(lines), seen

    def check(self, c, text, seen):
        """(tree, None) if the formula is legal and fits every known row of c, else (tree|None, problem)."""
        try:
            tree = parse_formula(text)
        except BadFormula as e:
            return None, f"'{text.strip()[:80]}' is not a valid formula ({e})."
        k = count(tree)
        if sorted(k["inputs"]) != [0, 1, 2]:
            return tree, f"'{text.strip()}' does not use each of x1, x2, x3 exactly once."
        inv = self.env.inventory
        for g in ("AND", "OR", "NOT"):
            if k[g] > inv.get(g, 0):
                return tree, f"'{text.strip()}' uses {k[g]} {g} gates but the inventory only has {inv.get(g, 0)}."
        tab = table(tree)
        if tab in ("00000000", "11111111"):
            return tree, f"'{text.strip()}' is constant ({tab}); no circuit is constant."
        for r in ROWS:
            v = seen.get((c,) + r)
            if v is not None and int(tab[ROWS.index(r)]) != v:
                return tree, (f"'{text.strip()}' gives {c}({r[0]}, {r[1]}, {r[2]}) = {tab[ROWS.index(r)]}, "
                              f"but the observed answer is {v}. (Its full output column is {tab}.)")
        return tree, None

    def ask_query(self, t, B):
        grid, seen = self.ledger()
        msgs = [{"role": "system", "content": self.env.system_prompt},
                {"role": "user", "content": f"Everything observed so far ('?' = not queried yet):\n{grid}\n\n"
                 f"This is query {t} of {B} ({B - t + 1} remaining, including this one).\n{SYNTAX}\n\n"
                 f"Pick one circuit that still has '?' inputs. Write TWO DIFFERENT formulas for it that both "
                 f"reproduce every known output of that circuit, as\n<formula>A = ...</formula>\n"
                 f"<formula>A = ...</formula>\nThen query a '?' input of that circuit on which the two formulas "
                 f"give different outputs: <query>A(a, b, c)</query>."}]
        last_ok = None
        for attempt in range(TURN_TRIES):
            reply = self.llm(msgs) or ""
            m = QUERY_RE.search(reply) or FENCED_RE.search(reply)
            q = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else None
            if q is not None and q not in seen:
                last_ok = q
            note = self.judge_turn(reply, q, seen)
            if note is None:
                return q
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": note + " Fix this and reply again with two formulas and one "
                      "<query>X(a, b, c)</query>."}]
        return last_ok

    def judge_turn(self, reply, q, seen):
        if q is None:
            return "Your message did not contain a query in the format <query>A(0, 1, 1)</query>."
        if q in seen:
            return f"{self.env.format_answer(*q, seen[q])} is already known (see the grid)."
        fs = [(c.upper(), f) for c, f in FORMULA_RE.findall(reply)]
        fs = [(c, f) for c, f in fs if c == q[0]]
        if len(fs) < 2:
            return (f"You queried circuit {q[0]} but did not give two <formula>{q[0]} = ...</formula> "
                    f"lines for {q[0]}.")
        trees = []
        for c, f in fs[:2]:
            tree, problem = self.check(c, f, seen)
            if problem:
                return "Your formula " + problem
            trees.append(tree)
        t1, t2 = table(trees[0]), table(trees[1])
        unknown = [i for i, r in enumerate(ROWS) if (q[0],) + r not in seen]
        if all(t1[i] == t2[i] for i in unknown):
            return (f"Your two formulas for {q[0]} give the same outputs on every '?' input ({t1}), so they do "
                    f"not compete. Find a genuinely different formula that also fits, or pick another circuit.")
        i = ROWS.index(q[1:])
        if t1[i] == t2[i]:
            diff = ", ".join(f"{q[0]}{ROWS[j]}".replace(" ", "") for j in unknown if t1[j] != t2[j])
            return (f"Both formulas give {q[0]}({q[1]}, {q[2]}, {q[3]}) = {t1[i]}, so that query cannot tell "
                    f"them apart. They differ on: {diff}.")
        return None

    def ask_circuit(self, c, others):
        """Final formula for circuit c -> (tree or None, model bits or None)."""
        grid, seen = self.ledger()
        used = "".join(f"\nYour accepted formula for {o}: {f}" for o, f in others.items())
        msgs = [{"role": "system", "content": self.env.system_prompt},
                {"role": "user", "content": f"The queries are over. Everything observed ('?' = never queried):\n"
                 f"{grid}\n\n{SYNTAX}\nThe three circuits together use exactly the inventory {self.inv_text()}."
                 f"{used}\n\nGive ONE formula for circuit {c} that reproduces every known output of {c}, as "
                 f"<formula>{c} = ...</formula>, then its 8 outputs {c}(0,0,0) ... {c}(1,1,1) as "
                 f"<bits>XXXXXXXX</bits>."}]
        bits = None
        for attempt in range(FINAL_TRIES):
            reply = self.llm(msgs, temperature=0.7 if attempt else 0.3) or ""
            for m in BITS_RE.finditer(reply):
                bits = re.sub(r"\s", "", m.group(1))
            fs = [f for cc, f in FORMULA_RE.findall(reply) if cc.upper() == c]
            if not fs:
                note = f"Your reply had no <formula>{c} = ...</formula>."
            else:
                tree, problem = self.check(c, fs[-1], seen)
                if problem is None:
                    problem = self.total_problem(c, tree, others)
                    if problem is None:
                        return tree, fs[-1].strip(), bits
                note = "Your formula " + problem
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": note + f" Reply again with one <formula>{c} = ...</formula> "
                      f"and <bits>XXXXXXXX</bits>."}]
        return None, None, bits

    def inv_text(self):
        return ", ".join(f"{n} {g}" for g, n in self.env.inventory.items() if n)

    def total_problem(self, c, tree, others):
        """Gate totals over A, B, C must equal the inventory once all three are known."""
        if len(others) < 2:
            return None
        tot = count(tree)
        for f in others.values():
            k = count(parse_formula(f))
            for g in ("AND", "OR", "NOT"):
                tot[g] += k[g]
        inv = self.env.inventory
        bad = [g for g in ("AND", "OR", "NOT") if tot[g] != inv.get(g, 0)]
        if not bad:
            return None
        need = {g: inv.get(g, 0) - (tot[g] - count(tree)[g]) for g in ("AND", "OR", "NOT")}
        return (f"fits the observations, but together with the other two circuits it would use "
                f"{', '.join(f'{tot[g]} {g}' for g in bad)} while the inventory has "
                f"{', '.join(f'{inv.get(g, 0)} {g}' for g in bad)}. Circuit {c} must use exactly "
                f"{need['AND']} AND, {need['OR']} OR, {need['NOT']} NOT.")

    def solve(self):
        env = self.env
        B = env.budget
        for t in range(1, B + 1):
            q = self.ask_query(t, B)
            if q is None:
                continue          # turn spent, as in the benchmark
            env.query(*q)
        _, seen = self.ledger()
        accepted, out = {}, {}
        for c in "ABC":
            tree, text, bits = self.ask_circuit(c, accepted)
            if tree is not None:
                accepted[c] = text
                col = table(tree)
            else:
                col = bits if bits and len(bits) == 8 else "?" * 8
            out[c] = "".join(str(seen[(c,) + r]) if (c,) + r in seen else (col[i] if col[i] in "01" else "0")
                             for i, r in enumerate(ROWS))
        return "<guess>" + "".join(out[c] for c in "ABC") + "</guess>"
