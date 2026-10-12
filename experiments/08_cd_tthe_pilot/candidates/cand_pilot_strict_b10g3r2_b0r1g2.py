"""Candidate-tracking harness (parent: bare).

The rules in the system prompt pin down the hypothesis space: every circuit starts from 3 inputs
and must end with 1 signal, so each circuit holds exactly two 2-input gates (AND/OR) plus any
number of NOTs, and the inventory is split across A, B, C without replacement. The harness
enumerates every circuit that the rules allow, keeps only those consistent with the answers so far,
and each turn shows the model a compact state: answers, which circuits are already pinned down,
the remaining candidate tables, and which inputs would still split them. The model writes every
query; replies that are malformed, repeat a query, or cannot split any remaining candidates are
re-asked. Querying stops once every circuit has a single candidate. The final 24 bits are assembled
from the model's per-circuit picks among the still-consistent candidates.

Child changes (b0r1g2): queries are ranked by how evenly they split the JOINT space (A, B, C
together, coupled through the inventory split -- e.g. finding the NOT in A rules it out for B and C),
not each circuit on its own; the model is shown only the top 3 and picks one. A reply that never
names an informative query costs nothing: no fallback query is spent, the model is simply asked
again. Candidates are ordered by joint weight, so the final pick defaults to the most likely table.
"""

import itertools
import math
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
LABELS = "ABC"
FULL = 0xFF


def _input_mask(i):
    # bit r of the mask = value of input i on row r (row r = 4*x1 + 2*x2 + x3, the benchmark order)
    return sum(1 << r for r in range(8) if (r >> (2 - i)) & 1)


def _bits(mask):
    return "".join(str((mask >> r) & 1) for r in range(8))


def _enumerate(n_and, n_or, n_not):
    """All output tables reachable by the rules with exactly this gate multiset -> {mask: expr}."""
    out = {}
    seen = set()

    def rec(sig, a, o, n):
        key = (tuple(sorted(s[0] for s in sig)), a, o, n)
        if key in seen:
            return
        seen.add(key)
        if len(sig) == 1 and a == 0 and o == 0 and n == 0:
            m, e = sig[0]
            if m not in out:
                out[m] = e
            return
        if n > 0:
            for i in range(len(sig)):
                m, e = sig[i]
                rec(sig[:i] + [(FULL ^ m, f"NOT {e}" if len(e) <= 2 else f"NOT({e})")] + sig[i + 1:], a, o, n - 1)
        if len(sig) >= 2:
            for i, j in itertools.combinations(range(len(sig)), 2):
                rest = [s for k, s in enumerate(sig) if k not in (i, j)]
                (mi, ei), (mj, ej) = sig[i], sig[j]
                if a > 0:
                    rec(rest + [(mi & mj, f"({ei} AND {ej})")], a - 1, o, n)
                if o > 0:
                    rec(rest + [(mi | mj, f"({ei} OR {ej})")], a, o - 1, n)

    rec([(_input_mask(i), f"x{i + 1}") for i in range(3)], n_and, n_or, n_not)
    return {m: (e[1:-1] if e.startswith("(") and e.endswith(")") else e)
            for m, e in out.items() if m not in (0, FULL)}


class Space:
    """Joint candidates over (A, B, C), coupled only through how the inventory is split."""

    def __init__(self, inventory):
        inv = {str(k).upper(): int(v) for k, v in inventory.items()}
        n_and, n_or, n_not = inv.get("AND", 0), inv.get("OR", 0), inv.get("NOT", 0)
        self.allocs = []          # [(per-circuit option tuple)]
        self.tables = {}          # option (a, o, n) -> {mask: expr}
        per = [(a, 2 - a, n) for a in range(3) for n in range(n_not + 1)]
        for combo in itertools.product(per, repeat=3):
            if (sum(c[0] for c in combo) == n_and and sum(c[1] for c in combo) == n_or
                    and sum(c[2] for c in combo) == n_not):
                self.allocs.append(combo)
        for combo in self.allocs:
            for opt in combo:
                if opt not in self.tables:
                    self.tables[opt] = _enumerate(*opt)
        self.relaxed = not self.allocs
        self.obs = {c: {} for c in LABELS}

    def observe(self, c, row, bit):
        self.obs[c][row] = bit

    def _fits(self, c, m):
        return all(((m >> r) & 1) == b for r, b in self.obs[c].items())

    def joint(self):
        """-> [per-allocation list of 3 {mask: expr} dicts], each a block of the joint space
        (the product of its three sets). Empty blocks are dropped."""
        blocks = []
        for combo in self.allocs:
            sets = [{m: e for m, e in self.tables[combo[k]].items() if self._fits(c, m)}
                    for k, c in enumerate(LABELS)]
            if all(sets):
                blocks.append(sets)
        if not blocks:
            # the answers contradict the rules as read here: fall back to "any non-constant
            # table consistent with the answers" so nothing true is ever excluded
            self.relaxed = True
            blocks = [[{m: "?" for m in range(1, FULL) if self._fits(c, m)} for c in LABELS]]
        return blocks

    def candidates(self, blocks=None):
        """-> {circuit: {mask: (expr, weight)}} marginal candidates; weight = number of joint
        (A, B, C) completions that use this table."""
        res = {c: {} for c in LABELS}
        for sets in blocks or self.joint():
            size = math.prod(len(s) for s in sets)
            for k, c in enumerate(LABELS):
                for m, e in sets[k].items():
                    ex, w = res[c].get(m, (e, 0))
                    res[c][m] = (ex, w + size // len(sets[k]))
        return res


def _row(a, b, x):
    return 4 * a + 2 * b + x


def _inp(r):
    return f"({(r >> 2) & 1},{(r >> 1) & 1},{r & 1})"


class Harness(CDHarness):
    MAX_TRIES = 4
    MENU = 3                  # queries shown to the model: the best joint splits only

    def _splits(self, cands):
        """Informative queries: (circuit, row, n1, n0, entropy) for every split of remaining candidates.
        n1/n0 count candidate tables of that circuit; the entropy ranking uses joint weights, so a
        query that also narrows the other circuits (through the inventory split) ranks higher."""
        out = []
        for c in LABELS:
            cs = cands[c]
            if len(cs) <= 1:
                continue
            tot = sum(w for _, w in cs.values())
            for r in range(8):
                n1 = sum(1 for m in cs if (m >> r) & 1)
                n0 = len(cs) - n1
                if n1 and n0:
                    p = sum(w for m, (_, w) in cs.items() if (m >> r) & 1) / tot
                    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
                    out.append((c, r, n1, n0, h))
        out.sort(key=lambda t: (-round(t[4], 9), LABELS.index(t[0]), t[1]))
        return out

    def _state_text(self, cands, show_menu=True):
        env = self.env
        lines = ["Answers so far:" if env.history else "No queries made yet."]
        for (c, a, b, x), bit in env.history:
            lines.append("  " + env.format_answer(c, a, b, x, bit))
        lines.append("")
        lines.append("What the rules allow (each circuit turns 3 inputs into 1 output, so it has exactly two "
                     "2-input gates plus possibly NOT gates; inputs are x1, x2, x3). Tables list outputs for "
                     "inputs 000,001,010,011,100,101,110,111:")
        for c in LABELS:
            cs = cands[c]
            if len(cs) == 1:
                m, (e, _) = next(iter(cs.items()))
                lines.append(f"  Circuit {c}: SOLVED, table {_bits(m)}  [{e}]. Do NOT query {c} again.")
            elif len(cs) <= 8:
                lines.append(f"  Circuit {c}: {len(cs)} candidates left:")
                for m, (e, _) in sorted(cs.items(), key=lambda t: -t[1][1]):
                    lines.append(f"      {_bits(m)}  [{e}]")
            else:
                lines.append(f"  Circuit {c}: {len(cs)} candidates left (too many to list).")
        if show_menu:
            sp = self._splits(cands)
            if sp:
                lines.append("")
                lines.append(f"The {min(len(sp), self.MENU)} most informative queries now (they split the remaining "
                             "possibilities for all three circuits most evenly):")
                for c, r, n1, n0, _ in sp[:self.MENU]:
                    lines.append(f"  {c}{_inp(r)}: {n1} candidates say 1, {n0} say 0")
        return "\n".join(lines)

    def _ask_query(self, cands, splits):
        env = self.env
        menu = {(c, r) for c, r, *_ in splits[:self.MENU]}
        useful = {(c, r) for c, r, *_ in splits}
        asked = {(q[0], _row(*q[1:])) for q, _ in env.history}
        base = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": self._state_text(cands) +
                 f"\n\nYou have {env.remaining} queries left. Choose ONE of the queries listed above. "
                 "Reply with one short sentence of reasoning, then exactly one <query>X(a, b, c)</query>."}]
        msgs = list(base)
        for t in range(self.MAX_TRIES):
            reply = self.llm(msgs, temperature=0.3 if t == 0 else 0.8, max_tokens=300)
            m = QUERY_RE.search(reply or "")
            if not m:
                fb = "Your reply did not contain a query in the required format."
            else:
                c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
                r = _row(a, b, x)
                # a listed query, or on the last try any query that still tells candidates apart
                if (c, r) in menu or ((c, r) in useful and t == self.MAX_TRIES - 1):
                    return c, a, b, x
                if (c, r) in asked:
                    fb = f"You already asked {c}({a}, {b}, {x}); the answer is in the list above."
                elif len(cands[c]) == 1:
                    fb = f"Circuit {c} is already solved; querying it wastes a query."
                elif (c, r) in useful:
                    fb = f"{c}({a}, {b}, {x}) tells you less than the listed queries."
                else:
                    fb = (f"Every remaining candidate for circuit {c} gives the same output on ({a}, {b}, {x}), "
                          "so that query gives no new information.")
            msgs = base + [{"role": "assistant", "content": reply or ""},
                           {"role": "user", "content": fb + " Copy one of these queries: " +
                            ", ".join(f"{c}{_inp(r)}" for c, r, *_ in splits[:self.MENU]) +
                            ". Reply with exactly one <query>X(a, b, c)</query>."}]
        return None               # no informative query from the model: spend nothing, ask afresh

    def _final(self, cands):
        env = self.env
        order = {c: [m for m, _ in sorted(cands[c].items(), key=lambda t: -t[1][1])] for c in LABELS}
        lines = [self._state_text(cands, show_menu=False), "",
                 "The querying phase is over. For each circuit, choose the candidate table that is "
                 "consistent with the answers. Numbered candidates (if a circuit has several left, they are "
                 "listed most likely first):"]
        for c in LABELS:
            for k, m in enumerate(order[c][:12], 1):
                lines.append(f"  {c}{k}: {_bits(m)}  [{cands[c][m][0]}]")
            if len(order[c]) > 12:
                lines.append(f"  ({len(order[c]) - 12} more candidates for {c} not shown)")
        lines.append("")
        lines.append("Reply with your choice for each circuit in the form 'A: <number>, B: <number>, C: <number>', "
                     "then the final answer: the 8 output bits of A, then B, then C (24 bits in total) "
                     "inside <guess> </guess>.")
        base = [{"role": "system", "content": env.system_prompt}, {"role": "user", "content": "\n".join(lines)}]
        picks = {}
        last_reply = ""
        msgs = list(base)
        for t in range(self.MAX_TRIES):
            reply = self.llm(msgs, temperature=0.2 if t == 0 else 0.7, max_tokens=500) or ""
            last_reply = reply
            for c in LABELS:
                if c in picks:
                    continue
                m = re.search(rf"\b{c}\s*[:=]\s*{c}?\s*(\d+)", reply)
                if m and 1 <= int(m.group(1)) <= min(12, len(order[c])):
                    picks[c] = order[c][int(m.group(1)) - 1]
            g = re.findall(r"<guess>\s*([01]{24})\s*</guess>", reply)
            if g:
                bits = g[-1]
                for k, c in enumerate(LABELS):
                    chunk = bits[8 * k:8 * k + 8]
                    mask = sum(int(ch) << r for r, ch in enumerate(chunk))
                    if mask in cands[c] and c not in picks:
                        picks[c] = mask
            if all(c in picks for c in LABELS):
                break
            missing = [c for c in LABELS if c not in picks]
            msgs = base + [{"role": "assistant", "content": reply},
                           {"role": "user", "content": "I could not read a valid candidate number for circuit(s) "
                            + ", ".join(missing) + ". Reply with 'A: <number>, B: <number>, C: <number>' using "
                            "the numbered candidates above."}]
        for c in LABELS:
            if c not in picks:
                # closest candidate to whatever the model last wrote for this circuit
                g = re.findall(r"[01]{24}", last_reply)
                if g:
                    chunk = g[-1][8 * LABELS.index(c):8 * LABELS.index(c) + 8]
                    picks[c] = min(order[c], key=lambda mm: sum(ch != b for ch, b in zip(_bits(mm), chunk)))
                else:
                    picks[c] = order[c][0]
        return "<guess>" + "".join(_bits(picks[c]) for c in LABELS) + "</guess>"

    def solve(self):
        env = self.env
        space = Space(env.inventory)
        wasted = 0
        while env.remaining > 0 and wasted < env.budget:
            cands = space.candidates()
            splits = self._splits(cands)
            if not splits:
                break                 # every circuit is pinned down: nothing left to learn
            q = self._ask_query(cands, splits)
            if q is None:
                wasted += 1
                continue
            c, a, b, x = q
            bit = env.query(c, a, b, x)
            space.observe(c, _row(a, b, x), bit)
        return self._final(space.candidates())
