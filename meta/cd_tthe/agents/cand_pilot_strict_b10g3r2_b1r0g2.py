"""Candidate-tracking harness with verification (parent: cand_pilot_strict_b10g3r2_b0r0g2).

The rules in the system prompt pin down the hypothesis space: every circuit starts from 3 inputs
and must end with 1 signal, so each circuit holds exactly two 2-input gates (AND/OR) plus any
number of NOTs, and the inventory is split across A, B, C without replacement. The harness
enumerates every circuit that the rules allow, keeps only those consistent with the answers so far,
and each turn shows the model a compact state: answers, which circuits are already pinned down,
the remaining candidate tables, and which inputs would still split them. The model writes every
query; replies that are malformed, repeat a query, or cannot split any remaining candidates are
re-asked. Querying stops once every circuit has a single candidate. The final 24 bits are assembled
from the model's per-circuit picks among the still-consistent candidates.

Added here: a circuit is often "pinned" only through the inventory split (a handful of its own rows
observed, the rest inferred from what the other circuits must be using), and the parent then stopped
with ~7 queries unused. The leftover budget is now spent on checks the model chooses from a menu:
first rows where the circuit's own rule-allowed tables (ignoring the split) still disagree with the
deduction, then any row of the circuit not yet seen directly. If a check contradicts the deduction,
the candidate space degrades per circuit (rules with the split -> rules without the split -> any
non-constant table) and normal splitting queries resume.
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
        # per-circuit tables ignoring how the inventory is split (any gate types the inventory has)
        self.loose = {}
        for a in range(3):
            if a > n_and or 2 - a > n_or:
                continue
            for n in range(n_not + 1):
                for m, e in (self.tables.get((a, 2 - a, n)) or _enumerate(a, 2 - a, n)).items():
                    self.loose.setdefault(m, e)
        self.relaxed = not self.allocs
        self.obs = {c: {} for c in LABELS}

    def observe(self, c, row, bit):
        self.obs[c][row] = bit

    def _fits(self, c, m):
        return all(((m >> r) & 1) == b for r, b in self.obs[c].items())

    def loose_fits(self, c):
        return {m: e for m, e in self.loose.items() if self._fits(c, m)}

    def candidates(self):
        """-> {circuit: {mask: (expr, weight)}} marginal candidates under the rules + answers."""
        res = {c: {} for c in LABELS}
        alive = False
        for combo in self.allocs:
            sets = []
            for k, c in enumerate(LABELS):
                sets.append({m: e for m, e in self.tables[combo[k]].items() if self._fits(c, m)})
            if all(sets):
                alive = True
                for k, c in enumerate(LABELS):
                    for m, e in sets[k].items():
                        ex, w = res[c].get(m, (e, 0))
                        res[c][m] = (ex, w + 1)
        if not alive:
            # the answers contradict the rules as read here: drop the inventory split first, then
            # (per circuit) fall back to "any non-constant table consistent with the answers"
            self.relaxed = True
            for c in LABELS:
                lf = self.loose_fits(c)
                res[c] = ({m: (e, 1) for m, e in lf.items()} if lf else
                          {m: ("?", 1) for m in range(1, FULL) if self._fits(c, m)})
        return res


def _row(a, b, x):
    return 4 * a + 2 * b + x


def _inp(r):
    return f"({(r >> 2) & 1},{(r >> 1) & 1},{r & 1})"


class Harness(CDHarness):
    MAX_TRIES = 4

    def _splits(self, cands):
        """Informative queries: (circuit, row, n1, n0, entropy) for every split of remaining candidates."""
        out = []
        for c in LABELS:
            cs = cands[c]
            if len(cs) <= 1:
                continue
            for r in range(8):
                n1 = sum(1 for m in cs if (m >> r) & 1)
                n0 = len(cs) - n1
                if n1 and n0:
                    p = n1 / len(cs)
                    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
                    out.append((c, r, n1, n0, h))
        out.sort(key=lambda t: (-t[4], LABELS.index(t[0]), t[1]))
        return out

    def _checks(self, space, cands):
        """Checks of pinned circuits: (circuit, row, n_disagree, 0, score). First the rows where the
        circuit's own rule-allowed tables (ignoring the inventory split) still disagree with the
        deduced table; only when there are none, every row of a pinned circuit not yet observed."""
        out, plain = [], []
        for c in LABELS:
            if len(cands[c]) != 1:
                continue
            pin = next(iter(cands[c]))
            lf = space.loose_fits(c)
            for r in range(8):
                if r in space.obs[c]:
                    continue
                k = sum(1 for m in lf if ((m >> r) & 1) != ((pin >> r) & 1))
                (out if k else plain).append((c, r, k, len(space.obs[c])))
        if out:
            out.sort(key=lambda t: (-t[2], LABELS.index(t[0]), t[1]))
            return out
        plain.sort(key=lambda t: (t[3], LABELS.index(t[0]), t[1]))
        return plain

    def _state_text(self, cands, show_menu=True, checks=None, space=None):
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
                if checks is None:
                    lines.append(f"  Circuit {c}: SOLVED, table {_bits(m)}  [{e}]. Do NOT query {c} again.")
                else:
                    seen = len(space.obs[c])
                    lines.append(f"  Circuit {c}: deduced table {_bits(m)}  [{e}] "
                                 f"({seen} of 8 outputs observed directly, the rest inferred).")
            elif len(cs) <= 8:
                lines.append(f"  Circuit {c}: {len(cs)} candidates left:")
                for m, (e, _) in sorted(cs.items(), key=lambda t: -t[1][1]):
                    lines.append(f"      {_bits(m)}  [{e}]")
            else:
                lines.append(f"  Circuit {c}: {len(cs)} candidates left (too many to list).")
        if checks:
            lines.append("")
            lines.append("Every circuit is deduced, but part of each table is inferred from the rules rather than "
                         "observed. Queries that would CHECK the deduction (the first ones test the most "
                         "inferred outputs; if an answer differs from the deduced table, the deduction was wrong):")
            for c, r, k, _ in checks[:8]:
                pin = next(iter(cands[c]))
                lines.append(f"  {c}{_inp(r)}: deduced output {(pin >> r) & 1}")
            if len(checks) > 8:
                lines.append(f"  ... and {len(checks) - 8} more not shown.")
        elif show_menu:
            sp = self._splits(cands)
            if sp:
                lines.append("")
                lines.append("Queries that can still tell candidates apart (the first ones split them most evenly"
                             + ("; any query not in this list gives no new information):" if len(sp) <= 8 else "):"))
                for c, r, n1, n0, _ in sp[:8]:
                    lines.append(f"  {c}{_inp(r)}: {n1} candidates say 1, {n0} say 0")
                if len(sp) > 8:
                    lines.append(f"  ... and {len(sp) - 8} weaker ones not shown.")
        return "\n".join(lines)

    def _ask_query(self, cands, splits, checks=None, space=None):
        env = self.env
        useful = {(c, r) for c, r, *_ in (checks or splits)}
        asked = {(q[0], _row(*q[1:])) for q, _ in env.history}
        ask = ("Choose ONE check from the list above (the first ones test the most)." if checks else
               "Choose ONE query from the list above that helps most.")
        base = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": self._state_text(cands, checks=checks, space=space) +
                 f"\n\nYou have {env.remaining} queries left. {ask} Reply with one short sentence of "
                 "reasoning, then exactly one <query>X(a, b, c)</query>."}]
        fallback = None
        msgs = list(base)
        for t in range(self.MAX_TRIES):
            reply = self.llm(msgs, temperature=0.3 if t == 0 else 0.8, max_tokens=300)
            m = QUERY_RE.search(reply or "")
            if not m:
                fb = "Your reply did not contain a query in the required format."
            else:
                c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
                r = _row(a, b, x)
                if (c, r) in useful:
                    return c, a, b, x
                if fallback is None:
                    fallback = (c, a, b, x)
                if (c, r) in asked:
                    fb = f"You already asked {c}({a}, {b}, {x}); the answer is in the list above."
                elif checks:
                    fb = f"{c}({a}, {b}, {x}) is not one of the listed checks."
                elif len(cands[c]) == 1:
                    fb = f"Circuit {c} is already solved; querying it wastes a query."
                else:
                    fb = (f"Every remaining candidate for circuit {c} gives the same output on ({a}, {b}, {x}), "
                          "so that query gives no new information.")
            msgs = base + [{"role": "assistant", "content": reply or ""},
                           {"role": "user", "content": fb + " Pick a query from the list " +
                            ("of checks" if checks else "of queries that can still tell candidates apart") +
                            ", and reply with exactly one <query>X(a, b, c)</query>."}]
        return fallback          # the model's own (weak) choice; None if it never produced one

    def _final(self, cands):
        env = self.env
        order = {c: [m for m, _ in sorted(cands[c].items(), key=lambda t: -t[1][1])] for c in LABELS}
        lines = [self._state_text(cands, show_menu=False), "",
                 "The querying phase is over. For each circuit, choose the candidate table that is "
                 "consistent with the answers. Numbered candidates:"]
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
            checks = None if splits else self._checks(space, cands)
            if not splits and not checks:
                break                 # every row of every circuit has been observed
            q = self._ask_query(cands, splits, checks, space)
            if q is None:
                wasted += 1
                continue
            c, a, b, x = q
            bit = env.query(c, a, b, x)
            space.observe(c, _row(a, b, x), bit)
        return self._final(space.candidates())
