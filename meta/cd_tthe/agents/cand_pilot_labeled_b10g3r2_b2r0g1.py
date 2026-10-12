"""Candidate: rule-derived bookkeeping around the frozen model.

The harness enumerates, from the task rules in the system prompt alone (3 inputs, AND/OR take two
signals, NOT takes one, every circuit reduces 3 signals to 1, no constant circuit, the inventory is
used exactly), every circuit function still consistent with the answers so far. The model still
writes every query and the final table; the harness
  * shows it a compact state each turn (known answers, rows already implied, open rows ranked by how
    evenly they split the remaining candidates) instead of the growing raw transcript,
  * vetoes queries whose answer is already implied and re-asks,
  * at the end shows the remaining candidate tables per circuit, and assembles the final guess from
    the model's reply, snapping each circuit to the consistent candidate nearest to what it wrote.

Changes over the parent (b0r0g2):
  * suggested queries are ranked by the binary entropy of the answer over the JOINT space (every
    (inventory split, table A, table B, table C) still consistent), not by the per-circuit count
    split: an answer on A also prunes splits and therefore B and C. Offline, with the model taking
    one of the top 4 at random, this cuts mean queries by 0.5-1.4 on NOT inventories and the
    unresolved-at-budget tail;
  * if a circuit is still ambiguous at the end, its candidates are listed most-likely-first (joint
    weight) and an unparsed circuit falls back to the most likely one instead of the first sorted.

Changes over the parent (b0r1g2):
  * only near-best queries are suggested (entropy >= SUGGEST_FRAC of the best), and a query well
    below the best (< VETO_FRAC) is vetoed and re-asked like a wasted one. In the parent's traces
    the model's 2nd query on every NOT=1 instance carried ~0.29 of the best entropy (e.g. A(0,0,0)
    while B(0,0,1) was suggested), and NOT inventories are the long tail (13-16 of 18 queries).
    If every ask of a turn is vetoed, the most informative query the model did propose is sent, so
    a stubborn model cannot stall the episode.

Changes over the parent (b1r0g2), synthesized from the b1r0 siblings' traces:
  * the query tag is accepted with its opening "<" swallowed by a code fence ("```query>C(0, 1, 1)</query>",
    from b1r0g0); the model wrote it 13 times across the b1r0 traces and on i228 of the parent it cost a
    re-ask because the reply also named a second input;
  * the final guess is snapped to the nearest JOINTLY consistent triple (one inventory split must
    explain A, B and C together), not circuit by circuit: snapping each circuit separately can return
    a combination no split allows, which is wrong for certain whenever the budget ends with ambiguity.

Changes over the parent (b1r1g2): the leftover budget tests the rule-derived space instead of being
thrown away.
  * Once every row is implied, the parent stopped (~6 of 18 queries unused) and returned the space's
    prediction, so a misread rule would give a wrong table with no warning. Now the model is told the
    predictions and spends the remaining queries checking implied rows. Suggested checks go first to
    rows the inventory split alone pins down (rows a per-circuit reading of the rules leaves open).
    The harness accepts any unanswered row; repeats are vetoed.
  * When an answer contradicts the space, the fallback is now per circuit: first the circuit's own
    rule space with no inventory coupling (any AND/OR mix, up to the NOT count), then any
    non-constant table. The parent's fallback dropped all three circuits to "any table" together.
    Querying then resumes on whatever became uncertain again.
"""

import itertools
import math
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<?query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
LOOSE_RE = re.compile(r"\b([A-C])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)")
GUESS_RE = re.compile(r"<guess>\s*([01\s]+?)\s*</guess>")
LABELS = "ABC"
ROWS = [(r >> 2 & 1, r >> 1 & 1, r & 1) for r in range(8)]
MAX_ASKS_PER_TURN = 4
SUGGEST_FRAC = 0.97
VETO_FRAC = 0.9


def _gate(op, u, v):
    return (u & v) if op == "AND" else (u | v)


def circuit_tables(n_and, n_or, n_not):
    """All 8-bit output tables of a 3-input circuit built from exactly these gates (two binary gates:
    first combine x_i, x_j, then combine with x_k; NOTs may sit on any of the 5 signals, stacking)."""
    if n_and + n_or != 2:
        return set()
    out = set()
    op_orders = set(itertools.permutations(["AND"] * n_and + ["OR"] * n_or))
    # parity patterns over (x_i, x_j, x_k, g1, g2) reachable with exactly n_not NOTs
    pars = [p for p in itertools.product((0, 1), repeat=5) if sum(p) <= n_not and (n_not - sum(p)) % 2 == 0]
    for (i, j, k) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        for op1, op2 in op_orders:
            for p in pars:
                tab = []
                for row in ROWS:
                    g1 = _gate(op1, row[i] ^ p[0], row[j] ^ p[1]) ^ p[3]
                    tab.append(_gate(op2, g1, row[k] ^ p[2]) ^ p[4])
                if 0 < sum(tab) < 8:
                    out.add(tuple(tab))
    return out


def joint_space(inv):
    """List of (setA, setB, setC) per way of splitting the inventory over the three circuits."""
    n_and, n_or, n_not = inv.get("AND", 0), inv.get("OR", 0), inv.get("NOT", 0)
    cache, joint = {}, []
    for ands in itertools.product(range(3), repeat=3):
        if sum(ands) != n_and or sum(2 - a for a in ands) != n_or:
            continue
        for nots in itertools.product(range(n_not + 1), repeat=3):
            if sum(nots) != n_not:
                continue
            sets = []
            for a, n in zip(ands, nots):
                key = (a, 2 - a, n)
                if key not in cache:
                    cache[key] = circuit_tables(*key)
                sets.append(cache[key])
            joint.append(sets)
    return joint


def fmt_table(t):
    return "".join(map(str, t))


class Harness(CDHarness):
    def _marginals(self):
        """Per circuit: set of tables consistent with all answers AND with some consistent split."""
        obs = {c: {} for c in LABELS}
        for (c, a, b, x), bit in self.env.history:
            obs[c][4 * a + 2 * b + x] = bit
        marg = {c: set() for c in LABELS}
        # joint weights: number of consistent (split, A, B, C) combinations with a given table / bit
        self._tw = {c: {} for c in LABELS}
        self._w1 = {}
        self._W = 0
        for sets in self._joint:
            filt = [{t for t in s if all(t[r] == v for r, v in obs[c].items())} for c, s in zip(LABELS, sets)]
            if all(filt):
                n = [len(f) for f in filt]
                tot = math.prod(n)
                self._W += tot
                for c, f, nc in zip(LABELS, filt, n):
                    marg[c] |= f
                    rest = tot // nc
                    for t in f:
                        self._tw[c][t] = self._tw[c].get(t, 0) + rest
                    for r in range(8):
                        self._w1[(c, r)] = self._w1.get((c, r), 0) + rest * sum(t[r] for t in f)
        if not all(marg.values()):            # rules misread -> per circuit, drop the coupling, then the structure
            allt = [t for t in itertools.product((0, 1), repeat=8) if 0 < sum(t) < 8]
            for c in LABELS:
                fit = lambda ts: {t for t in ts if all(t[r] == v for r, v in obs[c].items())}
                marg[c] = fit(self._loose) or fit(allt)
            self._W = 0                       # joint weights meaningless here: rank by count split
        return obs, marg

    def _score(self, c, r, cands):
        """Information of querying row r of circuit c: entropy of its answer over the joint space."""
        if self._W:
            p = self._w1.get((c, r), 0) / self._W
        else:
            p = sum(t[r] for t in cands) / len(cands)
        return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def _ranked(self, c, cands):
        """Candidate tables of circuit c, most likely (largest joint weight) first."""
        tw = self._tw.get(c, {}) if self._W else {}
        return sorted(cands, key=lambda t: (-tw.get(t, 0), t))

    def _state_text(self, obs, marg):
        lines, opts = [], []
        for c in LABELS:
            cands = marg[c]
            known = ", ".join(f"{c}({r >> 2 & 1},{r >> 1 & 1},{r & 1})={v}" for r, v in sorted(obs[c].items()))
            if len(cands) == 1:
                lines.append(f"Circuit {c}: SOLVED -> table {fmt_table(next(iter(cands)))}. Do not query {c} again.")
                continue
            implied, open_rows = [], []
            for r in range(8):
                vals = {t[r] for t in cands}
                if r in obs[c]:
                    continue
                if len(vals) == 1:
                    implied.append(f"{c}({r >> 2 & 1},{r >> 1 & 1},{r & 1})={next(iter(vals))}")
                else:
                    h = self._score(c, r, cands)
                    open_rows.append((h, r))
                    opts.append((h, c, r))
            lines.append(f"Circuit {c}: {len(cands)} possible structures remain.")
            lines.append(f"  answers so far: {known or 'none'}")
            if implied:
                lines.append(f"  already implied by the rules (querying these is wasted): {', '.join(implied)}")
            lines.append("  still uncertain inputs: " + ", ".join(
                f"({r >> 2 & 1}, {r >> 1 & 1}, {r & 1})" for _, r in sorted(open_rows, reverse=True)))
        opts.sort(key=lambda o: -o[0])
        top = [o for o in opts[:4] if o[0] >= SUGGEST_FRAC * opts[0][0]] if opts else []
        best = [f"{c}({r >> 2 & 1}, {r >> 1 & 1}, {r & 1})" for _, c, r in top]
        return "\n".join(lines), best, opts

    def solve(self):
        env = self.env
        self._joint = joint_space(env.inventory)
        B = env.budget
        sys_msg = {"role": "system", "content": env.system_prompt}
        calls_left = 6 * B
        while env.remaining > 0 and calls_left > 0:
            obs, marg = self._marginals()
            state, best, opts = self._state_text(obs, marg)
            if not opts:                       # everything determined: further queries add nothing
                break
            informative = {(c, r): h for h, c, r in opts}
            floor = VETO_FRAC * opts[0][0]
            note = ""
            sent = False
            fallback = None                    # most informative vetoed-but-informative proposal
            for _ in range(MAX_ASKS_PER_TURN):
                if calls_left <= 0:
                    break
                calls_left -= 1
                prompt = (f"Current state after {len(env.history)} queries ({env.remaining} remaining):\n{state}\n\n"
                          f"Queries that split the remaining possibilities most evenly: {', '.join(best)}.\n"
                          f"{note}Choose ONE query on a still-uncertain input. Reply with a short reason and exactly "
                          f"one <query>X(a, b, c)</query>.")
                reply = self.llm([sys_msg, {"role": "user", "content": prompt}])
                m = QUERY_RE.search(reply or "")
                if not m:
                    loose = {g for g in LOOSE_RE.findall(reply or "")}
                    if len(loose) == 1:
                        g = next(iter(loose))
                        m_vals = (g[0].upper(), int(g[1]), int(g[2]), int(g[3]))
                    else:
                        note = "Your previous reply had no query in the required format. "
                        continue
                else:
                    m_vals = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                c, a, b, x = m_vals
                if (c, 4 * a + 2 * b + x) not in informative:
                    why = ("was already answered" if 4 * a + 2 * b + x in obs[c]
                           else "is already determined by the rules")
                    note = f"Your previous choice {c}({a}, {b}, {x}) {why} and would waste a query. "
                    continue
                h = informative[(c, 4 * a + 2 * b + x)]
                if h < floor:
                    if fallback is None or h > fallback[0]:
                        fallback = (h, c, a, b, x)
                    note = (f"Your previous choice {c}({a}, {b}, {x}) tells much less than the suggested queries "
                            f"({', '.join(best)}); pick one of those. ")
                    continue
                env.query(c, a, b, x)
                sent = True
                break
            if not sent and fallback is not None:
                env.query(*fallback[1:])
                sent = True
            if not sent and calls_left <= 0:
                break
        return self._final()

    def _final(self):
        env = self.env
        obs, marg = self._marginals()
        lines = []
        for c in LABELS:
            cands = self._ranked(c, marg[c])
            if len(cands) == 1:
                lines.append(f"Circuit {c}: determined, table {fmt_table(cands[0])}")
            else:
                lines.append(f"Circuit {c}: still ambiguous; the tables consistent with all answers, most likely first, are: "
                             + ", ".join(fmt_table(t) for t in cands[:12]) + (" ..." if len(cands) > 12 else ""))
        answers = "\n".join(env.format_answer(c, a, b, x, bit) for (c, a, b, x), bit in env.history)
        prompt = (f"All queries are done. Answers received:\n{answers}\n\n"
                  "Each circuit's 8 outputs, in the order (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1):\n"
                  + "\n".join(lines) +
                  "\n\nFor an ambiguous circuit pick a table from its list (the first is the most likely). "
                  "First write one line per circuit with its own 8 digits, in the form:\nA: <A's 8 digits>\n"
                  "B: <B's 8 digits>\nC: <C's 8 digits>\nthen the three tables concatenated (A, then B, then C) into 24 digits.\n"
                  + env.last_turn_prompt)
        sys_msg = {"role": "system", "content": env.system_prompt}
        said = {}
        for _ in range(2):
            reply = self.llm([sys_msg, {"role": "user", "content": prompt}]) or ""
            gs = GUESS_RE.findall(reply)
            bits = re.sub(r"\s", "", gs[-1]) if gs else ""
            for i, c in enumerate(LABELS):    # per-circuit lines first: the 7B drops digits when concatenating
                rows = re.findall(rf"\b{c}\s*:\s*([01]{{8}})\b", reply)
                if rows:
                    said[c] = tuple(int(ch) for ch in rows[-1])
                elif len(bits) == 24 and c not in said:
                    said[c] = tuple(int(ch) for ch in bits[8 * i:8 * i + 8])
            if len(said) == len(LABELS):
                break
            prompt += "\nYour previous reply did not give 8 digits for every circuit. Reply with the three lines A: ..., B: ..., C: ..."
        out = self._snap_joint(obs, said) if self._W else None
        if out is None:
            out = []
            for c in LABELS:
                cands = self._ranked(c, marg[c])
                if c not in said:
                    out.append(cands[0])
                else:   # keep the model's table if consistent, else the consistent table nearest to it
                    out.append(min(cands, key=lambda t: sum(u != v for u, v in zip(t, said[c]))))
        return "<guess>" + "".join(fmt_table(t) for t in out) + "</guess>"

    def _snap_joint(self, obs, said):
        """Consistent (A, B, C) from a single inventory split nearest to what the model said; ties (and
        circuits it did not state) go to the most likely tables. Distance is additive over circuits, so
        within a split each circuit is snapped on its own and the best split wins."""
        best = None
        for sets in self._joint:
            filt = [[t for t in s if all(t[r] == v for r, v in obs[c].items())] for c, s in zip(LABELS, sets)]
            if not all(filt):
                continue
            pick, dist, like = [], 0, 1
            for c, f in zip(LABELS, filt):
                tw = self._tw[c]
                t = min(f, key=lambda t: (sum(u != v for u, v in zip(t, said[c])) if c in said else 0,
                                          -tw.get(t, 0), t))
                pick.append(t)
                dist += sum(u != v for u, v in zip(t, said[c])) if c in said else 0
                like *= tw.get(t, 0)
            key = (dist, -like, pick)
            if best is None or key < best:
                best = key
        return best[2] if best else None
