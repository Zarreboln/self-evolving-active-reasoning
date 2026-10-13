"""Bookkeeping harness: the frozen model still writes every query and every circuit table, but the
harness keeps the books it is bad at keeping.

From the task rules alone (inventory; each gate consumes its inputs until one signal is left; no
constant circuit; all components used) the harness enumerates every output table each circuit could
have, and after each answer keeps only the tables consistent with ALL answers (across circuits:
the gate allocation is shared). With that it
  * shows the model, each turn, a fresh compact state: answers so far, how many tables each circuit
    still allows, and which inputs are still undecided (most evenly splitting first);
  * vetoes a query whose answer is already forced (or repeated) and asks again -- the turn is not spent;
  * stops querying once every circuit has a single table left;
  * asks the model for each circuit's 8 outputs separately (listing the surviving tables), rejects a
    reply that contradicts the answers, and assembles the 24-bit guess from the three replies.
If the enumeration ever loses all candidates for a circuit (rules read differently than here), it
falls back to observation-only checks.
"""

import functools
import itertools
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
TABLE_RE = re.compile(r"<table>\s*([01](?:\s*[01]){7})\s*</table>")
BITS8_RE = re.compile(r"(?<![01])[01]{8}(?![01])")
FULL = 0xFF
INPUT_MASKS = (0xF0, 0xCC, 0xAA)          # bit r = output on row r, r = 4*a + 2*b + c
ROWS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
CIRCS = "ABC"


def row_of(a, b, c):
    return 4 * a + 2 * b + c


def table_str(f):
    return "".join(str((f >> r) & 1) for r in range(8))


def table_int(s):
    return sum(int(ch) << r for r, ch in enumerate(s))


@functools.lru_cache(maxsize=None)
def _reach(sig, n_and, n_or, n_not):
    if len(sig) == 1 and n_and == n_or == n_not == 0:
        return frozenset(sig)
    out = set()
    n = len(sig)
    if n_not:
        for i in range(n):
            s = list(sig); s[i] = FULL ^ s[i]
            out |= _reach(tuple(sorted(s)), n_and, n_or, n_not - 1)
    for i in range(n):
        for j in range(i + 1, n):
            rest = [sig[k] for k in range(n) if k not in (i, j)]
            if n_and:
                out |= _reach(tuple(sorted(rest + [sig[i] & sig[j]])), n_and - 1, n_or, n_not)
            if n_or:
                out |= _reach(tuple(sorted(rest + [sig[i] | sig[j]])), n_and, n_or - 1, n_not)
    return frozenset(out)


def circuit_tables(n_and, n_or, n_not):
    return frozenset(f for f in _reach(tuple(sorted(INPUT_MASKS)), n_and, n_or, n_not) if f not in (0, FULL))


def _splits(n, k):
    if k == 1:
        yield (n,); return
    for i in range(n + 1):
        for rest in _splits(n - i, k - 1):
            yield (i,) + rest


def allocations(inv):
    """Every way to distribute the inventory over A, B, C -> list of per-circuit table sets."""
    na, no, nn = inv.get("AND", 0), inv.get("OR", 0), inv.get("NOT", 0)
    out = []
    for a, o, n in itertools.product(_splits(na, 3), _splits(no, 3), _splits(nn, 3)):
        sets = [circuit_tables(a[i], o[i], n[i]) for i in range(3)]
        if all(sets):
            out.append(sets)
    return out


class Harness(CDHarness):
    MAX_CALLS = 70

    # ---------- bookkeeping ----------
    def _observed(self):
        obs = {c: {} for c in CIRCS}
        for (c, a, b, x), bit in self.env.history:
            obs[c][row_of(a, b, x)] = bit
        return obs

    def _alive(self, fixed=None):
        """Per-circuit tables consistent with all answers (and with any tables already fixed)."""
        fixed = fixed or {}
        obs = self._observed()
        per = {c: set() for c in CIRCS}
        for sets in self.allocs:
            ok = []
            for i, c in enumerate(CIRCS):
                s = {f for f in sets[i] if all(((f >> r) & 1) == v for r, v in obs[c].items())}
                if c in fixed:
                    s &= {fixed[c]}
                ok.append(s)
            if all(ok):
                for i, c in enumerate(CIRCS):
                    per[c] |= ok[i]
        if not fixed and not all(per.values()):   # the rules were read differently than the enumeration assumes
            self.space_ok = False
        return per

    def _open_rows(self, per):
        """[(circuit, (a,b,c), n_ones, n_total)] for rows whose output is not yet forced, best split first."""
        obs = self._observed()
        out = []
        for c in CIRCS:
            n = len(per[c])
            for a, b, x in ROWS:
                r = row_of(a, b, x)
                if r in obs[c]:
                    continue
                if self.space_ok:
                    ones = sum((f >> r) & 1 for f in per[c])
                    if 0 < ones < n:
                        out.append((c, (a, b, x), ones, n))
                else:
                    out.append((c, (a, b, x), 0, 0))
        out.sort(key=lambda o: (-min(o[2], o[3] - o[2]), -o[3]))
        return out

    def _call(self, msgs, **kw):
        self.calls += 1
        return self.llm(msgs, **kw) or ""

    # ---------- what the model sees ----------
    def _state_text(self, per):
        env = self.env
        lines = []
        if env.history:
            lines.append("Answers so far:")
            lines += [env.format_answer(c, a, b, x, bit) for (c, a, b, x), bit in env.history]
        else:
            lines.append("No queries made yet.")
        lines.append(f"\n{len(env.history)} queries used, {env.remaining} remaining.")
        if self.space_ok:
            lines.append("\nWhat the rules (your gate inventory) plus these answers still allow -- computed for you:")
            open_rows = self._open_rows(per)
            for c in CIRCS:
                if len(per[c]) == 1:
                    lines.append(f"* Circuit {c}: SOLVED, its outputs (000..111) are {table_str(next(iter(per[c])))}. "
                                 f"Do not query {c} again.")
                else:
                    rows = [o for o in open_rows if o[0] == c]
                    desc = ", ".join(f"{c}({a}, {b}, {x})" for _, (a, b, x), _, _ in rows)
                    lines.append(f"* Circuit {c}: {len(per[c])} possible output tables left. "
                                 f"Inputs whose output is still unknown: {desc}.")
            if open_rows:
                best = [f"{c}({a}, {b}, {x})" for c, (a, b, x), o, n in open_rows[:4]]
                lines.append("Queries that split the remaining possibilities most evenly (most informative): "
                             + ", ".join(best) + ".")
        return "\n".join(lines)

    # ---------- episode ----------
    def solve(self):
        env = self.env
        self.calls = 0
        self.space_ok = True
        self.allocs = allocations(env.inventory)
        if not self.allocs:
            self.space_ok = False
        asked = set()
        while env.remaining > 0 and self.calls < self.MAX_CALLS - 6:
            per = self._alive() if self.space_ok else {c: set() for c in CIRCS}
            if self.space_ok and all(len(per[c]) == 1 for c in CIRCS):
                break
            open_rows = self._open_rows(per)
            if not open_rows:
                break
            allowed = {(c, abx) for c, abx, _, _ in open_rows}
            note = ""
            sent = False
            for attempt in range(4):
                msgs = [{"role": "system", "content": env.system_prompt},
                        {"role": "user", "content": self._state_text(per) + note +
                         "\n\nChoose ONE query whose output is still unknown. Think briefly, then reply with "
                         "exactly one <query>X(a, b, c)</query>."}]
                reply = self._call(msgs, temperature=0.7 if attempt == 0 else 0.9, max_tokens=500)
                m = QUERY_RE.findall(reply)
                if not m:
                    note = "\n\n(Your last reply had no <query>X(a, b, c)</query>. Use exactly that format.)"
                    continue
                c, a, b, x = m[-1][0].upper(), int(m[-1][1]), int(m[-1][2]), int(m[-1][3])
                if (c, (a, b, x)) not in allowed:
                    why = ("you already asked it" if (c, a, b, x) in asked else
                           "its output is already forced by the answers and the rules, so it gives no information")
                    note = (f"\n\n(Rejected {c}({a}, {b}, {x}): {why}. It was not sent and did not cost a query. "
                            f"Pick an input listed as still unknown.)")
                    continue
                env.query(c, a, b, x)
                asked.add((c, a, b, x))
                sent = True
                break
            if not sent and self.calls >= self.MAX_CALLS - 6:
                break
        return self._final()

    def _final(self):
        env = self.env
        obs = self._observed()
        fixed, final = {}, {}
        for c in CIRCS:
            per = self._alive(fixed) if self.space_ok else {k: set() for k in CIRCS}
            cands = sorted(per[c], key=table_str) if self.space_ok else []
            known = ", ".join(f"{c}({a}, {b}, {x}) = {obs[c][row_of(a, b, x)]}"
                              for a, b, x in ROWS if row_of(a, b, x) in obs[c]) or "none"
            if cands and len(cands) == 1:
                lst = f"Exactly one table is consistent with the rules and all answers: {table_str(cands[0])}."
            elif cands:
                shown = cands[:12]
                lst = (f"{len(cands)} tables are still consistent with the rules and all answers"
                       + (" (first 12 shown)" if len(cands) > 12 else "") + ":\n"
                       + "\n".join(f"  {table_str(f)}" for f in shown) + "\nPick the one you find most plausible.")
            else:
                lst = "Fill unknown inputs with your best reasoning about the gates."
            note = ""
            chosen = None
            said = []
            for attempt in range(4):
                msgs = [{"role": "system", "content": env.system_prompt},
                        {"role": "user", "content":
                         f"The queries are over. Now write circuit {c}'s outputs only.\n"
                         f"Observed answers for {c}: {known}.\n{lst}\n"
                         f"Write the 8 outputs in the order {c}(0,0,0), {c}(0,0,1), {c}(0,1,0), {c}(0,1,1), "
                         f"{c}(1,0,0), {c}(1,0,1), {c}(1,1,0), {c}(1,1,1) as <table>XXXXXXXX</table>." + note}]
                reply = self._call(msgs, temperature=0.2 if attempt == 0 else 0.7, max_tokens=400)
                m = TABLE_RE.findall(reply)
                bits = re.sub(r"\s", "", m[-1]) if m else None
                if bits is None:
                    loose = BITS8_RE.findall(reply)
                    bits = loose[-1] if loose else None
                if bits is None:
                    note = "\n(Your last reply had no <table>XXXXXXXX</table> with 8 bits. Use exactly that format.)"
                    continue
                said.append(bits)
                f = table_int(bits)
                bad = [f"{c}({a}, {b}, {x})" for a, b, x in ROWS
                       if row_of(a, b, x) in obs[c] and ((f >> row_of(a, b, x)) & 1) != obs[c][row_of(a, b, x)]]
                if bad:
                    note = (f"\n(Your table {bits} contradicts the observed answer for {', '.join(bad)}. "
                            f"Write a table consistent with every observed answer.)")
                    continue
                if cands and f not in cands:
                    note = (f"\n(Your table {bits} cannot be built from the gates the rules allow. "
                            f"Choose one of the listed tables.)")
                    continue
                chosen = bits
                break
            if chosen is None:
                # assemble from what the model said: prefer a stated table that matches the observations
                chosen = next((s for s in said if all(((table_int(s) >> r) & 1) == v for r, v in obs[c].items())),
                              said[-1] if said else "0" * 8)
            final[c] = chosen
            if self.space_ok and table_int(chosen) in cands:
                fixed[c] = table_int(chosen)   # later circuits must share the gate allocation with this one
        return "<guess>" + "".join(final[c] for c in CIRCS) + "</guess>"
