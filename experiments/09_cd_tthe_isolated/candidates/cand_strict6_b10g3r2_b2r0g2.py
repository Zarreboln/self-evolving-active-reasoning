"""Branch b2r0g2 (parent: b1r1g0), adversarial audit + synthesis. Disagreements on this batch between the
parent and b0r1g1 (no known-cell veto) support the parent's veto: b0r1g1 spent queries on cells the board
already marked deduced (B000, C000 on i230/i266/i284) and on i266 ran out of queries with two triples still
allowed; the parent pinned one triple on all 10. What the traces do NOT support is the parent's margin: the
model sweeps the first '?' cell in row order (A000, A001, ... then B, then C), and a rule-model simulation of
that sweep over all triples for a 3 AND / 1 NOT / 3 OR inventory needs more than 18 queries on ~15% of them
(the parent's i266 used 17). Repair: the board also lists the '?' cells whose answer is sure to rule out at
least VALUE_FRAC of the most any '?' cell can (counted over the triples the rules and answers still allow);
a query on another '?' cell is not sent and the model is shown that list (at most WEAK_MAX times in a row,
then the query goes through). Vetoed exchanges are dropped from the conversation once a query is sent.
Everything else is the parent's.

Branch b1r1g0 (parent: b1r0g0), conservative repair. Trace-visible failure of the parent: on one
instance the model sent a <guess> mid-episode (3 cells still '?', 9 queries left) and repeated it through
30 identical 'no query' re-asks until the conversation exceeded the model's context window -> crash, no
guess. Repair: after a reply with no query, the re-ask names the board's '?' cells when the model tried to
guess early; after MISS_MAX consecutive query-less replies (or if a model call fails) the query phase
ends and the parent's final phase runs on the board as it stands. Everything else is the parent's.

Branch b1r0g0 (parent: b0r1g1), conservative repair. Two trace-visible failures of the parent:
(1) the model re-queries cells the board already marks observed/deduced (12 queries on this batch, 5 in one
episode that then used 17 of 18), and (2) when the budget runs out with '?' cells left, the model fills them
with a table the rules exclude (a constant circuit) and keeps it through every re-ask, so the parent's
fallback returns a certainly-wrong circuit. Repairs: a query on an already-known cell is not sent to the
environment; the model is told its value and asked for a '?' cell instead (at most VETO_MAX times in a row,
then the query goes through). In the final phase, if a circuit still has '?' cells and the rules leave at
most LIST_MAX complete tables for it, those tables are listed and the model copies one; the fallback prefers
any listed table the model wrote. Everything else is the parent's.

Parent (b0r1g1, parent: b0r0g1). Hypothesis: episodes go wrong because the model never SEES what the
rules + answers already imply. It re-queries rows whose output is already forced (sweeping one circuit
row by row), leaves other circuits under-probed, and at the end fills unqueried rows by guessing a gate
formula -- even when the answers already pin every row. It does not need a query chooser or vetoes, only
a visible knowledge state.

One mechanism, a deduction board: after every answer the harness computes, from the task rules alone
(inventory split without replacement across A, B, C; each AND/OR consumes two signals, NOT replaces one,
until one signal is left; no constant circuit) and all answers so far, which of the 24 outputs are
observed, which are forced (same value in every circuit triple still allowed), and which are open ('?').
The board is appended to each answer in the model's own ongoing conversation (bare's query protocol
otherwise). Querying stops when the board has no '?'. In the final phase the model copies each circuit's
row of the board as <table>XXXXXXXX</table>, filling any '?' itself; a table that contradicts a known
cell, or that no circuit allowed by the rules can produce, is sent back with the reason.
If the rule enumeration ever explains no triple (rules read differently), the board shows observed
cells only."""

import itertools
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
CELL_RE = re.compile(r"(?<![A-Za-z])([A-C])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)")
FENCE_RE = re.compile(r"```\s*query\s*\n?(.*?)```", re.S | re.I)
TABLE_RE = re.compile(r"<table>\s*([01](?:\s*[01]){7})\s*</table>")
BITS8_RE = re.compile(r"(?<![01])[01]{8}(?![01])")
ROWS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
CIRCS = "ABC"
MAX_CALLS = 80
FINAL_TRIES = 4
VETO_MAX = 2
WEAK_MAX = 2
VALUE_FRAC = 0.75
LIST_MAX = 8
MISS_MAX = 4


def _tables(n_and, n_or, n_not):
    """Output tables (8-bit ints, bit r = output on row r, row r = a*4+b*2+c) of every circuit that uses
    exactly these gates; constant circuits are excluded (task rules)."""
    out, seen = set(), set()
    x = [sum(((r >> (2 - k)) & 1) << r for r in range(8)) for k in range(3)]

    def rec(sigs, a, o, n):
        key = (sigs, a, o, n)
        if key in seen:
            return
        seen.add(key)
        if len(sigs) == 1 and a == o == n == 0:
            out.add(sigs[0])
            return
        for i in range(len(sigs)):
            if n:
                rec(tuple(sorted(sigs[:i] + (255 ^ sigs[i],) + sigs[i + 1:])), a, o, n - 1)
            for j in range(i + 1, len(sigs)):
                rest = sigs[:i] + sigs[i + 1:j] + sigs[j + 1:]
                if a:
                    rec(tuple(sorted(rest + (sigs[i] & sigs[j],))), a - 1, o, n)
                if o:
                    rec(tuple(sorted(rest + (sigs[i] | sigs[j],))), a, o - 1, n)

    rec(tuple(sorted(x)), n_and, n_or, n_not)
    return {t for t in out if t not in (0, 255)}


def _splits(total):
    return [(i, j, total - i - j) for i in range(total + 1) for j in range(total - i + 1)]


def _joint_space(inv):
    """Every (tA, tB, tC) the task rules allow for this inventory."""
    space, cache = set(), {}
    for a3 in _splits(inv.get("AND", 0)):
        for o3 in _splits(inv.get("OR", 0)):
            for n3 in _splits(inv.get("NOT", 0)):
                per = []
                for k in range(3):
                    g = (a3[k], o3[k], n3[k])
                    if g not in cache:
                        cache[g] = _tables(*g)
                    per.append(cache[g])
                if all(per):
                    space.update(itertools.product(*per))
    return space


def _bits(t):
    return "".join(str((t >> r) & 1) for r in range(8))


def _row(a, b, x):
    return a * 4 + b * 2 + x


class Harness(CDHarness):
    def solve(self):
        env = self.env
        self.calls = 0
        try:
            self.space = _joint_space(env.inventory)
        except Exception:
            self.space = set()
        self._query_phase()
        return self._final_phase()

    def _call(self, msgs, **kw):
        self.calls += 1
        return self.llm(msgs, **kw) or ""

    # ---- the deduction board ----
    def _observed(self):
        obs = {c: {} for c in CIRCS}
        for (c, a, b, x), bit in self.env.history:
            obs[c][_row(a, b, x)] = bit
        return obs

    def _alive(self, fixed=None):
        fixed = fixed or {}
        return [s for s in self.space
                if all(((s[CIRCS.index(c)] >> _row(a, b, x)) & 1) == bit for (c, a, b, x), bit in self.env.history)
                and all(s[CIRCS.index(c)] == t for c, t in fixed.items())]

    def _board(self, fixed=None):
        """{c: {row: (value or None, 'observed' | 'deduced' | 'open')}} and the alive triples."""
        obs = self._observed()
        alive = self._alive(fixed)
        board = {}
        for i, c in enumerate(CIRCS):
            board[c] = {}
            for r in range(8):
                if r in obs[c]:
                    board[c][r] = (obs[c][r], "observed")
                elif alive and len({(s[i] >> r) & 1 for s in alive}) == 1:
                    board[c][r] = ((alive[0][i] >> r) & 1, "deduced")
                else:
                    board[c][r] = (None, "open")
        return board, alive

    def _value(self, board, alive):
        """{(c, row): how many alive triples each answer to that '?' cell is sure to rule out} and the set of
        '?' cells whose sure elimination is at least VALUE_FRAC of the best one (empty if the rules explain
        nothing)."""
        n, value = len(alive), {}
        for i, c in enumerate(CIRCS):
            for r in range(8):
                if board[c][r][0] is None and alive:
                    ones = sum((s[i] >> r) & 1 for s in alive)
                    value[(c, r)] = min(ones, n - ones)
        best = max(value.values(), default=0)
        good = {k for k, v in value.items() if best and v >= VALUE_FRAC * best}
        return value, good

    @staticmethod
    def _parse_query(reply):
        """The model's query: the last <query>..</query>; else a ```query``` block naming one cell; else the
        one cell the reply names, if it names exactly one distinct cell."""
        m = QUERY_RE.findall(reply)
        if m:
            return m[-1]
        for block in reversed(FENCE_RE.findall(reply)):
            cells = CELL_RE.findall(block)
            if cells:
                return cells[-1]
        cells = {(c.upper(), a, b, x) for c, a, b, x in CELL_RE.findall(reply)}
        return next(iter(cells)) if len(cells) == 1 else None

    def _cell(self, c, r):
        return f"{c}({r >> 2},{(r >> 1) & 1},{r & 1})"

    def _board_text(self, board, alive=None):
        lines = ["KNOWLEDGE BOARD (observed = answered by a query; deduced = forced by your gate inventory, the "
                 "building rules and all answers so far, so querying it would waste a query; ? = still unknown):"]
        for c in CIRCS:
            cells = []
            for a, b, x in ROWS:
                v, kind = board[c][_row(a, b, x)]
                cells.append(f"{c}({a},{b},{x})=" + ("?" if v is None else f"{v} {kind}"))
            n_open = sum(1 for r in range(8) if board[c][r][0] is None)
            lines.append(f"Circuit {c} ({n_open} unknown): " + ", ".join(cells))
        if alive:
            _, good = self._value(board, alive)
            if good:
                lines.append(f"{len(alive)} combinations of circuits A, B, C still fit the rules and all answers. "
                             f"The unknown cells whose answer rules out the most of them (whichever way it "
                             f"comes out): " + ", ".join(self._cell(c, r) for c in CIRCS for r in range(8)
                                                       if (c, r) in good) + ".")
        return "\n".join(lines)

    # ---- query phase: bare's conversation, each answer followed by the board ----
    def _query_phase(self):
        env = self.env
        board, alive = self._board()
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": self._board_text(board, alive) + "\n\nPlease start with your first query."}]
        base = len(msgs)          # vetoed exchanges after this point are dropped once a query goes through
        vetoes = misses = weak = 0
        while env.remaining > 0 and self.calls < MAX_CALLS - 3 * FINAL_TRIES:
            try:
                reply = self._call(msgs)
            except Exception:
                break
            msgs.append({"role": "assistant", "content": reply})
            m = self._parse_query(reply)
            if not m:
                misses += 1
                if misses > MISS_MAX:
                    break
                open_cells = [f"{k}({a},{b},{x})" for k in CIRCS for a, b, x in ROWS
                              if board[k][_row(a, b, x)][0] is None]
                if "<guess>" in reply and open_cells:
                    note = (f"It is not time to guess yet: the board still has {len(open_cells)} unknown cells ("
                            + ", ".join(open_cells) + f") and you have {env.remaining} queries left. Query one "
                            "of those cells. Reply with exactly one <query>X(a, b, c)</query>.")
                else:
                    note = ("Your message did not contain a query in the required format. Reply with exactly "
                            "one <query>X(a, b, c)</query>, for example <query>A(0, 1, 1)</query>, with the "
                            "angle-bracket tags (not a code block).")
                # keep only the latest failed exchange so repeated re-asks cannot grow the conversation
                del msgs[base:-1]
                msgs.append({"role": "user", "content": note})
                continue
            m = [m]
            misses = 0
            c, a, b, x = m[-1][0].upper(), int(m[-1][1]), int(m[-1][2]), int(m[-1][3])
            r = _row(a, b, x)
            v, kind = board[c][r]
            if v is not None and vetoes < VETO_MAX:
                vetoes += 1
                msgs.append({"role": "user", "content": f"{c}({a},{b},{x}) is already known on the board: "
                             f"{v} ({kind}). Querying it would waste a query, so it was not sent. Choose a cell "
                             f"marked '?' instead. {env.remaining} queries remaining. Reply with exactly one "
                             f"<query>X(a, b, c)</query>."})
                continue
            value, good = self._value(board, alive)
            if v is None and good and (c, r) not in good and weak < WEAK_MAX:
                weak += 1
                best = max(value.values())
                msgs.append({"role": "user", "content": f"{c}({a},{b},{x}) is unknown, but of the {len(alive)} "
                             f"combinations still possible its answer is only sure to rule out {value[(c, r)]}, "
                             f"while other unknown cells are sure to rule out up to {best}. It was not sent. "
                             f"Choose one of these cells instead: "
                             + ", ".join(self._cell(k, q) for k in CIRCS for q in range(8) if (k, q) in good)
                             + f". {env.remaining} queries remaining. Reply with exactly one "
                             f"<query>X(a, b, c)</query>."})
                continue
            vetoes = weak = 0
            del msgs[base:-1]
            bit = env.query(c, a, b, x)
            board, alive = self._board()
            if all(board[k][q][0] is not None for k in CIRCS for q in range(8)):
                break
            msgs.append({"role": "user", "content": f"{env.format_answer(c, a, b, x, bit)}\n\n"
                         f"{self._board_text(board, alive)}\n\nPlease send your next query. "
                         f"{env.remaining} queries remaining."})
            base = len(msgs)

    # ---- final phase: copy the board, one circuit at a time ----
    def _final_phase(self):
        fixed, final = {}, {}
        use_rules = bool(self._alive())
        for c in CIRCS:
            board, alive = self._board(fixed if use_rules else None)
            if not alive:
                board, alive = self._board()
            i = CIRCS.index(c)
            cells = board[c]
            known = {r: v for r, (v, _) in cells.items() if v is not None}
            template = "".join("?" if cells[r][0] is None else str(cells[r][0]) for r in range(8))
            listing = "\n".join(f"{c}({a},{b},{x}) = " + ("?" if cells[_row(a, b, x)][0] is None else
                                                         f"{cells[_row(a, b, x)][0]} ({cells[_row(a, b, x)][1]})")
                                for a, b, x in ROWS)
            ask = (f"The queries are over. Circuit {c} on the knowledge board (rows 000..111):\n{listing}\n\n"
                   f"Write circuit {c}'s 8 outputs in the order {c}(0,0,0), {c}(0,0,1), ..., {c}(1,1,1), "
                   f"keeping every observed and deduced value"
                   + (", and replacing each '?' with your best deduction from the gates" if len(known) < 8 else "")
                   + f". Board row: {template}. Answer as <table>XXXXXXXX</table>.")
            options = sorted({_bits(s[i]) for s in alive}) if use_rules and alive and len(known) < 8 else []
            if not 1 < len(options) <= LIST_MAX:
                options = []
            if options:
                ask += (f"\nThe gate rules together with all answers allow exactly these {len(options)} tables for "
                        f"circuit {c}: " + ", ".join(options) + ". Copy the one you judge most likely.")
            note, chosen, said = "", None, []
            for k in range(FINAL_TRIES):
                msgs = [{"role": "system", "content": self.env.system_prompt},
                        {"role": "user", "content": ask + note}]
                reply = self._call(msgs, temperature=0.2 if k == 0 else 0.7, max_tokens=600)
                m = TABLE_RE.findall(reply)
                bits = re.sub(r"\s", "", m[-1]) if m else None
                if bits is None:
                    loose = BITS8_RE.findall(reply)
                    bits = loose[-1] if loose else None
                if bits is None:
                    note = "\n(Your last reply had no <table>XXXXXXXX</table> with 8 bits. Use exactly that format.)"
                    continue
                said.append(bits)
                t = sum(int(ch) << r for r, ch in enumerate(bits))
                bad = [r for r, v in known.items() if ((t >> r) & 1) != v]
                if bad:
                    note = (f"\n(Your table {bits} disagrees with the board at "
                            + ", ".join(f"{c}({r >> 2},{(r >> 1) & 1},{r & 1})" for r in bad)
                            + ". Keep every observed and deduced value.)")
                    continue
                if use_rules and alive and not any(s[i] == t for s in alive):
                    note = (f"\n(Your table {bits} cannot be produced by any circuit the rules allow together "
                            f"with the other circuits. "
                            + (f"Copy one of: {', '.join(options)}.)" if options else "Change some '?' rows.)"))
                    continue
                chosen = bits
                break
            if chosen is None:
                chosen = next((s for s in said if s in options), None)
            if chosen is None:
                chosen = next((s for s in said
                               if all(((sum(int(ch) << r for r, ch in enumerate(s)) >> r) & 1) == v
                                      for r, v in known.items())),
                              said[-1] if said else template.replace("?", "0"))
            final[c] = chosen
            t = sum(int(ch) << r for r, ch in enumerate(chosen))
            if use_rules and any(s[i] == t for s in alive):
                fixed[c] = t
        return "<guess>" + "".join(final[c] for c in CIRCS) + "</guess>"
