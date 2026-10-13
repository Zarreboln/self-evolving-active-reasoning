"""Branch b1r1g2 (parent: b1r0g2). Cross-branch audit of batch b1 (same 10 instances, three branches): pooling
every answer any branch received pins exactly one rule-allowed triple per instance, and the parent's guess equals
it on all 10 (b0r1g1 misses i253 with a constant table, b1r0g0 misses i253 by picking 1 of 3 leftover triples and
crashes on i130). The parent's querying is at the one-step information optimum (its model picked the most
informative cell in ~95% of turns; simulating 1- and 2-step lookahead rankings gave no fewer queries), so no
change there. The parent's weak point is the one b1r0g0 crashed on, which the parent's loop shares: on i130 the
model declared the circuits solved with 3 cells still open, wrote a <guess>, and every format reminder was
appended to the conversation together with its reply, so it copied itself for 30 calls until the prompt exceeded
the context window.
Hypothesis: if a rejected reply is never left in the conversation (the latest message is re-asked with the reason,
e.g. "the board still has these unknown cells; queries continue"), the stop/echo loop cannot form, no episode
crashes on context length, and episodes keep the parent's query counts and correct guesses.

Changes vs parent (rule enumeration, board, odds display, vetoes and final phase are unchanged):
  * A reply without a query, or a vetoed query (known cell, predictable cell), is not appended. The next call
    re-sends the conversation with the latest message replaced by: its answer line, the reason, the board.
    A reply that tries to finish (<guess>, a 24-bit string, "no further queries") while '?' cells remain is
    told which cells are open and that the final answer is asked for later.
  * After FRESH_AFTER failed replies in a row, the re-ask drops the conversation and lists the answers so far;
    after REASK_MAX the query phase ends and the final phase runs on the board.
  * Accepted replies are kept only up to their query tag, at most KEEP_CHARS characters.
  * A context-length error from the model call is retried once with the system prompt and the latest message;
    any other error is raised as before.

Parent's notes (b1r0g2, parent: b0r1g1):
Branch b1r0g2 (parent: b0r1g1). Audit of the parent's 10 traces: the rule-derived board is sound (no answer
ever contradicted it; 9/10 episodes end with exactly one rule-allowed triple, and the guess equals it), but the
model ignores it while querying. It sweeps each circuit row by row: 12 of 128 queries asked a cell the board had
already marked deduced (zero information), and others asked cells whose answer was ~97% predictable. On the
hardest instance this used all 18 queries and left 3 triples, and in the final phase the model then wrote a
constant table (forbidden by the rules) four times, and the fallback kept it.
Hypothesis: when the model is shown only what is still unknown and how predictable each unknown cell is, and is
not charged for asking a known cell, it resolves every circuit well within 18 queries. The final-phase fixes
remove the remaining way an unresolved episode can end up with a table the rules forbid.

Changes vs parent (the rule enumeration and board semantics are unchanged):
  * Each turn shows a compact board (one row per circuit, '?' for unknown) and lists the unknown cells with
    the share of still-allowed circuit triples in which that cell is 1, and the few cells closest to 50%.
    Older turns keep only their answer lines, so the model is not re-reading 18 stale boards.
  * A query on an observed or deduced cell is not sent to the environment: the harness replies with the
    known value ("no query spent") and asks again.
  * A query whose answer is at least 90% predictable, while some unknown cell is between 30% and 70%, is sent
    back once with that fact; if the model repeats it, it is executed.
  * Final phase: when the rules leave few tables for a circuit, they are listed and the model picks one; a
    reply that is not allowed gets the list again. The last-resort fallback is the first allowed table that
    matches what the model wrote, otherwise the first one listed, never a table the rules forbid.
"""

import itertools
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
TABLE_RE = re.compile(r"<table>\s*([01](?:\s*[01]){7})\s*</table>")
BITS8_RE = re.compile(r"(?<![01])[01]{8}(?![01])")
ROWS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
CIRCS = "ABC"
GUESS_RE = re.compile(r"<guess>|(?<![01])[01]{24}(?![01])")
STOP_RE = re.compile(r"no (?:further|more|additional) quer|final (?:submission|answer|guess)", re.I)
MAX_CALLS = 70
FINAL_TRIES = 4
REASK_MAX = 8          # consecutive replies without a usable query before the query phase gives up
FRESH_AFTER = 3        # from this many, the re-ask drops the conversation and lists the answers instead
KEEP_CHARS = 600       # an accepted reply is kept in the conversation up to its query, at most this long


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


def _row_name(c, r):
    return f"{c}({r >> 2},{(r >> 1) & 1},{r & 1})"


def _compact(reply, matches):
    """The accepted reply as kept in the conversation: cut after its last query, at most KEEP_CHARS."""
    end = reply.rfind("</query>") + len("</query>") if matches else len(reply)
    text = reply[:end]
    return text if len(text) <= KEEP_CHARS else "..." + text[-KEEP_CHARS:]


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
        try:
            return self.llm(msgs, **kw) or ""
        except Exception as e:                       # only a too-long prompt is recovered; anything else is raised
            if len(msgs) <= 2 or not re.search(r"context|too long|maximum|token", str(e), re.I):
                raise
        self.calls += 1
        return self.llm([msgs[0], msgs[-1]], **kw) or ""

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

    def _odds(self, board, alive):
        """{(c, row): share of alive triples with output 1} for every open cell (empty if no rule model)."""
        if not alive:
            return {}
        return {(c, r): sum((s[i] >> r) & 1 for s in alive) / len(alive)
                for i, c in enumerate(CIRCS) for r in range(8) if board[c][r][0] is None}

    def _board_text(self, board, alive):
        lines = ["KNOWLEDGE BOARD, rows in the order (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) (1,0,1) (1,1,0) (1,1,1). "
                 "0/1 = already known (answered by a query, or forced by your gate inventory, the building rules and "
                 "all answers so far); ? = still unknown:"]
        for c in CIRCS:
            row = " ".join("?" if board[c][r][0] is None else str(board[c][r][0]) for r in range(8))
            n_open = sum(1 for r in range(8) if board[c][r][0] is None)
            lines.append(f"Circuit {c}: {row}   ({'done' if n_open == 0 else f'{n_open} unknown'})")
        odds = self._odds(board, alive)
        if odds:
            lines.append(f"{len(alive)} combinations of circuits A, B, C still fit the rules and the answers. "
                         "Unknown cells, with the share of those combinations in which the output is 1:")
            for c in CIRCS:
                cells = [f"{_row_name(c, r)} {round(100 * odds[(c, r)])}%" for r in range(8) if (c, r) in odds]
                if cells:
                    lines.append(f"  {c}: " + ", ".join(cells))
            near = sorted(odds, key=lambda k: (abs(odds[k] - 0.5), k))[:3]
            lines.append("A cell near 50% tells you the most; near 0% or 100% its answer is almost certain. "
                         "Closest to 50% now: " + ", ".join(_row_name(c, r) for c, r in near) + ".")
        else:
            open_cells = [_row_name(c, r) for c in CIRCS for r in range(8) if board[c][r][0] is None]
            lines.append("Unknown cells: " + ", ".join(open_cells) + ".")
        lines.append("Query only an unknown (?) cell.")
        return "\n".join(lines)

    def _all_known(self, board):
        return all(board[k][r][0] is not None for k in CIRCS for r in range(8))

    # ---- query phase: bare's conversation; the latest message carries the board ----
    def _query_phase(self):
        env = self.env
        board, alive = self._board()
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": self._board_text(board, alive) + "\n\nPlease start with your first query."}]
        head = "Please start with your first query."   # short form of the latest user message
        warned, note, fails = None, "", 0
        while env.remaining > 0 and self.calls < MAX_CALLS - 3 * FINAL_TRIES and fails < REASK_MAX:
            req = msgs
            if note:                                         # rejected reply: re-ask, nothing is appended
                ask = (f"{note}\n\n{self._board_text(board, alive)}\n\n"
                       f"Please send your next query, on a ? cell. {env.remaining} queries remaining.")
                if fails >= FRESH_AFTER:                     # stuck: drop the conversation it keeps echoing
                    done = "\n".join("- " + env.format_answer(*q, bit) for q, bit in env.history)
                    req = [msgs[0], {"role": "user", "content": (f"Answers so far:\n{done}\n\n" if done else "") + ask}]
                else:
                    req = msgs[:-1] + [{"role": "user", "content": f"{head}\n\n{ask}"}]
            reply = self._call(req)
            m = QUERY_RE.findall(reply)
            if not m:
                fails += 1
                unknown = [_row_name(k, r) for k in CIRCS for r in range(8) if board[k][r][0] is None]
                if GUESS_RE.search(reply) or STOP_RE.search(reply):
                    note = (f"(Re-ask {fails}) Not accepted: your reply tried to finish, but the board below still has "
                            f"{len(unknown)} unknown cell(s): {', '.join(unknown[:6])}{' ...' if len(unknown) > 6 else ''}. "
                            f"The circuits are not determined yet and {env.remaining} queries remain, so the queries "
                            f"continue; the final answer is asked for later. Reply with exactly one "
                            f"<query>X(a, b, c)</query> on a ? cell.")
                else:
                    note = (f"(Re-ask {fails}) Your last reply contained no query in the required format. "
                            f"Reply with exactly one <query>X(a, b, c)</query> on a ? cell.")
                continue
            fails = 0
            c, a, b, x = m[-1][0].upper(), int(m[-1][1]), int(m[-1][2]), int(m[-1][3])
            r = _row(a, b, x)
            v, kind = board[c][r]
            if v is not None:                                # known cell: answer from the board, no query spent
                note = (f"Your last reply asked {_row_name(c, r)}, which is already known: it is {v} "
                        f"({'you queried it' if kind == 'observed' else 'forced by the rules and earlier answers'}). "
                        f"No query was spent.")
                continue
            odds = self._odds(board, alive)
            if odds and (c, r) != warned:
                p = odds[(c, r)]
                better = [k for k, q in odds.items() if 0.3 <= q <= 0.7]
                if (p <= 0.1 or p >= 0.9) and better:
                    warned = (c, r)
                    likely = 1 if p >= 0.9 else 0
                    note = (f"Not sent yet: your last reply asked {_row_name(c, r)}, which is {likely} in "
                            f"{round(100 * max(p, 1 - p))}% of the combinations still possible, so this query would most "
                            f"likely teach you nothing. Cells near 50% are much more informative. Send this same query "
                            f"again if you still want it.")
                    continue
            warned, note = None, ""
            bit = env.query(c, a, b, x)
            board, alive = self._board()
            if self._all_known(board):
                break
            ans = env.format_answer(c, a, b, x, bit)
            msgs[-1] = {"role": "user", "content": head}     # the board in it is outdated now
            msgs.append({"role": "assistant", "content": _compact(reply, m)})
            msgs.append({"role": "user", "content": f"{ans}\n\n{self._board_text(board, alive)}\n\n"
                         f"Please send your next query. {env.remaining} queries remaining."})
            head = ans

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
            counts = {}
            for s in alive:
                counts[s[i]] = counts.get(s[i], 0) + 1
            options = [_bits(t) for t in sorted(counts, key=lambda t: (-counts[t], t))]
            listing = "\n".join(f"{c}({a},{b},{x}) = " + ("?" if cells[_row(a, b, x)][0] is None else
                                                         f"{cells[_row(a, b, x)][0]} ({cells[_row(a, b, x)][1]})")
                                for a, b, x in ROWS)
            opt_text = ""
            if len(known) < 8 and 1 < len(options) <= 8:
                opt_text = (f"\nThe rules and all answers allow only these tables for circuit {c}: "
                            + ", ".join(options) + ". Pick the one you find most likely.")
            ask = (f"The queries are over. Circuit {c} on the knowledge board (rows 000..111):\n{listing}\n\n"
                   f"Write circuit {c}'s 8 outputs in the order {c}(0,0,0), {c}(0,0,1), ..., {c}(1,1,1), "
                   f"keeping every observed and deduced value"
                   + (", and replacing each '?' with your best deduction from the gates" if len(known) < 8 else "")
                   + f". Board row: {template}.{opt_text} Answer as <table>XXXXXXXX</table>.")
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
                            + ", ".join(_row_name(c, r) for r in bad)
                            + ". Keep every observed and deduced value.)")
                    continue
                if use_rules and alive and t not in counts:
                    note = (f"\n(Your table {bits} cannot be produced by any circuit the rules allow together "
                            f"with the other circuits."
                            + (f" The allowed tables are: {', '.join(options[:8])}. Copy one of them." if options else
                               " Change some '?' rows.") + ")")
                    continue
                chosen = bits
                break
            if chosen is None:
                ok = [s for s in said if sum(int(ch) << r for r, ch in enumerate(s)) in counts]
                chosen = ok[0] if ok else (options[0] if options else
                                           next((s for s in said if all(int(s[r]) == v for r, v in known.items())),
                                                template.replace("?", "0")))
            final[c] = chosen
            t = sum(int(ch) << r for r, ch in enumerate(chosen))
            if use_rules and t in counts:
                fixed[c] = t
        return "<guess>" + "".join(final[c] for c in CIRCS) + "</guess>"
