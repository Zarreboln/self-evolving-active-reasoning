"""Deduction-board harness. The frozen model still writes every query and the final guess; the
harness does the bookkeeping a 7B model gets wrong.

From the task rules alone (inventory; each circuit turns 3 inputs into 1 signal, so it holds
exactly two AND/OR gates, NOTs anywhere; no constant circuit; all parts used, no replacement)
the harness enumerates every joint (A, B, C) table, and after each answer keeps only
those consistent with every answer so far. Each turn the model sees a compact board: observed
rows, rows already forced by the rules, and open rows with how many joint tables remain under
each outcome. A query on a row that is already known or forced is vetoed and re-asked (free).
The episode ends early once a single joint table remains.

b0r1g2 (audit of b0r0g2 traces): the model took the first suggested row only ~60% of the time;
offline, a player picking any open row leaves >1 table after 18 queries in 2-18% of episodes for
the larger rule spaces, while one restricted to the best-balanced rows never does. So the prompt
now names every equally-best row, and a query whose split is worse than the best (smaller side
below BALANCE_TAU x the best) is vetoed and re-asked; the model still picks which best row. The final guess is written by the
model from the board; a guess that contradicts an answer or the rules is vetoed, and as a last
resort it is repaired to the nearest consistent table.

b1r0g1 (audit of b0r1g2 traces): the one shaky episode failed at TRANSCRIPTION, not deduction --
the board said A = 10000000, B = 00000001 and the model wrote 32 bits four times running (it
miscounts runs of equal bits when it has to concatenate), each identical retry got the identical
reply, and only the nearest-table repair saved it. The final answer is now written as three
separate 8-bit fields <A>..</A> <B>..</B> <C>..</C>, each checked on its own, and the harness
concatenates them in the benchmark's order; a retry names the faulty field and its bit count.

b1r1g1 (same audit, different cause): the solver is deterministic for a given context (seeded,
cached), so a re-ask that repeats the context returns the rejected reply verbatim -- i228's four
final tries were one try, and a query veto whose note repeats is a fixed point too; the retry
budget was effectively 1. Retries now happen IN the conversation: the rejected reply stays as the
model's own turn and a specific correction follows (what was wrong, the open best rows / the
fields to fix), so every retry has a new context and the model sees its own mistake. First asks
are unchanged.
"""

import itertools
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"<guess>\s*([01\s]+?)\s*</guess>")
FIELD_RE = {c: re.compile(rf"<{c}>\s*([01\s]*?)\s*</{c}>", re.I) for c in "ABC"}
ROWS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
LABELS = "ABC"


def _op(g, x, y):
    return (x & y) if g == "AND" else (x | y)


def circuit_tables(n_and, n_or, n_not):
    """Non-constant tables of one circuit built from exactly these gates (n_and + n_or == 2)."""
    out = set()
    ops = ["AND"] * n_and + ["OR"] * n_or
    for g1, g2 in set(itertools.permutations(ops)):
        for pair in ((0, 1), (0, 2), (1, 2)):
            third = ({0, 1, 2} - set(pair)).pop()
            # NOTs on the 3 inputs, the inner gate's output, the final output (stacking allowed)
            for dist in itertools.product(range(n_not + 1), repeat=5):
                if sum(dist) != n_not:
                    continue
                p = [d % 2 for d in dist]
                t = []
                for v in ROWS:
                    w = [v[i] ^ p[i] for i in range(3)]
                    m = _op(g1, w[pair[0]], w[pair[1]]) ^ p[3]
                    t.append(_op(g2, m, w[third]) ^ p[4])
                t = tuple(t)
                if len(set(t)) > 1:
                    out.add(t)
    return out


def joint_tables(inv):
    n_and, n_or, n_not = inv.get("AND", 0), inv.get("OR", 0), inv.get("NOT", 0)
    if n_and + n_or != 6:
        return []          # outside what the rules describe: leave the model on its own
    cache, triples = {}, set()
    for aA in range(3):
        for aB in range(3):
            aC = n_and - aA - aB
            if not 0 <= aC <= 2:
                continue
            for nA in range(n_not + 1):
                for nB in range(n_not + 1 - nA):
                    nC = n_not - nA - nB
                    sets = []
                    for k in ((aA, nA), (aB, nB), (aC, nC)):
                        if k not in cache:
                            cache[k] = circuit_tables(k[0], 2 - k[0], k[1])
                        sets.append(cache[k])
                    triples.update(itertools.product(*sets))
    return [ta + tb + tc for ta, tb, tc in triples]       # 24-tuples, A rows then B then C


def bit_index(c, a, b, x):
    return 8 * LABELS.index(c) + ROWS.index((a, b, x))


def row_name(i):
    return f"{LABELS[i // 8]}({', '.join(map(str, ROWS[i % 8]))})"


class Harness(CDHarness):
    MAX_RETRIES = 4
    BALANCE_TAU = 1.0       # accept only rows whose smaller side is >= this x the best one

    def best_rows(self, open_rows):
        if not open_rows:
            return []
        top = open_rows[0][0]
        return [i for s, i, _, _ in open_rows if s >= self.BALANCE_TAU * top]

    def board(self, alive, observed):
        """Text board + the set of open (still undetermined) bit indices."""
        n = len(alive)
        lines, open_rows = [], []
        for ci, c in enumerate(LABELS):
            cells = []
            for r in range(8):
                i = 8 * ci + r
                ones = sum(t[i] for t in alive)
                if i in observed:
                    cells.append(f"  {row_name(i)} = {observed[i]}   (observed)")
                elif ones in (0, n):
                    cells.append(f"  {row_name(i)} = {1 if ones else 0}   (forced by the rules + answers; do not query)")
                else:
                    open_rows.append((min(ones, n - ones), i, n - ones, ones))
                    cells.append(f"  {row_name(i)} = ?   (open: if 0 -> {n - ones} tables left, if 1 -> {ones} left)")
            known = "".join(str(observed[8 * ci + r]) if 8 * ci + r in observed else
                            (str(alive[0][8 * ci + r]) if sum(t[8 * ci + r] for t in alive) in (0, n) else "?")
                            for r in range(8))
            lines.append(f"Circuit {c}: rows 000..111 = {known}")
            lines.extend(cells)
        open_rows.sort(reverse=True)
        return "\n".join(lines), open_rows

    def turn_prompt(self, alive, observed, remaining, note=""):
        text, open_rows = self.board(alive, observed)
        best = ", ".join(row_name(i) for i in self.best_rows(open_rows)[:8])
        return (f"{note}Answers so far:\n" +
                ("\n".join(f"- {self.env.format_answer(c, a, b, x, bit)}" for (c, a, b, x), bit in self.env.history)
                 or "- (none yet)") +
                f"\n\nDEDUCTION BOARD ({len(alive)} joint tables of A, B, C still fit the rules and "
                f"every answer):\n{text}\n\n"
                f"{remaining} queries remaining. Query ONE open row (marked ?). Rows whose two outcomes "
                f"leave similar numbers of tables are the most informative. The most balanced rows now "
                f"(equally good, pick any one of them): {best}. "
                f"Think in at most two sentences, then reply with exactly one <query>X(a, b, c)</query>.")

    def final_prompt(self, alive, observed, note=""):
        text, _ = self.board(alive, observed)
        if len(alive) == 1:
            t = alive[0]
            head = ("The answers and the rules now determine every circuit exactly:\n"
                    f"A = {''.join(map(str, t[:8]))}, B = {''.join(map(str, t[8:16]))}, C = {''.join(map(str, t[16:]))}\n")
        else:
            opts = "\n".join(f"  option {k + 1}: A={''.join(map(str, t[:8]))} B={''.join(map(str, t[8:16]))} "
                             f"C={''.join(map(str, t[16:]))}" for k, t in enumerate(alive[:6]))
            head = f"{len(alive)} joint tables still fit every answer, e.g.:\n{opts}\nPick one of them.\n"
        return (f"{note}Queries are over. DEDUCTION BOARD:\n{text}\n\n{head}"
                "Write the final answer as three separate fields, one per circuit, each exactly 8 bits "
                "(rows 000..111, no spaces), copying each circuit's line above:\n"
                "<A>8 bits of A</A> <B>8 bits of B</B> <C>8 bits of C</C>\n"
                "Reply with only these three fields.")

    @staticmethod
    def parse_fields(reply):
        """24 bits from the three per-circuit fields (or a whole <guess>), plus what was wrong."""
        parts, bad = [], []
        for c in LABELS:
            ms = FIELD_RE[c].findall(reply or "")
            bits = re.sub(r"\s", "", ms[-1]) if ms else ""
            if len(bits) != 8:
                bad.append(f"field <{c}> had {len(bits)} bits" if ms else f"field <{c}> was missing")
            parts.append(bits)
        if not bad:
            return "".join(parts), ""
        gs = GUESS_RE.findall(reply or "")
        whole = re.sub(r"\s", "", gs[-1]) if gs else ""
        if len(whole) == 24:
            return whole, ""
        if any(parts) or not whole:      # keep each circuit on its own 8 rows for the repair
            return ("".join(b[:8].ljust(8, "0") for b in parts) if any(parts) else None), "; ".join(bad)
        return whole, "; ".join(bad)

    @staticmethod
    def correct(convo, reply, correction, ask):
        """Re-ask IN the conversation: the rejected reply stays as the model's own turn and the
        correction follows it, so no retry repeats an earlier context (an identical context gets
        the identical reply back)."""
        convo.append({"role": "assistant", "content": reply or "(no reply)"})
        convo.append({"role": "user", "content": f"{correction}{ask}"})

    def solve(self):
        env = self.env
        alive = joint_tables(env.inventory or {})
        observed = {}
        sysmsg = {"role": "system", "content": env.system_prompt}
        ask_q = "Reply with exactly one <query>X(a, b, c)</query>."
        while env.remaining > 0 and len(alive) != 1:
            accepted = None
            last_parsed = None
            last_open = None        # the model's last query that is at least still informative
            if alive:
                user = self.turn_prompt(alive, observed, env.remaining)
            else:
                user = ("\n".join(env.format_answer(c, a, b, x, bit) for (c, a, b, x), bit in env.history)
                        + f"\n{env.remaining} queries remaining. {ask_q}")
            convo = [sysmsg, {"role": "user", "content": user}]
            for _ in range(self.MAX_RETRIES):
                reply = self.llm(convo, max_tokens=400)
                m = QUERY_RE.search(reply or "")
                if not m:
                    self.correct(convo, reply, "Your last reply had no <query>X(a, b, c)</query>. ", ask_q)
                    continue
                q = (m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                last_parsed = q
                i = bit_index(*q)
                n = len(alive)
                _, open_rows = self.board(alive, observed)
                best = self.best_rows(open_rows)
                pick = (f" Pick one of these open rows instead: {', '.join(row_name(k) for k in best[:8])}. "
                        if best else " ")
                if i in observed:
                    self.correct(convo, reply, f"{row_name(i)} was already observed (= {observed[i]}), so asking "
                                 f"it again teaches nothing.{pick}", ask_q)
                    continue
                if alive and sum(t[i] for t in alive) in (0, n):
                    self.correct(convo, reply, f"{row_name(i)} is already forced by the rules + answers "
                                 f"(= {alive[0][i]}), so asking it teaches nothing.{pick}", ask_q)
                    continue
                last_open = q
                if alive and i not in best:
                    ones = sum(t[i] for t in alive)
                    self.correct(convo, reply, f"{row_name(i)} is a weak query (outcomes leave {n - ones} or "
                                 f"{ones} tables).{pick}", ask_q)
                    continue
                accepted = q
                break
            if accepted is None:
                accepted = last_open or last_parsed     # the model's own query, even if it teaches less
            if accepted is None:
                break                           # the model never produced a query: stop asking
            bit = env.query(*accepted)
            i = bit_index(*accepted)
            observed[i] = bit
            alive = [t for t in alive if t[i] == bit]

        if not alive:      # rules not understood (should not happen): the benchmark's own final turn
            hist = "\n".join(env.format_answer(c, a, b, x, bit) for (c, a, b, x), bit in env.history)
            return self.llm([sysmsg, {"role": "user", "content": hist + "\n" + env.last_turn_prompt}])

        guess = None
        alive_set = set(alive)
        ask_g = "Reply with only the three fields <A>8 bits</A> <B>8 bits</B> <C>8 bits</C>."
        convo = [sysmsg, {"role": "user", "content": self.final_prompt(alive, observed)}]
        for k in range(self.MAX_RETRIES):
            reply = self.llm(convo, temperature=0.2 if k else 0.7, max_tokens=200)
            bits, err = self.parse_fields(reply)
            if err:
                self.correct(convo, reply, f"Your answer was malformed: {err}. Each field needs exactly 8 bits; "
                             "copy each circuit's 8-bit line from the board one field at a time. ", ask_g)
                guess = guess or bits
                continue
            guess = bits
            t = tuple(int(ch) for ch in bits)
            if t in alive_set:
                return f"<guess>{bits}</guess>"
            bad = [row_name(j) for j in range(24) if j in observed and t[j] != observed[j]]
            wrong = [c for ci, c in enumerate(LABELS) if not any(u[8 * ci:8 * ci + 8] == t[8 * ci:8 * ci + 8]
                                                               for u in alive)]
            self.correct(convo, reply,
                         ("Your guess contradicts the answers at " + ", ".join(bad[:4]) + ". " if bad else
                          "Your guess is not one of the tables that fit the rules and answers. ") +
                         (f"Fields to fix: {', '.join('<' + c + '>' for c in wrong)}. " if wrong else ""), ask_g)
        # repair: the consistent table nearest to what the model last wrote
        t = tuple(int(ch) for ch in (guess or "").ljust(24, "0")[:24])
        best = min(alive, key=lambda u: sum(x != y for x, y in zip(u, t)))
        return f"<guess>{''.join(map(str, best))}</guess>"
