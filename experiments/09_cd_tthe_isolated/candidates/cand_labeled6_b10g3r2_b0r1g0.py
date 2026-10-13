"""Conservative repair of `bare`: same query loop and wording, plus fixes for trace-visible
failures.
  1. Final guess: the model kept mistranscribing answers it had already seen (9/10 bare guesses
     contradicted observed rows; one was 22 bits long). The final turn now shows the model the
     24 rows in guess order (observed answers filled in, '?' for rows never queried) and asks it
     to fill only the '?' rows. The harness assembles the 24 bits: observed rows keep the
     environment's answer, '?' rows take the model's value.
  2. Wasted turns: a reply with no query, or one repeating an already-answered query, no longer
     spends the turn; the model is re-asked (a bounded number of times).
  3. Rows the rules already decide (b0r1): the model asked rows in order (A, then B, then C), ran
     out with ~6 rows unasked and guessed them wrong in 10/10 episodes. From the task rules alone
     (each circuit: 3 inputs -> 1 output, so exactly two AND/OR gates, NOTs anywhere; inventory
     used exactly; no constant circuit) the harness keeps the joint tables consistent with every
     answer. A query on a row those tables all agree on is vetoed like a repeat; querying stops
     once no row is open; the final table shows such rows with their value instead of '?'."""

import itertools
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
ROW_RE = re.compile(r"([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*(?:=|:|->)\s*([01])")
GUESS_RE = re.compile(r"<guess>\s*([01]{24})\s*</guess>")
ROWS = [(c, a, b, x) for c in "ABC" for a in (0, 1) for b in (0, 1) for x in (0, 1)]
MAX_RETRIES = 6          # re-asks per episode for missing / repeated queries
MAX_FORCED_RETRIES = 3   # re-asks per turn for a query on a row the rules already decide


def _circuit_tables(n_and, n_not):
    """Non-constant 8-row tables of one circuit with n_and ANDs, 2 - n_and ORs, n_not NOTs."""
    out = set()
    for g1, g2 in set(itertools.permutations(["AND"] * n_and + ["OR"] * (2 - n_and))):
        for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            # NOT parity on the 3 inputs, the first gate's output, the final output
            for dist in itertools.product(range(n_not + 1), repeat=5):
                if sum(dist) != n_not:
                    continue
                p = [d % 2 for d in dist]
                t = []
                for v in itertools.product((0, 1), repeat=3):
                    w = [v[n] ^ p[n] for n in range(3)]
                    m = ((w[i] & w[j]) if g1 == "AND" else (w[i] | w[j])) ^ p[3]
                    t.append(((m & w[k]) if g2 == "AND" else (m | w[k])) ^ p[4])
                if len(set(t)) > 1:
                    out.add(tuple(t))
    return out


def joint_tables(inv):
    """Every 24-bit (A, B, C) table the inventory allows; [] if the inventory is outside the rules."""
    n_and, n_or, n_not = inv.get("AND", 0), inv.get("OR", 0), inv.get("NOT", 0)
    if n_and + n_or != 6:
        return []
    cache, out = {}, set()
    for aA in range(3):
        for aB in range(3):
            aC = n_and - aA - aB
            if not 0 <= aC <= 2:
                continue
            for nA in range(n_not + 1):
                for nB in range(n_not + 1 - nA):
                    keys = ((aA, nA), (aB, nB), (aC, n_not - nA - nB))
                    for key in keys:
                        if key not in cache:
                            cache[key] = _circuit_tables(*key)
                    out.update(ta + tb + tc for ta, tb, tc in itertools.product(*(cache[k] for k in keys)))
    return list(out)


class Harness(CDHarness):
    def solve(self):
        env = self.env
        self.alive = joint_tables(env.inventory or {})
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        last_line = None
        B = env.budget
        retries = 0
        forced_retries = 0
        while env.remaining > 0:
            if self.alive and not self._open(self._observed()):
                break                    # every row is observed or decided by the rules
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            m = QUERY_RE.search(reply or "")
            if m:
                c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
                seen = self._observed()
                forced = self._forced(seen)
                unasked = [f"{q[0]}({q[1]}, {q[2]}, {q[3]})" for q in self._open(seen)]
                if (c, a, b, x) in seen and retries < MAX_RETRIES:
                    retries += 1
                    msgs.append({"role": "user", "content":
                                 f"You already asked {c}({a}, {b}, {x}) and got "
                                 f"{env.format_answer(c, a, b, x, seen[(c, a, b, x)])}. Repeating it gives no "
                                 f"new information and this turn was not counted. Inputs not asked yet: "
                                 f"{', '.join(unasked)}.\nReply with exactly one new <query>X(a, b, c)</query>. "
                                 f"{env.remaining} queries remaining."})
                    continue
                if (c, a, b, x) in forced and forced_retries < MAX_FORCED_RETRIES:
                    forced_retries += 1
                    msgs.append({"role": "user", "content":
                                 f"{c}({a}, {b}, {x}) is already decided: the circuit rules, the inventory and "
                                 f"your answers so far leave only {env.format_answer(c, a, b, x, forced[(c, a, b, x)])}. "
                                 f"Asking it gives no new information and this turn was not counted. Inputs still "
                                 f"open: {', '.join(unasked)}.\nReply with exactly one <query>X(a, b, c)</query> "
                                 f"on an open input. {env.remaining} queries remaining."})
                    continue
            else:
                if retries < MAX_RETRIES:
                    retries += 1
                    msgs.append({"role": "user", "content": "Your message did not contain a query in the "
                                 "required format. This turn was not counted. Reply with exactly one "
                                 f"<query>X(a, b, c)</query>. {env.remaining} queries remaining."})
                    continue
                break
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            forced_retries = 0
            self.alive = [t for t in self.alive if t[ROWS.index((c, a, b, x))] == bit]   # [] -> stop using the rules
            last_line = env.format_answer(c, a, b, x, bit)
            if env.remaining > 0:
                msgs.append({"role": "user", "content": f"{last_line}\nPlease send your next query. {env.remaining} queries remaining."})
        return self._final(msgs, last_line)

    def _observed(self):
        return {tuple(q): bit for q, bit in self.env.history}

    def _forced(self, seen):
        """Unobserved rows on which every table still consistent with the rules agrees."""
        if not self.alive:
            return {}
        out = {}
        for i, q in enumerate(ROWS):
            if q in seen:
                continue
            ones = sum(t[i] for t in self.alive)
            if ones in (0, len(self.alive)):
                out[q] = 1 if ones else 0
        return out

    def _open(self, seen):
        forced = self._forced(seen)
        return [q for q in ROWS if q not in seen and q not in forced]

    def _final(self, msgs, last_line):
        env = self.env
        seen = self._observed()
        forced = self._forced(seen)
        known = {**forced, **seen}
        unknown = [q for q in ROWS if q not in known]
        table = "\n".join(f"{q[0]}({q[1]}, {q[2]}, {q[3]}) = {known[q] if q in known else '?'}" for q in ROWS)
        prompt = ((f"{last_line}\n" if last_line else "") +
                  "Here are all 24 rows in the submission order. Rows with a number are certain (observed "
                  "answers, or rows the circuit rules and your answers leave only one value for). Rows with "
                  "'?' are still open.\n\n" + table + "\n\n")
        if unknown:
            prompt += ("Using the inventory and the circuit rules, decide the value of every '?' row. "
                       "Reason briefly about which gates each circuit can use, then list each '?' row "
                       "with its value, one per line, in the form `B(1, 1, 1) = 0`. Finally give the full "
                       "table in <guess> </guess> markers (24 bits, A then B then C).")
        else:
            prompt += "Give the full table in <guess> </guess> markers (24 bits, A then B then C)."
        msgs.append({"role": "user", "content": prompt})
        filled = {}
        reply = ""
        for _ in range(2):
            reply = self.llm(msgs) or ""
            g = GUESS_RE.findall(reply)
            rows = {}
            for c, a, b, x, v in ROW_RE.findall(reply):
                rows[(c.upper(), int(a), int(b), int(x))] = int(v)    # last mention wins
            for i, q in enumerate(ROWS):
                if q in known:
                    continue
                if q in rows:
                    filled[q] = rows[q]
                elif g and q not in filled:
                    filled[q] = int(g[-1][i])
            if all(q in filled for q in unknown):
                break
            missing = [f"{q[0]}({q[1]}, {q[2]}, {q[3]})" for q in unknown if q not in filled]
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": "Please give a value (0 or 1) for these rows, one per "
                         "line, in the form `B(1, 1, 1) = 0`: " + ", ".join(missing)})
        bits = "".join(str(known[q]) if q in known else str(filled.get(q, 0)) for q in ROWS)
        return f"<guess>{bits}</guess>"
