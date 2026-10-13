"""Conservative repair of `bare`: same query loop and wording, plus two fixes for trace-visible
failures.
  1. Final guess: the model kept mistranscribing answers it had already seen (9/10 bare guesses
     contradicted observed rows; one was 22 bits long). The final turn now shows the model the
     24 rows in guess order (observed answers filled in, '?' for rows never queried) and asks it
     to fill only the '?' rows. The harness assembles the 24 bits: observed rows keep the
     environment's answer, '?' rows take the model's value.
  2. Wasted turns: a reply with no query, or one repeating an already-answered query, no longer
     spends the turn; the model is re-asked (a bounded number of times)."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
ROW_RE = re.compile(r"([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*(?:=|:|->)\s*([01])")
GUESS_RE = re.compile(r"<guess>\s*([01]{24})\s*</guess>")
ROWS = [(c, a, b, x) for c in "ABC" for a in (0, 1) for b in (0, 1) for x in (0, 1)]
MAX_RETRIES = 6          # re-asks per episode for missing / repeated queries


class Harness(CDHarness):
    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        last_line = None
        B = env.budget
        retries = 0
        while env.remaining > 0:
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            m = QUERY_RE.search(reply or "")
            if m:
                c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
                seen = self._observed()
                if (c, a, b, x) in seen and retries < MAX_RETRIES:
                    retries += 1
                    unasked = [f"{q[0]}({q[1]}, {q[2]}, {q[3]})" for q in ROWS if q not in seen]
                    msgs.append({"role": "user", "content":
                                 f"You already asked {c}({a}, {b}, {x}) and got "
                                 f"{env.format_answer(c, a, b, x, seen[(c, a, b, x)])}. Repeating it gives no "
                                 f"new information and this turn was not counted. Inputs not asked yet: "
                                 f"{', '.join(unasked)}.\nReply with exactly one new <query>X(a, b, c)</query>. "
                                 f"{env.remaining} queries remaining."})
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
            last_line = env.format_answer(c, a, b, x, bit)
            if env.remaining > 0:
                msgs.append({"role": "user", "content": f"{last_line}\nPlease send your next query. {env.remaining} queries remaining."})
        return self._final(msgs, last_line)

    def _observed(self):
        return {tuple(q): bit for q, bit in self.env.history}

    def _final(self, msgs, last_line):
        env = self.env
        seen = self._observed()
        unknown = [q for q in ROWS if q not in seen]
        table = "\n".join(f"{q[0]}({q[1]}, {q[2]}, {q[3]}) = {seen[q] if q in seen else '?'}" for q in ROWS)
        prompt = ((f"{last_line}\n" if last_line else "") +
                  "Here are all 24 rows in the submission order. Rows with a number are the answers you "
                  "observed (they are certain). Rows with '?' were never queried.\n\n" + table + "\n\n")
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
                if q in seen:
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
        bits = "".join(str(seen[q]) if q in seen else str(filled.get(q, 0)) for q in ROWS)
        return f"<guess>{bits}</guess>"
