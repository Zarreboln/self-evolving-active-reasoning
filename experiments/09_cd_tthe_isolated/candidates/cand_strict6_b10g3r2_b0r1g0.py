"""Branch b0r1g0 (parent: bare). Conservative repair of three failures visible in bare's traces:

  * the single free-form final reply is often not 24 bits, and every bare guess contradicted answers the
    environment had already given -> the final guess is asked per circuit (8 bits, observed rows shown,
    '?' for the rest); a table contradicting an observed answer is vetoed once with the offending rows
    named; observed rows are then kept as answered and the harness concatenates the three tables;
  * a reply without a query (premature guesses, a degenerate '000...' spill) cost a turn and stayed in
    the context, poisoning later turns -> it is dropped and the model re-asked; only queries spend budget;
  * C (or B) was often left with 0-2 queries -> the per-turn message also says how many queries each
    circuit has had so far.
Everything else is bare's protocol and wording."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
TABLE_RE = re.compile(r"<table>\s*([01](?:[\s,]*[01]){7})\s*</table>")
BITS8_RE = re.compile(r"(?<![01])[01]{8}(?![01])")
ROWS = [(a, b, x) for a in (0, 1) for b in (0, 1) for x in (0, 1)]
MAX_BAD = 8          # non-query replies tolerated in the whole episode before giving up the query phase


class Harness(CDHarness):
    def _counts(self):
        n = {c: 0 for c in "ABC"}
        for (c, *_), _bit in self.env.history:
            n[c] += 1
        return ", ".join(f"{c}: {n[c]}" for c in "ABC")

    def _parse_table(self, reply):
        m = TABLE_RE.findall(reply or "")
        if m:
            return re.sub(r"[^01]", "", m[-1])
        m = BITS8_RE.findall(reply or "")
        return m[-1] if m else None

    def _final_circuit(self, c, answers_txt):
        env = self.env
        obs = {(a, b, x): bit for (cc, a, b, x), bit in env.history if cc == c}
        grid = "\n".join(f"{c}({a}, {b}, {x}) = {obs.get((a, b, x), '?')}" for a, b, x in ROWS)
        ask = (f"All answers you received:\n{answers_txt}\n\n"
               f"Now write all 8 outputs of circuit {c} only. Rows you queried are fixed; "
               f"'?' rows were not queried -- deduce them from the answers and the gate rules.\n{grid}\n\n"
               f"Reply with the 8 outputs of {c} in row order {c}(0,0,0) ... {c}(1,1,1) as "
               f"<table>XXXXXXXX</table> (exactly 8 bits).")
        msgs = [{"role": "system", "content": env.system_prompt}, {"role": "user", "content": ask}]
        table = None
        for _ in range(2):
            reply = self.llm(msgs)
            t = self._parse_table(reply)
            if t is None:
                msgs += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Your reply did not contain <table>XXXXXXXX</table> with "
                          f"exactly 8 bits. Write circuit {c}'s 8 outputs in that format."}]
                continue
            table = t
            bad = [r for i, r in enumerate(ROWS) if r in obs and int(t[i]) != obs[r]]
            if not bad:
                break
            wrong = ", ".join(f"{c}{r} = {obs[r]}" for r in bad)
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"That table contradicts answers you received: {wrong}. "
                      f"Write circuit {c}'s 8 outputs again as <table>XXXXXXXX</table>, keeping every queried row."}]
        bits = list(table) if table else ["0"] * 8
        for i, r in enumerate(ROWS):           # an answered row is never guessed against its own answer
            if r in obs:
                bits[i] = str(obs[r])
        return "".join(bits)

    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        B = env.budget
        bad = 0
        while env.remaining > 0 and bad < MAX_BAD:
            reply = self.llm(msgs)
            m = QUERY_RE.search(reply or "")
            if not m:
                # drop the reply (it is not a query and tends to be copied later) and re-ask; no budget spent
                bad += 1
                last = msgs[-1]["content"].split("\n\n(Your previous reply")[0]
                msgs[-1] = {"role": "user", "content": last + f"\n\n(Your previous reply had no query; retry {bad}. "
                            "Do not give the final guess yet: reply with exactly one <query>X(a, b, c)</query>.)"}
                continue
            msgs.append({"role": "assistant", "content": reply})
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            line = env.format_answer(c, a, b, x, bit)
            left = env.remaining
            if left > 0:
                msgs.append({"role": "user", "content": f"{line}\nPlease send your next query. {left} queries "
                             f"remaining. Queries so far per circuit -- {self._counts()}."})
        answers_txt = "\n".join(env.format_answer(c, a, b, x, bit) for (c, a, b, x), bit in env.history) or "(none)"
        guess = "".join(self._final_circuit(c, answers_txt) for c in "ABC")
        return f"<guess>{guess}</guess>"
