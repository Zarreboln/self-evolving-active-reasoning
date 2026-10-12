"""Conservative repair of `bare`: same relay loop and benchmark wording, plus
  * each turn prompt also shows how many queries each circuit has had and that the budget
    cannot cover all 24 rows (bare traces: A and B swept row-by-row, C left with 2 queries),
  * the final-guess turn lists every observed answer grouped by circuit,
  * the final guess is checked against the observed answers; on a missing/contradicting guess
    the model is re-asked once, and observed cells are then taken from the answers."""

import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
GUESS_RE = re.compile(r"<guess>\s*([01\s]{24,}?)\s*</guess>")
BITS_RE = re.compile(r"[01]{24}")


def _cell(c, a, b, x):
    return "ABC".index(c) * 8 + a * 4 + b * 2 + x


class Harness(CDHarness):
    def _counts(self):
        n = {c: 0 for c in "ABC"}
        for (c, *_), _bit in self.env.history:
            n[c] += 1
        return n

    def _ledger(self):
        lines = []
        for c in "ABC":
            obs = [self.env.format_answer(*q, bit) for q, bit in self.env.history if q[0] == c]
            lines.append(f"Circuit {c}: " + ("; ".join(obs) if obs else "no queries"))
        return "All answers so far:\n" + "\n".join(lines)

    def _coverage_note(self, left):
        n = self._counts()
        unq = [c for c in "ABC" if n[c] == 0]
        note = (f"Queries per circuit so far: A={n['A']}, B={n['B']}, C={n['C']}. "
                f"{left} queries cannot read every remaining row, so use the gate inventory to infer "
                f"rows you do not query")
        return note + (f"; circuit(s) {', '.join(unq)} not queried yet." if unq else ".")

    def _parse(self, text):
        t = text or ""
        m = GUESS_RE.findall(t)
        src = re.sub(r"\s", "", m[-1]) if m else re.sub(r"\s", "", t)
        bits = BITS_RE.findall(src)
        return bits[-1] if bits else None

    def _conflicts(self, guess):
        out = []
        for (c, a, b, x), bit in self.env.history:
            if int(guess[_cell(c, a, b, x)]) != bit:
                out.append(self.env.format_answer(c, a, b, x, bit))
        return out

    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        last_line = None
        B = env.budget
        for t in range(1, B + 1):
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            m = QUERY_RE.search(reply or "")
            if not m:
                # the benchmark gives no second chance: the turn is spent
                msgs.append({"role": "user", "content": "Your message did not contain a query in the "
                             "required format. Reply with exactly one <query>X(a, b, c)</query>."})
                if t == B:
                    last_line = None      # the previous answer was already delivered with its own prompt
                continue
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            last_line = env.format_answer(c, a, b, x, bit)
            if t < B:
                msgs.append({"role": "user", "content": f"{last_line}\nPlease send your next query. "
                             f"{B - t} queries remaining.\n{self._coverage_note(B - t)}"})
        final = (f"{last_line}\n" if last_line else "") + self._ledger() + "\n" + env.last_turn_prompt
        msgs.append({"role": "user", "content": final})
        reply = self.llm(msgs)
        guess = self._parse(reply)
        bad = self._conflicts(guess) if guess else None
        if guess is None or bad:
            msgs.append({"role": "assistant", "content": reply})
            why = ("Your guess did not contain a 24-bit table." if guess is None else
                   "Your guess contradicts these observed answers: " + "; ".join(bad) + ".")
            msgs.append({"role": "user", "content": why + "\n" + env.last_turn_prompt})
            reply2 = self.llm(msgs)
            g2 = self._parse(reply2)
            if g2 is not None or guess is None:
                guess, reply = g2, reply2
        if guess is None:
            return reply
        bits = list(guess)
        for (c, a, b, x), bit in env.history:      # observed cells are facts, not guesses
            bits[_cell(c, a, b, x)] = str(bit)
        return "<guess>" + "".join(bits) + "</guess>"
