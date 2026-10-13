"""Branch b0r0g1 (parent: bare). Hypothesis: episodes fail at ANSWER PRODUCTION, not at querying --
the 18 answers often already pin the circuits, but the 7B cannot emit a correct 24-bit string in one
free-form reply (wrong lengths, rows contradicting its own observations, hallucinated rows).

One mechanism: the query phase is bare's, unchanged. The single free-form final guess is replaced by a
per-circuit elicitation: the model sees that circuit's observed rows with '?' for the rest and writes
all 8 rows; the harness vetoes a table that contradicts an observed answer or that no circuit built
under the task rules (inventory split without replacement across A, B, C, given the tables already
accepted) can produce, tells the model why, and re-asks. The harness concatenates the accepted tables.
The rule check uses only env.inventory, the rules in the system prompt and env.history."""

import itertools
import re

from harness_base import CDHarness

QUERY_RE = re.compile(r"<query>\s*([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*</query>")
ROW_RE = re.compile(r"([A-Ca-c])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*(?:=|->|:)\s*([01])")
ROWS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
MAX_TRIES = 6


def _tables(n_and, n_or, n_not):
    """Output tables (8-bit ints, bit r = output on row r, row r = a*4+b*2+c) of every circuit that uses
    exactly these gates: each AND/OR consumes two available signals, NOT replaces one, until one
    signal remains; constant circuits are excluded (task rules)."""
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


class Harness(CDHarness):
    def solve(self):
        self._query_phase()
        return self._final_phase()

    # ---- query phase: bare's protocol, unchanged ----
    def _query_phase(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Please start with your first query."}]
        B = env.budget
        for t in range(1, B + 1):
            reply = self.llm(msgs)
            msgs.append({"role": "assistant", "content": reply})
            m = QUERY_RE.search(reply or "")
            if not m:
                msgs.append({"role": "user", "content": "Your message did not contain a query in the "
                             "required format. Reply with exactly one <query>X(a, b, c)</query>."})
                continue
            c, a, b, x = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                bit = env.query(c, a, b, x)
            except Exception:
                break
            line = env.format_answer(c, a, b, x, bit)
            if t < B:
                msgs.append({"role": "user", "content": f"{line}\nPlease send your next query. {B - t} queries remaining."})

    # ---- final phase: per-circuit tables, vetoed against answers and the rules ----
    def _final_phase(self):
        env = self.env
        obs = {c: {} for c in "ABC"}
        for (c, a, b, x), bit in env.history:
            obs[c][(a, b, x)] = bit
        try:
            space = _joint_space(env.inventory)
        except Exception:
            space = set()
        alive = [s for s in space
                 if all(((s["ABC".index(c)] >> (a * 4 + b * 2 + x)) & 1) == bit
                        for (c, a, b, x), bit in env.history)]
        use_rules = bool(alive)   # if the rule model cannot explain the answers, check answers only

        accepted = {}
        # most-observed circuit first, so its (best-pinned) table constrains the others
        for c in sorted("ABC", key=lambda k: -len(obs[k])):
            accepted[c] = self._elicit(c, obs, alive if use_rules else None, accepted)
            if use_rules:
                i = "ABC".index(c)
                narrowed = [s for s in alive if s[i] == accepted[c]]
                if narrowed:
                    alive = narrowed
        return "<guess>" + "".join(_bits(accepted[c]) for c in "ABC") + "</guess>"

    def _summary(self, obs):
        lines = []
        for c in "ABC":
            got = ", ".join(f"{c}({a}, {b}, {x}) = {v}" for (a, b, x), v in sorted(obs[c].items()))
            lines.append(f"Circuit {c}: {got if got else 'never queried'}")
        return "\n".join(lines)

    def _elicit(self, c, obs, alive, accepted):
        env = self.env
        i = "ABC".index(c)
        template = "\n".join(f"{c}({a},{b},{x}) = {obs[c].get((a, b, x), '?')}" for a, b, x in ROWS)
        done = "".join(f"\nYou already settled circuit {k}: {_bits(t)} (rows 000..111)." for k, t in accepted.items())
        msgs = [{"role": "system", "content": env.system_prompt},
                {"role": "user", "content":
                 f"The queries are over. Everything the circuits answered:\n{self._summary(obs)}{done}\n\n"
                 f"Now write the complete output table of circuit {c}. Rows marked '?' were not queried: "
                 f"deduce them from the observed rows, the inventory and how circuits are built. "
                 f"Keep every observed row exactly as observed. Reason briefly, then end with all 8 rows "
                 f"in exactly this form, each '?' replaced by 0 or 1:\n{template}"}]
        best, fallback = None, None
        for k in range(MAX_TRIES):
            reply = self.llm(msgs, temperature=0.7 if k < 2 else 1.0)
            msgs.append({"role": "assistant", "content": reply})
            rows = {}
            for m in ROW_RE.finditer(reply or ""):
                if m.group(1).upper() == c:
                    rows[(int(m.group(2)), int(m.group(3)), int(m.group(4)))] = int(m.group(5))  # last wins
            missing = [r for r in ROWS if r not in rows]
            if missing:
                msgs.append({"role": "user", "content":
                             "Your reply did not give all 8 rows. Missing: "
                             + ", ".join(f"{c}({a},{b},{x})" for a, b, x in missing)
                             + f". End your reply with all 8 rows in this form:\n{template}"})
                continue
            t = sum(rows[r] << (r[0] * 4 + r[1] * 2 + r[2]) for r in ROWS)
            fallback = t
            wrong = [(r, v) for r, v in obs[c].items() if rows[r] != v]
            if wrong:
                msgs.append({"role": "user", "content":
                             "That table contradicts the answers you received: "
                             + "; ".join(f"{c}({a},{b},{x}) was observed = {v}, you wrote {1 - v}"
                                         for (a, b, x), v in wrong)
                             + f". Fix it. End with all 8 rows:\n{template}"})
                continue
            best = t
            if alive is None or any(s[i] == t for s in alive):
                return t
            unknown = [r for r in ROWS if r not in obs[c]]
            msgs.append({"role": "user", "content":
                         f"Table {_bits(t)} matches the observed rows, but no circuit that can be built "
                         f"from the inventory (with the gates the other circuits need) produces it. "
                         f"Rethink which gates circuit {c} uses and change some of the unqueried rows: "
                         + ", ".join(f"{c}({a},{b},{x})" for a, b, x in unknown)
                         + f". End with all 8 rows:\n{template}"})
        if best is not None:
            return best
        return fallback if fallback is not None else 0
