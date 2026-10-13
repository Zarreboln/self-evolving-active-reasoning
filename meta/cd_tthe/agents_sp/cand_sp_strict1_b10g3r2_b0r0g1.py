"""Twist board. Hypothesis: bare episodes fail because the model never states what the solution
could be, so it drills peripheral details (feelings, mail delays, where a note lay) that the story
does not contain, keeps narrowing a line the referee already rejected, and ends with a hedged
summary of its questions instead of a story.

Mechanism (one): the model keeps a board of 3 different one-sentence candidate twists (the single
fact that would resolve the puzzle), rewritten every few turns and after every Yes from all answers
so far; each question is asked with the board in view and must check the core twist of a candidate,
broad before narrow; the final answer is the model's best-fitting candidate written as a story.
Each call sees a fresh, compact context (puzzle + numbered answers + board), not a growing chat.
Question hygiene: a reply is re-asked if it is not one short yes/no question, is a reworded
repeat of an earlier one (the referee's answer would not change), or invents names.
"""
import re

from sp_harness_base import SPHarness, BudgetExhausted

REFRESH_EVERY = 4
STOP = set("a an the of to in on at for by with from is was were are be been did does do had has have "
           "it its this that his her their he she they someone something any some or and as".split())


def _norm(q):
    q = re.sub(r"^\s*Q\s*:\s*", "", q or "", flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def _words(q):
    return {w for w in _norm(q).split() if w not in STOP}


class Harness(SPHarness):
    def _facts(self):
        if not self.env.history:
            return "(none yet)"
        lines = []
        for i, (q, a) in enumerate(self.env.history, 1):
            q = re.sub(r"^\s*Q\s*:\s*", "", q, flags=re.I)
            a = "not a question" if "not a question" in a.lower() else a
            lines.append(f"{i}. {q} -> {a}")
        return "\n".join(lines)

    def _board(self):
        msgs = [{"role": "user", "content": (
            f"Situation puzzle:\n{self.env.surface}\n\n"
            "Questions already asked and the referee's answers (the referee knows the hidden story):\n"
            f"{self._facts()}\n\n"
            "Write 3 candidate solutions. Each is ONE sentence stating the core twist: the single hidden fact "
            "that makes the strange situation make sense (who really did it, what really happened, and why). "
            "No backstory, no invented names, places or small details. Each candidate must keep everything "
            "answered Yes and contradict nothing answered No. The 3 candidates must differ in the core twist "
            "itself, not just in details.\n"
            "Format:\nC1: ...\nC2: ...\nC3: ...")}]
        out = self.llm(msgs, temperature=0.8, max_tokens=300) or ""
        cs = re.findall(r"C\d\s*[:.)-]\s*(.+?)(?=\n\s*C\d\s*[:.)-]|\Z)", out, flags=re.S)
        cs = [" ".join(c.split()) for c in cs if c.strip()]
        return "\n".join(f"C{i}: {c}" for i, c in enumerate(cs[:3], 1)) if cs else out.strip()

    def _question(self, board, t):
        base = (
            f"Situation puzzle:\n{self.env.surface}\n\n"
            f"Questions already asked and the referee's answers:\n{self._facts()}\n\n"
            f"Candidate solutions:\n{board}\n\n"
            f"Turn {t} of {self.env.budget}. Ask ONE short yes/no question (at most 18 words) that checks the "
            "core twist of one candidate, or tells candidates apart. Ask broad questions (is this kind of thing "
            "true at all?) before narrow ones, and only narrow down after a Yes. Do not ask about minor details "
            "the story probably never mentions (feelings, appearance, weather, what someone noticed) -- those get "
            "'Unknown'. Never re-ask a reworded version of an earlier question: its answer will not change.\n"
            "Output exactly one line:\nQ: [your yes/no question]")
        msgs = [{"role": "system", "content": self.env.system_prompt}, {"role": "user", "content": base}]
        tries, out = [], ""
        for _ in range(3):
            out = self.llm(msgs) or ""
            m = re.search(r"Q\s*:\s*(.+)", out)
            q = (m.group(1) if m else out).strip().splitlines()[0].strip() if out.strip() else ""
            if "?" in q:
                q = q[: q.index("?") + 1]
            problem = self._problem(q)
            if not problem:
                return "Q: " + q
            if q:
                tries.append((len(problem), len(q.split()), q))
            msgs = msgs + [{"role": "assistant", "content": out},
                           {"role": "user", "content": f"Not accepted: {problem}. Ask a different question, one line:\nQ: [your yes/no question]"}]
        return "Q: " + min(tries)[2] if tries else out

    def _problem(self, q):
        if not q or not q.endswith("?"):
            return "that is not a single yes/no question ending with '?'"
        w = _words(q)
        for i, (prev, a) in enumerate(self.env.history, 1):
            p = _words(prev)
            if _norm(q) == _norm(prev) or (w and p and len(w & p) / len(w | p) >= 0.6):
                return f"it repeats question {i} in other words (answer was {a})"
        if len(q.split()) > 22:
            return "too long; check ONE claim in at most 18 words"
        known = set(re.findall(r"[A-Za-z]+", self.env.surface))
        names = [x for x in re.findall(r"(?<=\s)[A-Z][a-z]+", q) if x not in known]
        if names:
            return f"it uses invented names ({', '.join(names)}); refer to people by role instead"
        return ""

    def solve(self):
        env = self.env
        board, since = "", REFRESH_EVERY
        t = 0
        while env.remaining > 0:
            last_yes = bool(env.history) and env.history[-1][1].lower().startswith("yes")
            if since >= REFRESH_EVERY or last_yes:
                board, since = self._board(), 0
            t += 1
            since += 1
            try:
                env.ask(self._question(board, t))
            except BudgetExhausted:
                break
        board = self._board()
        msgs = [{"role": "user", "content": (
            f"Situation puzzle:\n{env.surface}\n\n"
            f"Questions asked and the referee's answers:\n{self._facts()}\n\n"
            f"Candidate solutions:\n{board}\n\n"
            "Pick the candidate that best fits ALL the answers (fix it where an answer contradicts it; every Yes "
            "is a true fact of the story) and write the hidden story as one coherent, definite account in 3-5 "
            "sentences: who, what really happened, how, and why, so the puzzle is fully explained. State it as "
            "fact; do not hedge, do not list the questions, do not mention the referee.\n" + env.final_prompt)}]
        return self.llm(msgs, temperature=0.3) or ""
