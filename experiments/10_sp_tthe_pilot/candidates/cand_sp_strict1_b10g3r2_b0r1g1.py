"""Gap tracker. Hypothesis: episodes fail not for lack of candidate explanations but because the model
never integrates what the referee has already confirmed. A decisive Yes ("Emily is alive", "the family
did attend the burial", "the sender is not John") does not change what it asks next: it keeps drilling
how a small mechanism worked (devices, routes, timers, which pet) -- most of those get No -- instead of
asking about the contradiction the confirmed facts expose (she is alive, so why was she mourned?), and
the final story then drops or contradicts the confirmed facts.

Mechanism (one): the model keeps a short note with two parts, rewritten from the answers so far at the
start, after every Yes and every few turns:
  KNOWN   -- what the answers establish, stated as plain facts (no speculation),
  MYSTERY -- the single most important thing the hidden story still has to explain, sharpened by the
             KNOWN facts (when a fact clashes with what the puzzle seems to say, the clash is the mystery).
Every question is written with the note in view and must aim at MYSTERY, about the main people and
events rather than how small details worked. The final story must contain every KNOWN fact and resolve
the MYSTERY. No candidate solutions are kept between turns.
Each call sees a fresh, compact context (puzzle + numbered answers + note), not a growing chat.
Question hygiene (from the parent): a reply is re-asked if it is not one short yes/no question, is a
reworded repeat of an earlier one, invents names, or joins two claims with "or".
"""
import re

from sp_harness_base import SPHarness, BudgetExhausted

REFRESH_EVERY = 3
STOP = set("a an the of to in on at for by with from is was were are be been did does do had has have "
           "it its this that his her their he she they someone something any some or and as".split())


def _norm(q):
    q = re.sub(r"^\s*Q\s*:\s*", "", q or "", flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def _words(q):
    return {w for w in _norm(q).split() if w not in STOP}


def _verdict(a):
    a = re.sub(r"^\s*A\s*:\s*", "", a or "", flags=re.I).strip().lower()
    if "not a question" in a:
        return "not a question"
    for v in ("yes", "no", "unknown"):
        if a.startswith(v):
            return v.capitalize()
    return a[:40]


class Harness(SPHarness):
    def _facts(self):
        if not self.env.history:
            return "(none yet)"
        lines = []
        for i, (q, a) in enumerate(self.env.history, 1):
            q = re.sub(r"^\s*Q\s*:\s*", "", q, flags=re.I)
            lines.append(f"{i}. {q} -> {_verdict(a)}")
        return "\n".join(lines)

    def _note(self):
        msgs = [{"role": "user", "content": (
            f"Situation puzzle:\n{self.env.surface}\n\n"
            "Questions already asked and the referee's answers (the referee knows the hidden story; "
            "Unknown means the story does not say):\n"
            f"{self._facts()}\n\n"
            "Write a short note with two parts.\n"
            "KNOWN: what the answers so far establish, as plain facts in 1-4 short sentences. Turn every "
            "Yes into a statement, and include the No answers that rule out an obvious explanation. Only "
            "what the answers say -- no guesses. If nothing is established yet, write 'nothing yet'.\n"
            "MYSTERY: one question about a PERSON in the puzzle (named, or implied such as the sender, the "
            "dead person, the one who was buried) that the KNOWN facts leave unexplained: who that person really "
            "is, what really happened to them, or what they did. Not about an object (a letter, a note, a "
            "postcard) or how it travelled. Use the KNOWN facts: if one clashes with what the puzzle seems to "
            "say, the clash is the mystery (for example: if someone said to be dead is alive, the mystery is "
            "why they were believed dead). A vague fact ('someone she knows', 'a hidden reason') is not an "
            "explanation: then the mystery is exactly who or what it is. If the last answers were all No, the "
            "mystery was framed wrongly: pick a different person or a more basic question about them.\n"
            "Format:\nKNOWN: ...\nMYSTERY: ...")}]
        out = self.llm(msgs, temperature=0.5, max_tokens=250) or ""
        k = re.search(r"KNOWN\s*:\s*(.+?)(?=\n\s*MYSTERY\s*:|\Z)", out, flags=re.S | re.I)
        m = re.search(r"MYSTERY\s*:\s*(.+)", out, flags=re.S | re.I)
        known = " ".join(k.group(1).split()) if k else ""
        mystery = " ".join(m.group(1).split()) if m else ""
        if not mystery:
            mystery = " ".join(out.split())[:300] or "How could the situation in the puzzle happen?"
        return known or "nothing yet", mystery

    def _question(self, note, t):
        known, mystery = note
        base = (
            f"Situation puzzle:\n{self.env.surface}\n\n"
            f"Questions already asked and the referee's answers:\n{self._facts()}\n\n"
            f"Established so far: {known}\n"
            f"Still unexplained: {mystery}\n\n"
            f"Turn {t} of {self.env.budget}. Ask ONE short yes/no question (at most 18 words) about a person: "
            "who they really are, where they were, what happened to them, or what they did. It must help explain "
            "what is still unexplained. Do not ask about an object's route or mechanism (devices, mail, hiding "
            "places, timing) or about anyone's feelings or beliefs -- those get 'No' or 'Unknown' or tell nothing. "
            "Take every established fact as true and build on it. "
            "Check one claim only (no 'or'). Never re-ask a reworded version of an earlier question: its "
            "answer will not change.\n"
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
                return f"it repeats question {i} in other words (answer was {_verdict(a)})"
        if len(q.split()) > 22:
            return "too long; check ONE claim in at most 18 words"
        if re.search(r"(,\s*or\b|\bor\s+(is|was|were|did|does|do|has|had|through|by|via|someone|something)\b)", q, flags=re.I):
            return "it joins two different claims with 'or'; the answer would not tell them apart"
        if re.search(r"\b(believe|believed|believes|feel|feels|felt|hope|hopes|hoped|satisfaction|think|thinks|thought)\b", q, flags=re.I):
            return "it asks about someone's feelings or beliefs; ask what a person is, did, or what happened"
        known = set(re.findall(r"[A-Za-z]+", self.env.surface))
        names = [x for x in re.findall(r"(?<=\s)[A-Z][a-z]+", q) if x not in known]
        if names:
            return f"it uses invented names ({', '.join(names)}); refer to people by role instead"
        return ""

    def solve(self):
        env = self.env
        note, since = None, REFRESH_EVERY
        t = 0
        while env.remaining > 0:
            last_yes = bool(env.history) and _verdict(env.history[-1][1]) == "Yes"
            if note is None or since >= REFRESH_EVERY or last_yes:
                note, since = self._note(), 0
            t += 1
            since += 1
            try:
                env.ask(self._question(note, t))
            except BudgetExhausted:
                break
        known, mystery = self._note()
        msgs = [{"role": "user", "content": (
            f"Situation puzzle:\n{env.surface}\n\n"
            f"Questions asked and the referee's answers:\n{self._facts()}\n\n"
            f"Established: {known}\n"
            f"Still to explain: {mystery}\n\n"
            "Write the hidden story as one coherent, definite account in 3-5 sentences: who, what really "
            "happened, how, and why, so the puzzle is fully explained. It must contain every established fact "
            "(every Yes is a true fact of the story; nothing answered No may appear) and give the most likely "
            "explanation of what is still to explain. State it as fact; do not hedge, do not list the "
            "questions, do not mention the referee.\n" + env.final_prompt)}]
        return self.llm(msgs, temperature=0.3) or ""
