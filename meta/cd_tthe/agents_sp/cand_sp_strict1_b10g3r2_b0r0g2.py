"""Bare AR-Bench loop plus a fact sheet, a model-written question agenda, a question filter and a
committed final story.

Every turn the model sees one fresh message instead of the raw chat: the puzzle, the referee's
answers grouped as CONFIRMED (Yes) / RULED OUT (No) / NOT IN THE STORY (Unknown), and a FOCUS for
this question. The focus comes from an agenda the model wrote itself:
  1. assumptions the puzzle makes a reader take for granted (asked first: situation puzzles hinge
     on one of them being false),
  2. after a Yes: follow up on that confirmed fact,
  3. otherwise: short candidate explanations, re-listed by the model every few questions and tested
     in turn.
Each proposed question is checked (one yes/no question, short, not about a character's beliefs, no
names absent from the puzzle, not a near-repeat); a failing reply is re-asked from a clean context
with the reason, and a near-repeat is never sent while any other candidate exists. The final answer
is the model's story written from the fact sheet, committed to one explanation, "A:" tag stripped."""
import re

from sp_harness_base import SPHarness

GUIDE = """
How to ask well:
- The host reads each question ALONE, with no memory of earlier ones. Name people and things fully; never write "this", "these", "that event" referring to an earlier question.
- Ask about what actually happened (who did what, when, how, why), never about what a character believes, thinks, suspects or feels.
- One short question (under 20 words), one fact, answerable by Yes or No. No "or" alternatives, no "What/Why/How" questions.
- Do not invent names or details; use only people and things from the puzzle or from confirmed facts.
- Ask "Did ...?" / "Was ...?" about what happened, not "Could ...?" / "Is it possible ...?".
- Do not ask about what the puzzle already states, and do not chase minor details (exact times, counts, places) -- go for who is behind it, what really happened, how, and why.
"""

STOP = set("a an the of to in on at for by with from and or is was were are be been did does do has had have it its "
           "that this any as into about her his their them they he she him there someone something".split())
BELIEF = re.compile(r"\b(believe|believes|believed|think|thinks|thought|suspect|suspects|suspected|feel|feels|felt|"
                    r"consider|considers|considered|theor\w*|opinion|wonder\w*)\b", re.I)
WH = re.compile(r"^(what|why|how|who|whom|whose|where|when|which)\b", re.I)
SPEC = re.compile(r"^(could|might|may|can|would)\b|\b(is|was) it possible\b", re.I)
HEDGE = re.compile(r"\b(likely|possibly|perhaps|maybe|might|may|probably|seems?|appears?)\b", re.I)


def _words(q):
    return {w for w in re.findall(r"[a-z0-9']+", q.lower()) if w not in STOP}


def _strip_q(q):
    return re.sub(r"^\s*Q\s*:\s*", "", q or "").strip()


def _extract(reply):
    """The model's question: the first line carrying a '?', 'Q:' tag removed, cut after the first '?'."""
    lines = [l.strip() for l in (reply or "").strip().splitlines() if l.strip()]
    pick = next((l for l in lines if "?" in l), lines[0] if lines else "")
    pick = re.sub(r"^\**\s*(Q|Question)\s*\d*\s*[:.)-]\s*\**", "", pick, flags=re.I).strip().strip("*").strip()
    if "?" in pick:
        pick = pick[:pick.index("?") + 1]
    return pick


def _items(text, n):
    """Numbered / bulleted lines of a model reply, markup removed."""
    out = []
    for l in (text or "").splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s*(.+)$", l)
        if m:
            s = re.sub(r"\*\*|__", "", m.group(1)).strip()
            s = re.sub(r"^[^:]{0,40}:\s+", "", s) if len(s.split(":")[0].split()) <= 5 and ":" in s[:45] else s
            if len(s.split()) >= 3:
                out.append(s[:220])
    return out[:n]


class Harness(SPHarness):
    MAX_TRY = 4
    REFRESH_EVERY = 3
    MAX_FOLLOW = 2

    # ---------------------------------------------------------------- views
    def _is_yes(self, a):
        return (a or "").strip().lower().startswith("yes")

    def _facts(self):
        if not self.env.history:
            return "No questions asked yet."
        groups = {"yes": [], "no": [], "unk": [], "bad": []}
        for q, a in self.env.history:
            al = (a or "").lower()
            key = "yes" if al.startswith("yes") else "no" if al.startswith("no") else "bad" if "not a question" in al else "unk"
            groups[key].append(_strip_q(q))
        out = []
        for key, title in [("yes", "CONFIRMED (host said Yes)"), ("no", "RULED OUT (host said No)"),
                           ("unk", "NOT PART OF THE STORY (host said Unknown)"),
                           ("bad", "REJECTED by the host as not a yes/no question")]:
            if groups[key]:
                out.append(title + ":\n" + "\n".join(f"- {q}" for q in groups[key]))
        return "\n".join(out)

    def _context(self):
        n = len(self.env.history)
        return (f"Puzzle: {self.env.surface}\n\n"
                f"Answers so far ({n} of {self.env.budget} questions used, {self.env.remaining} left):\n{self._facts()}\n")

    def _chat(self, prompt, temperature=0.7, max_tokens=300):
        return self.llm([{"role": "system", "content": "You are a sharp solver of lateral-thinking situation puzzles."},
                         {"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens) or ""

    # ---------------------------------------------------------------- agenda
    def _premises(self):
        out = self._chat(
            f"Puzzle: {self.env.surface}\n\n"
            "Situation puzzles hinge on a hidden twist: something the puzzle makes a reader take for granted is false. "
            "Write down what a reader takes for granted, one short plain statement each (under 15 words), using only "
            "people and things in the puzzle:\n"
            "1. WHO: who a reader assumes is behind it (is it really someone other than the main character?)\n"
            "2. REAL: an event a reader assumes really happened as described (a death, an accident, a crime)\n"
            "3. SAME: a person or object a reader assumes is exactly who or what it seems\n"
            "4. TIME: when a reader assumes something was done, written or sent\n"
            "5. CAUSE: the ordinary cause a reader assumes\n"
            "Format:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...", temperature=0.5)
        return _items(out, 5)

    def _hypotheses(self):
        out = self._chat(
            self._context() +
            "\nList 4 VERY DIFFERENT candidate explanations of the hidden story. Build on the CONFIRMED facts: each must "
            "agree with every CONFIRMED fact and contradict no RULED OUT fact. Keep them realistic (no gadgets, ghosts or "
            "time travel unless confirmed). Each one short sentence (under 20 words) saying who did what and why, "
            "no invented names.\n"
            "Format:\n1. ...\n2. ...\n3. ...\n4. ...", temperature=0.8)
        return _items(out, 4)

    def _next_focus(self):
        hist = self.env.history
        yes_idx = [i for i, (_, a) in enumerate(hist) if self._is_yes(a)]
        if yes_idx and len(hist) - 1 - yes_idx[-1] <= 1 and self.followups < self.MAX_FOLLOW:
            self.followups += 1
            return (f"The host CONFIRMED: \"{_strip_q(hist[yes_idx[-1]][0])}\". Build on this clue: ask what it means "
                    "for the core of the mystery -- who is really behind it, what really happened, how, or why. "
                    "Do not narrow down minor details.")
        self.followups = 0
        if self.premises:
            p = self.premises.pop(0)
            return f"Check whether this assumption is really true (it may be the twist): \"{p}\"."
        if not self.hyps or self.since_refresh >= self.REFRESH_EVERY or self.yes_since_refresh:
            self.hyps = self._hypotheses() or self.hyps
            self.hidx, self.since_refresh, self.yes_since_refresh = 0, 0, False
        if self.hyps:
            h = self.hyps[self.hidx % len(self.hyps)]
            self.hidx += 1
            return f"Test this candidate explanation with one direct question about its key fact: \"{h}\"."
        return "Ask about the most important fact of the hidden story that is still open."

    # ---------------------------------------------------------------- questions
    def _check(self, q):
        if not q or "?" not in q:
            return "Your reply was not a question. Write one yes/no question ending with '?'."
        if WH.match(q):
            return "That is not a yes/no question. Rewrite it so it can be answered Yes or No (start with Is/Was/Did/Does...)."
        if len(q.split()) > 25:
            return "Too long. Ask about one fact in under 20 words."
        if SPEC.search(q):
            return "Do not ask whether something could be possible; ask directly whether it happened (Did ...? / Was ...?)."
        if BELIEF.search(q):
            return ("Do not ask what a character believes, thinks, suspects or feels -- the host only knows what happened. "
                    "Ask about a concrete fact of the story.")
        known = set(re.findall(r"[A-Za-z]+", self.env.surface)) | {w for h, _ in self.env.history for w in re.findall(r"[A-Za-z]+", h)}
        names = [w for w in re.findall(r"(?<=[a-z,;] )[A-Z][a-z]+", q) if w not in known]
        if names:
            return f"Do not invent names ({', '.join(names)}); refer to people only as the puzzle does or by their role."
        if self._dup(q):
            return f"You already asked almost the same thing: \"{self._dup(q)}\". Ask about something different."
        return None

    def _dup(self, q):
        w = _words(q)
        for h, _ in self.env.history:
            p = _strip_q(h)
            pw = _words(p)
            if q.lower() == p.lower() or (w and pw and (len(w & pw) / len(w | pw) >= 0.6
                                                        or len(w & pw) / min(len(w), len(pw)) >= 0.85)):
                return p
        return None

    def _ask_model(self, t, focus):
        base = (self._context() + f"\nFOCUS for this question: {focus}\n\n{self.env.turn_prompt(t)}\n"
                "Output exactly one line:\nQ: [your question]")
        sys_msg = {"role": "system", "content": self.env.system_prompt + GUIDE}
        cands, notes = [], []
        for k in range(self.MAX_TRY):
            user = base if not notes else base + "\n\nDo NOT repeat these rejected attempts:\n" + "\n".join(notes)
            q = _extract(self.llm([sys_msg, {"role": "user", "content": user}],
                                  temperature=0.7 if k == 0 else 1.0, max_tokens=80))
            why = self._check(q)
            if why is None:
                return q
            if q:
                cands.append(q)
            notes.append(f"- \"{q}\" -> {why}")
            if k == 1:                       # two failures on one focus: move to the next agenda item
                focus = self._next_focus()
                base = (self._context() + f"\nFOCUS for this question: {focus}\n\n{self.env.turn_prompt(t)}\n"
                        "Output exactly one line:\nQ: [your question]")
        fresh = [c for c in cands if "?" in c and not self._dup(c)]
        return (fresh or [c for c in cands if "?" in c] or cands or ["(no question)"])[0]

    # ---------------------------------------------------------------- episode
    def solve(self):
        env = self.env
        self.premises = self._premises()
        self.hyps, self.hidx, self.since_refresh, self.yes_since_refresh = [], 0, 0, False
        self.followups = 0
        t = 0
        while env.remaining > 0:
            t += 1
            q = self._ask_model(t, self._next_focus())
            a = env.ask("Q: " + _strip_q(q))
            self.since_refresh += 1
            self.yes_since_refresh |= self._is_yes(a)

        cands = self._hypotheses()
        user = (self._context() +
                ("\nYour candidate explanations:\n" + "\n".join(f"- {h}" for h in cands) + "\n" if cands else "") +
                "\nThe questioning is over. Write the hidden story.\n"
                "- Commit to ONE explanation: the one that fits every CONFIRMED fact, contradicts no RULED OUT fact, "
                "and fully explains the puzzle.\n"
                "- Tell it as the story itself in the past tense, 4-6 plain sentences: who the people are, what really "
                "happened, how it led to the situation in the puzzle, and why.\n"
                "- No hedging (likely, possibly, perhaps, may, might), no invented names, no mention of questions, "
                "answers or the host.\n" + env.final_prompt)
        msgs = [{"role": "system", "content": env.system_prompt}, {"role": "user", "content": user}]
        stories = []
        for temp in (0.3, 0.8, 0.8):
            a = re.sub(r"^\s*\**A\s*:\s*\**\s*", "", (self.llm(msgs, temperature=temp, max_tokens=400) or "").strip()).strip()
            if a:
                stories.append(a)
        if not stories:
            stories = [re.sub(r"^\s*\**A\s*:\s*\**\s*", "", (self.llm(msgs, temperature=1.0, max_tokens=400) or "").strip()).strip()]
        return max(stories, key=self._support)

    def _support(self, story):
        """How well a model-written story agrees with the referee: share of the confirmed questions' new words
        it uses, minus words found only in ruled-out questions, minus hedging."""
        surf = _words(self.env.surface)
        yes, no = set(), set()
        for q, a in self.env.history:
            (yes if self._is_yes(a) else no if (a or "").lower().startswith("no") else set()).update(_words(_strip_q(q)) - surf)
        no -= yes
        sw = _words(story)
        return (len(sw & yes) / max(1, len(yes)) - 0.5 * len(sw & no) / max(1, len(no))
                - 0.02 * len(HEDGE.findall(story)))
