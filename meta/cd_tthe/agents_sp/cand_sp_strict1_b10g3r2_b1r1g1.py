"""Premise checks, then one anchored working story instead of a spray of unrelated hypotheses; atomic,
non-redundant questions; a final answer rewritten from the working story.

Every turn the model sees one fresh message: the puzzle, the referee's answers grouped as CONFIRMED (Yes) /
RULED OUT (No) / NOT IN THE STORY (Unknown), and a FOCUS for this question. The focus comes from:
  1. assumptions the puzzle makes a reader take for granted (asked first: situation puzzles hinge on one
     of them being false),
  2. a WORKING STORY the model keeps: its single best explanation, which must contain every confirmed fact
     (re-asked when it drops them), plus the parts of it that are not confirmed yet; those parts are tested
     one at a time. The story is rewritten after every Yes and every few questions,
  3. after a run of answers without a Yes: one broad question about a basic aspect of the story not asked
     about yet (who, on purpose or by mistake, is the event what it seems, how, why).
Each proposed question is checked (one short yes/no question about one fact, no "or", not about a
character's beliefs, no names absent from the puzzle, not a near-repeat and not a re-wording of an answered
question that adds nothing new); a failing reply is re-asked with the reason. The final answer is the
model's story written from the confirmed facts and its last working story, the best-supported of three
drafts, "A:" tag stripped.
DEAD TERMS: a person or thing the model brought in (a noun after "the/a/an/any/his/her/X's") that is not in
the puzzle and not in any question the referee answered Yes, and that the referee already denied (No to "a X",
or No/Unknown to two questions naming it), is dead. A question that names a dead term again is re-asked with the
denial quoted; working stories (and their UNSURE parts) and final drafts are told to leave the outright-denied
ones out, re-asked and scored down when they use them: the model stops spending questions on an invented frame the
referee has not taken up."""
import re

from sp_harness_base import SPHarness

GUIDE = """
How to ask well:
- The host reads each question ALONE, with no memory of earlier ones. Name people and things fully; never write "this", "these", "that event" referring to an earlier question.
- Ask about what actually happened (who did what, when, how, why), never about what a character believes, thinks, suspects or feels.
- One short question (under 16 words) about ONE fact, answerable by Yes or No. No "or" alternatives, no "and" joining two claims, no "What/Why/How" questions.
- Ask broad before narrow: first whether a kind of thing is true at all ("Was a family member involved?"), and only after a Yes ask for the specific detail.
- Do not invent names or details; use only people and things from the puzzle or from confirmed facts.
- Ask "Did ...?" / "Was ...?" about what happened, not "Could ...?" / "Is it possible ...?".
- Do not ask about what the puzzle already states or what is already CONFIRMED, and do not chase minor details (exact times, counts, places, handwriting, weather) -- go for who is behind it, what really happened, how, and why.
"""

DIMENSIONS = [
    "whether the main character of the puzzle did it or set it up themself",
    "whether the key event of the puzzle really happened the way it is described (it may be staged, faked, mistaken or not what it seems)",
    "whether it was done on purpose or happened by accident or mistake",
    "what kind of person is behind it (family member, friend, stranger, official, the main character)",
    "why it was done -- the motive or purpose behind it",
    "how it was physically done -- the method",
    "when it was set up -- long before the puzzle's events or just before",
]

STOP = set("a an the of to in on at for by with from and or is was were are be been did does do has had have it its "
           "that this any as into about her his their them they he she him there someone something not else other than".split())
BELIEF = re.compile(r"\b(believe|believes|believed|think|thinks|thought|suspect|suspects|suspected|feel|feels|felt|"
                    r"consider|considers|considered|theor\w*|opinion|wonder\w*)\b", re.I)
WH = re.compile(r"^(what|why|how|who|whom|whose|where|when|which)\b", re.I)
SPEC = re.compile(r"^(could|might|may|can|would)\b|\b(is|was) it possible\b", re.I)
OR = re.compile(r"\bor\b|\beither\b|\brather than\b", re.I)
POSS = re.compile(r"\b(?:the|her|his|their|[A-Z][a-z]+'s)\s+([a-z]+)")
ROLE = re.compile(r"^(neighbou?rs?|friends?|colleagues?|co-?workers?|landlord|landlady|tenants?|roommates?|child|children|kids?|"
                  r"sons?|daughters?|mother|father|mom|dad|parents?|sisters?|brothers?|siblings?|twins?|husband|wife|"
                  r"spouse|partner|boyfriend|girlfriend|ex|boss|employer|employees?|assistant|secretary|doctors?|nurses?|"
                  r"therapist|stranger|relatives?|cousins?|uncle|aunt|grand\w+|teacher|students?|classmates?|prankster|"
                  r"clerk|editor|reporter|journalist|officer|police|detective|lawyer|agent|guard|servant|maid|butler|"
                  r"housekeeper|caretaker|postman|mailman|courier|driver|pilot|\w+ists?|\w+ians?)$")
HEDGE = re.compile(r"\b(likely|possibly|perhaps|maybe|might|may|probably|seems?|appears?)\b", re.I)
ATAG = re.compile(r"^\s*\**\s*A\s*:\s*\**\s*")


NP = re.compile(r"\b(the|a|an|any|his|her|their|its|[A-Z][a-z]+'s)\s+([a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2})", re.I)
FUNC = set("the a an any his her their its was were is are did does do had has have be been being who whom which that "
           "when while after before during to of in on at by for from with as and or not into about over under than so "
           "if because still also just only ever".split())
GENERIC = set("same first last main only end truth reason cause purpose way time night day morning evening event events "
              "situation story puzzle person people place case result fact one ones other others whole real actual true "
              "new old local young small big large specific certain particular secret special different own kind type part "
              "lot few many some several death dead funeral burial body grave home house room family member friend "
              "relative life name police use involve believe think know".split())
NOT_NOUN = set("remain install equip set lead get make take give become want need try help leave find send write receive "
               "inherit live act die kill appear happen change create keep stay put move come go see look work start stop "
               "begin cause bring hold carry hide fake stage plan arrange own allow manage reach return "
               "outside inside within where near beneath around through behind without".split())


def _stem(w):
    w = re.sub(r"['’]s$", "", w.lower())
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 5 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def _nps(q):
    """(determiner, content words) of each noun phrase opened by the/a/an/any/possessive; verbs, adverbs and
    generic words dropped."""
    out = []
    for m in NP.finditer(q or ""):
        ws = []
        for w in m.group(2).lower().split():
            w = re.sub(r"['’]s$", "", w)
            if w in FUNC:
                break
            if w in NOT_NOUN:
                break
            if len(w) > 2 and w not in GENERIC and not re.search(r"(ed|ing|ly)$", w):
                ws.append(w)
        if ws:
            out.append((m.group(1).lower(), ws))
    return out


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
    MAX_TRY = 5
    REFRESH_EVERY = 2
    STALL = 3

    # ---------------------------------------------------------------- views
    def _kind(self, a):
        al = (a or "").strip().lower()
        al = al[2:].strip() if al.startswith("a:") else al
        return "yes" if al.startswith("yes") else "no" if al.startswith("no") else "bad" if "not a question" in al else "unk"

    def _facts(self):
        if not self.env.history:
            return "No questions asked yet."
        groups = {"yes": [], "no": [], "unk": [], "bad": []}
        for q, a in self.env.history:
            k = self._kind(a)
            groups[k].append(self.stmt.get(_strip_q(q), _strip_q(q)) if k == "yes" else _strip_q(q))
        out = []
        for key, title in [("yes", "CONFIRMED (host said Yes) -- these are true facts of the story"),
                           ("no", "RULED OUT (host said No)"),
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

    def _yes_no_words(self):
        surf = _words(self.env.surface)
        got = {"yes": set(), "no": set(), "unk": set(), "bad": set()}
        for q, a in self.env.history:
            got[self._kind(a)].update(_words(_strip_q(q)) - surf)
        yes = got["yes"]
        return yes, got["no"] - yes, got["unk"] - yes - got["no"]

    def _dead(self):
        """Dead terms: stem -> (word, phrase, question, reply, denied). A noun-phrase word not in the puzzle or any
        Yes question (first four letters), either denied outright (No to "a/an/any ... X ...") or named in two
        questions answered No/Unknown."""
        est = {w[:4] for w in _words(self.env.surface)}
        for q, a in self.env.history:
            if self._kind(a) == "yes":
                est |= {w[:4] for w in _words(_strip_q(q)) | _words(self.stmt.get(_strip_q(q), ""))}
        cnt, dead = {}, {}
        for q, a in self.env.history:
            k = self._kind(a)
            if k in ("yes", "bad"):
                continue
            q, seen = _strip_q(q), set()
            for det, ws in _nps(q):
                for w in ws:
                    s = _stem(w)
                    if w[:4] in est or s in seen:
                        continue
                    seen.add(s)
                    cnt[s] = cnt.get(s, 0) + 1
                    denied = k == "no" and det in ("a", "an", "any")
                    if s not in dead and (denied or cnt[s] >= 2):
                        dead[s] = (w, " ".join(ws), q, a, denied)
        return dead

    def _dead_in(self, text, dead=None, denied_only=False):
        dead = self._dead() if dead is None else dead
        hit = {}
        for _, ws in _nps(text):
            for w in ws:
                d = dead.get(_stem(w))
                if d and (d[4] or not denied_only):
                    hit[_stem(w)] = d
        return list(hit.values())

    def _dead_note(self):
        """The outright-denied phrases, for working stories and final drafts."""
        phrases = sorted({d[1] for d in self._dead().values() if d[4]})
        if not phrases:
            return ""
        return ("\nThe host answered No when asked whether these were involved -- leave them out of the story: " +
                "; ".join(phrases) + ".\n")

    def _statement(self, q):
        """The model restates a question the referee answered Yes as a plain fact (shown instead of the question)."""
        out = self._chat(f"A yes/no question was answered YES:\n{q}\n\nRewrite it as one plain statement of fact "
                         "(same people and things, no question mark, no extra details).\nFormat:\nFACT: ...",
                         temperature=0.2, max_tokens=60)
        m = re.search(r"FACT\s*:\s*(.+)", out)
        f = " ".join((m.group(1) if m else out).split()).strip("* ")
        return f if f and "?" not in f and len(f.split()) <= 2 * len(q.split()) + 5 else q

    def _missing(self, text):
        """Confirmed facts a model-written story leaves out: under half of a fact's own words (beyond the puzzle's)
        appear in it."""
        surf, tw, out = _words(self.env.surface), _words(text), []
        for q, a in self.env.history:
            if self._kind(a) != "yes":
                continue
            q = _strip_q(q)
            f = self.stmt.get(q, q)
            cov = []
            for src in (q, f):
                need = _words(src) - surf
                if need:
                    cov.append(len(need & tw) / len(need))
            if cov and max(cov) < 0.5:
                out.append(f)
        return out

    def _invented(self, text):
        known = set(re.findall(r"[A-Za-z]+", self.env.surface)) | {w for h, _ in self.env.history for w in re.findall(r"[A-Za-z]+", h)}
        return [w for w in re.findall(r"(?<=[a-z,;] )[A-Z][a-z]+", text) if w not in known]

    def _support(self, story):
        """How well a model-written story agrees with the referee: share of the confirmed questions' new words
        it uses, minus words found only in ruled-out or not-in-the-story questions, minus hedging."""
        yes, no, unk = self._yes_no_words()
        sw = _words(story)
        return (len(sw & yes) / max(1, len(yes)) - 0.3 * len(sw & no) / max(1, len(no))
                - 0.3 * len(sw & unk) / max(1, len(unk)) - 0.02 * len(HEDGE.findall(story)))

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

    def _working_story(self):
        """The model's single best explanation built on every confirmed fact, and its unconfirmed parts.
        Re-asked once when it leaves out most of what the referee confirmed."""
        prompt = (self._context() +
                  "\nWrite your single best explanation of the hidden story so far.\n"
                  "- It MUST contain every CONFIRMED fact (each one is true) and contradict no RULED OUT fact.\n"
                  "- Where nothing is confirmed yet, fill the gap with the simplest realistic guess; no invented names.\n"
                  "- 2-3 sentences: who is behind it, what really happened, how, and why.\n"
                  "Then list the 3 most important parts of your story that are NOT confirmed yet, each a short plain "
                  "statement (under 15 words), most important first: who is behind it, what really happened, why -- "
                  "not minor details (training, background, appearance, exact place or time).\n"
                  "Format:\nSTORY: ...\nUNSURE:\n1. ...\n2. ...\n3. ...") + self._dead_note()
        best, user, dead = None, prompt, self._dead()
        for temp in (0.6, 0.8, 0.8):
            out = self._chat(user, temperature=temp, max_tokens=350)
            m = re.search(r"STORY\s*:\s*(.+?)(?=\n\s*\**\s*UNSURE|\Z)", out, flags=re.S | re.I)
            story = " ".join((m.group(1) if m else out.split("UNSURE")[0]).split()).strip("* ")
            tail = out[m.end():] if m else out
            unsure = _items(tail, 3)
            miss = self._missing(story)
            gone = sorted({d[1] for d in self._dead_in(story + " " + " ".join(unsure), dead, denied_only=True)})
            score = (-len(miss), -len(gone), -len(self._invented(story)), self._support(story))
            if best is None or score > best[0]:
                best = (score, story, unsure)
            if not miss and not gone:
                break
            user = (prompt + "\n\nYour previous attempt:\nSTORY: " + story + "\n" +
                    ("It leaves out these CONFIRMED facts, which are TRUE and must be part of the story:\n" +
                     "\n".join(f"- {f}" for f in miss) + "\n" if miss else "") +
                    (f"It (or its UNSURE list) relies on {', '.join(gone)}, which the host never confirmed; drop "
                     "them and use a different explanation for that part.\n" if gone else "") +
                    "Rewrite it.")
        return best[1], best[2]

    def _next_focus(self):
        hist = self.env.history
        if self.premises:
            p = self.premises.pop(0)
            return f"Check whether this assumption is really true (it may be the twist): \"{p}\"."
        streak = 0
        for _, a in reversed(hist):
            if self._kind(a) == "yes":
                break
            streak += 1
        if streak >= self.STALL and self.since_broad >= self.STALL and self.dims:
            self.since_broad = 0
            d = self.dims.pop(0)
            return ("Your recent questions all missed. Step back and ask ONE broad, basic question (under 12 words) "
                    f"about {d}. Ask about a possibility not yet RULED OUT.")
        last_yes = bool(hist) and self._kind(hist[-1][1]) == "yes"
        if not self.story or not self.unsure or self.since_refresh >= self.REFRESH_EVERY or last_yes:
            self.story, self.unsure = self._working_story()
            self.since_refresh = 0
        dead = self._dead()
        self.unsure = [u for u in self.unsure if not self._dead_in(u, dead)]
        if self.unsure:
            u = self.unsure.pop(0)
            return (f"Your current best story: \"{self.story}\"\nTest this part of it, which is not confirmed yet, "
                    f"with one short direct question about its key fact: \"{u}\".")
        return "Ask about the most important fact of the hidden story that is still open."

    # ---------------------------------------------------------------- questions
    def _check(self, q):
        if not q or "?" not in q:
            return 6, "Your reply was not a question. Write one yes/no question ending with '?'."
        if WH.match(q):
            return 6, "That is not a yes/no question. Rewrite it so it can be answered Yes or No (start with Is/Was/Did/Does...)."
        if len(q.split()) > 18:
            return 1, "Too long. Ask about ONE fact in under 16 words."
        if OR.search(q):
            return 3, "Do not offer alternatives with 'or'; ask about ONE possibility so Yes or No settles it."
        if SPEC.search(q):
            return 2, "Do not ask whether something could be possible; ask directly whether it happened (Did ...? / Was ...?)."
        if BELIEF.search(q):
            return 2, ("Do not ask what a character believes, thinks, suspects or feels -- the host only knows what happened. "
                    "Ask about a concrete fact of the story.")
        known = set(re.findall(r"[A-Za-z]+", self.env.surface)) | {w for h, _ in self.env.history for w in re.findall(r"[A-Za-z]+", h)}
        names = [w for w in re.findall(r"(?<=[a-z,;] )[A-Z][a-z]+", q) if w not in known]
        if names:
            return 5, f"Do not invent names ({', '.join(names)}); refer to people only as the puzzle does or by their role."
        known = {w[:4] for w in _words(self.env.surface)}
        for h, a in self.env.history:
            if self._kind(a) == "yes":
                known |= {w[:4] for w in _words(h) | _words(self.stmt.get(_strip_q(h), ""))}
        for noun in POSS.findall(q):
            if ROLE.match(noun) and noun[:4] not in known:
                return 2, (f"No {noun} is in the puzzle or the confirmed facts, so the host cannot answer a question that "
                           f"takes one for granted. First ask whether any {noun} is involved at all (use \"a\" or "
                           "\"any\", not \"the\" or a possessive).")
        hit = self._dead_in(q)
        if hit:
            w, ph, hq, ha, _ = hit[0]
            return 5, (f"The host never confirmed any {ph}: it answered {ha} to \"{hq}\". Drop the {w} idea "
                       "completely and ask about a different possibility.")
        d = self._dup(q)
        if d:
            return 4, f"You already asked almost the same thing: \"{d[0]}\" (answer: {d[1]}). Ask about something different."
        return None

    def _dup(self, q):
        """A near-repeat of an answered question, or a re-wording of one that adds at most one new word
        (words compared by their first four letters, so 'write' / 'written' match)."""
        w = {x[:4] for x in _words(q)}
        for h, a in self.env.history:
            p = _strip_q(h)
            pw = {x[:4] for x in _words(p)}
            if q.lower() == p.lower():
                return p, a
            if not (w and pw):
                continue
            inter = len(w & pw)
            if (inter / len(w | pw) >= 0.6 or inter / min(len(w), len(pw)) >= 0.85
                    or (inter / len(pw) >= 0.7 and len(w - pw) <= 1)):
                return p, a
        return None

    def _ask_model(self, t, focus):
        def base_for(f):
            return (self._context() + f"\nFOCUS for this question: {f}\n\n{self.env.turn_prompt(t)}\n"
                    "Output exactly one line:\nQ: [your question]")
        base = base_for(focus)
        sys_msg = {"role": "system", "content": self.env.system_prompt + GUIDE}
        cands, notes = [], []
        for k in range(self.MAX_TRY):
            user = base if not notes else base + "\n\nDo NOT repeat these rejected attempts:\n" + "\n".join(notes)
            q = _extract(self.llm([sys_msg, {"role": "user", "content": user}],
                                  temperature=0.7 if k == 0 else 1.0, max_tokens=80))
            bad = self._check(q)
            if bad is None:
                return q
            if q:
                cands.append((bad[0], k, q))
            notes.append(f"- \"{q}\" -> {bad[1]}")
            if k == 1:                       # two failures on one focus: move to the next agenda item
                base = base_for(self._next_focus())
        return min(cands)[2] if cands else "(no question)"       # least serious failure

    # ---------------------------------------------------------------- episode
    def solve(self):
        env = self.env
        self.premises = self._premises()
        self.story, self.unsure, self.since_refresh = "", [], 0
        self.stmt = {}
        self.dims, self.since_broad = list(DIMENSIONS), self.STALL
        t = 0
        while env.remaining > 0:
            t += 1
            q = self._ask_model(t, self._next_focus())
            a = env.ask("Q: " + _strip_q(q))
            if self._kind(a) == "yes":
                self.stmt[_strip_q(q)] = self._statement(_strip_q(q))
            self.since_refresh += 1
            self.since_broad += 1

        story, _ = self._working_story()
        user = (self._context() +
                f"\nYour best explanation so far: {story}\n"
                "\nThe questioning is over. Write the hidden story.\n"
                "- Start from your best explanation and correct it where it disagrees with the answers: every CONFIRMED "
                "fact must appear in the story, and nothing RULED OUT may appear.\n"
                "- Tell it as the story itself in the past tense, 4-6 plain sentences: who the people are, what really "
                "happened, how it led to the situation in the puzzle, and why.\n"
                "- No hedging (likely, possibly, perhaps, may, might), no invented names, no mention of questions, "
                "answers or the host, and do not list what did NOT happen.\n- Write ONE story only.\n" + self._dead_note() +
                env.final_prompt)
        msgs = [{"role": "system", "content": env.system_prompt}, {"role": "user", "content": user}]
        stories = []
        for temp in (0.3, 0.8, 0.8):
            stories += self._drafts(self.llm(msgs, temperature=temp, max_tokens=400))
        if not stories:
            stories = self._drafts(self.llm(msgs, temperature=1.0, max_tokens=400)) or [story]
        dead = self._dead()
        score = lambda s: (-len(self._missing(s)), -len(self._dead_in(s, dead, True)), -len(self._invented(s)),
                           self._support(s))
        best = max(stories, key=score)
        miss, names = self._missing(best), self._invented(best)
        if miss or names:                    # one repair round: the model rewrites its own draft
            fix = ("\n\nYour draft:\n" + best + "\n" +
                   ("It leaves out these CONFIRMED facts, which are TRUE and must be in the story:\n" +
                    "\n".join(f"- {f}" for f in miss) + "\n" if miss else "") +
                   (f"It invents names ({', '.join(sorted(set(names)))}); refer to people by their role.\n" if names else "") +
                   "Rewrite the whole story with these corrections.\n" + env.final_prompt)
            for fixed in self._drafts(self.llm([msgs[0], {"role": "user", "content": user + fix}], temperature=0.3,
                                               max_tokens=400)):
                if score(fixed) > score(best):
                    best = fixed
        return best

    @staticmethod
    def _drafts(reply):
        """A model reply often holds a story and then a second "A: ..." story; each part is a separate draft."""
        parts = re.split(r"\n\s*\**\s*A\s*:\s*\**", "\n" + (reply or "").strip())
        out = [ATAG.sub("", " ".join(p.split())).strip() for p in parts]
        return [p for p in out if len(p.split()) >= 8]
