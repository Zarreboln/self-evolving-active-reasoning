"""Witness dossier: ask every suspect about the others without naming anyone, then judge each
suspect by what the others said about them.

What the published methods leave alone, this changes:

  * questions are templated and name no one -- who was near {location} around {time}, who else
    could get at {weapon}, who had trouble with {victim} -- so whom a suspect brings up is their
    own choice and not an echo of the question. Fifteen turns, three per suspect.
  * state is a dossier keyed by SUBJECT, not by turn: what a suspect said about themself, and what
    each other suspect said that names them. Names are matched in code, so the count of distinct
    other speakers who bring a suspect up (their salience) costs no model call.
  * the remaining turns go to the two most-named suspects: each is confronted with what was said
    about them, and someone who has not named them yet is asked about their movements.
  * the final judgment reads the dossier, own words and others' words in separate sections, with
    five votes; salience breaks ties.
"""
from __future__ import annotations

import re
from collections import Counter

from ..harness_base import DCHarness

LETTERS = "ABCDE"
TITLES = {"dr", "doctor", "professor", "prof", "mr", "mrs", "ms", "miss", "mx", "sir", "lady",
          "lord", "madam", "madame", "captain", "capt", "detective", "officer", "inspector", "chef",
          "father", "sister", "brother", "judge", "colonel", "major", "sergeant", "lieutenant",
          "reverend", "rev", "dame", "count", "countess", "baron", "baroness", "jr", "sr", "the",
          "of"}

TIME_WORDS = {"morning", "afternoon", "evening", "night", "midnight", "noon", "late", "early",
              "dawn", "dusk", "sunset", "sunrise", "midday", "late-night"}


def case_facts(info):
    """time / location / victim / weapon out of AR-Bench's fixed background layout."""
    def field(pat, default):
        m = re.search(pat, info, re.M)
        return m.group(1).strip().rstrip(".") if m else default
    def inline(v):                                   # "The study" -> "the study", mid-sentence
        w = v.split(" ", 1)
        return (w[0].lower() + v[len(w[0]):]) if w[0] in ("The", "A", "An") or (
            len(w[0]) > 1 and w[0][1:].islower() and w[0].lower() in TIME_WORDS) else v
    return {"time": inline(field(r"^time:\s*(.+)$", "the time of the murder")),
            "location": inline(field(r"^location:\s*(.+)$", "the crime scene")),
            "victim": field(r"^- name:\s*(.+)$", "the victim"),
            "weapon": inline(field(r"^- murder weapon:\s*(.+)$", "the murder weapon"))}


def name_tokens(name):
    out = []
    for t in re.split(r"[\s,]+", name):
        t = t.strip(".'\"()")
        if len(t) > 1 and t.lower() not in TITLES:
            out.append(t)
    return out


def build_matchers(suspects, victim=""):
    """One case-sensitive pattern per suspect: the full name, or any name part no one else shares."""
    toks = {s: name_tokens(s) for s in suspects}
    cnt = Counter(t for s in suspects for t in set(toks[s]))
    vt = set(name_tokens(victim))
    pats = {}
    for s in suspects:
        alts = [re.escape(s)] + [re.escape(t) for t in toks[s] if cnt[t] == 1 and t not in vt]
        pats[s] = re.compile(r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % "|".join(alts))
    return pats


def mentioned(text, pats, exclude=None):
    return [s for s, p in pats.items() if s != exclude and p.search(text or "")]


def parse_answer(text, pats, suspects):
    """The letter right after `Answer:`, else a name on that line, else the last name in the tail.

    AR-Bench's extractor takes the LAST capital A-E on the answer line, so `Answer: A. Dr. Nora
    Vale` reads as D; this reads the first token, and falls back to names before giving up.
    """
    text = text or ""
    ms = list(re.finditer(r"Answer\W{0,4}\s*(.*)", text, re.I))
    if ms:
        line = ms[-1].group(1)
        m = re.match(r"^[\s*\[\(\"']*([A-E])(?![A-Za-z])", line)
        if m:
            return LETTERS.index(m.group(1))
        hit = mentioned(line, pats)
        if len(hit) == 1:
            return suspects.index(hit[0])
        m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", line)
        if m:
            return LETTERS.index(m.group(1))
    tail = text[-300:]
    last, pos = None, -1
    for s, p in pats.items():
        for mm in p.finditer(tail):
            if mm.start() > pos:
                last, pos = s, mm.start()
    return suspects.index(last) if last else None


GUIDE = ("How to weigh the evidence: what OTHER suspects say about a person counts for more than what "
         "that person says about themself. Denials, vague answers and self-serving stories are weak "
         "evidence either way. Strong evidence is a concrete fact -- being at {location} around "
         "{time}, being able to get hold of {weapon}, or having a real reason to harm {victim} -- "
         "especially when someone other than the suspect confirms it. The murderer must have all "
         "three: the opportunity, the means and the motive.")

OPENERS = [
    "Around {time}, where were you, and who did you see near {location}? What were they doing?",
    "Apart from yourself, who could have gotten hold of {weapon}, and how?",
    "Who had a serious problem with {victim} recently, and what was it about?",
]


class Harness(DCHarness):
    N_FINAL = 5
    N_FOCUS = 2

    def _ask(self, who, question):
        if self.env.remaining <= 0:
            return None
        reply = self.env.interrogate(who, question)
        self._log.append((who, question, reply, mentioned(reply, self._pats, exclude=who)))
        return reply

    def _salience(self):
        speakers = {s: set() for s in self.env.suspects}
        for who, _, _, named in self._log:
            for s in named:
                speakers[s].add(who)
        return {s: len(v) for s, v in speakers.items()}

    def _dossier(self):
        out = []
        for i, s in enumerate(self.env.suspects):
            own = ["  - asked: %s\n    answered: %s" % (q, r) for w, q, r, _ in self._log if w == s]
            said = ["  - %s: %s" % (w, r) for w, _, r, named in self._log if s in named]
            out.append("=== %s. %s ===\nWhat %s said:\n%s\nWhat the other suspects said that names "
                       "%s:\n%s" % (LETTERS[i], s, s, "\n".join(own) or "  (nothing)", s,
                                    "\n".join(said) or "  (nobody named them)"))
        return "\n\n".join(out)

    def solve(self) -> int:
        env = self.env
        f = case_facts(env.init_info)
        self._pats = build_matchers(env.suspects, f["victim"])
        self._log = []

        for opener in OPENERS:
            for s in env.suspects:
                self._ask(s, opener.format(**f))

        sal = self._salience()
        focus = sorted(env.suspects, key=lambda s: (-sal[s], env.suspects.index(s)))[:self.N_FOCUS]
        per = max(1, env.remaining // max(1, len(focus)))
        for t in focus:
            claims, seen = [], set()
            for w, _, r, named in self._log:
                if t in named and r.strip() not in seen:
                    seen.add(r.strip())
                    claims.append((w, r))
            asked = 0
            for w, r in claims[:2]:                       # confront with what was said about them
                if asked >= per - 1:
                    break
                self._ask(t, '%s told me: "%s" What do you say to that?'
                          % (w, " ".join(r.split())[:300]))
                asked += 1
            silent = [s for s in env.suspects
                      if s != t and s not in {w for w, _ in claims}] or \
                     [s for s in env.suspects if s != t]
            for s in silent:
                if asked >= per:
                    break
                self._ask(s, "Did you notice anything about %s around %s?" % (t, f["time"]))
                asked += 1
        for t in focus:                                   # witnesses ran out: pin down the alibi
            if env.remaining > 0:
                self._ask(t, "Who can confirm where you were around %s?" % f["time"])

        choices = ", ".join("%s. %s" % (LETTERS[i], s) for i, s in enumerate(env.suspects))
        prompt = (env.system_prompt
                  + "\nWhat the interrogation found, suspect by suspect:\n\n" + self._dossier()
                  + "\n\n" + GUIDE.format(**f) + "\n\n"
                  + "Now tell me who is more likely to be the true murderer, %s.\n" % choices
                  + "You should only output the index of the candidate suspect like: A, B, C, D or E.\n"
                  + "you should strictly follow this answer format:\n"
                  + "Reason: [Your inference step by step]\nAnswer: [A, B, C, D or E]\n")
        votes = Counter()
        for _ in range(self.N_FINAL):
            idx = parse_answer(self.llm([{"role": "user", "content": prompt}]), self._pats,
                               env.suspects)
            if idx is not None:
                votes[idx] += 1
        sal_i = {i: sal[s] for i, s in enumerate(env.suspects)}
        return max(range(len(env.suspects)), key=lambda i: (votes[i], sal_i[i], -i))
