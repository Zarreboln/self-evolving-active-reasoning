"""Contradiction hunt: choose whom to question by where the accounts disagree, and stop once they
point one way.

The published methods pick the next suspect by round-robin or by the model's free choice. Here
the next question is fixed by an open issue in what has been said so far:

  * opening, five turns: every suspect gives their whereabouts at {time} and who can confirm them.
  * then every turn, an auditor pass reads the statements as a numbered list and names ONE open
    issue -- a claim another suspect's account contradicts, a claim at odds with the case facts, or
    an alibi nobody has confirmed -- as `TARGET | WITNESS | ISSUE`. If a witness who has not yet
    been asked about that target is named, the witness is asked to confirm or refute it; otherwise
    the target is confronted with it. A repeated question falls back to coverage: the least-asked
    suspect is asked who can confirm their account.
  * state is an issue log and a strike count per suspect (how often the auditor's issue was theirs).
  * stopping: from turn 12 on, once one suspect has drawn 4 strikes and leads the next by 3, accuse.
  * the final judgment reads the record plus the issue log, five votes; strikes break ties.
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

AUDIT = """Statements so far:
{statements}

Issues already investigated:
{issues}

Find the single most important OPEN issue in these statements that is not already investigated:
a claim that another suspect's statement contradicts, a claim that does not fit the case facts
(the time, the place, the weapon), or an alibi or claim that no one else has confirmed yet.
Reply in exactly one line, in this format:
TARGET: <full name of the suspect whose account is in question> | WITNESS: <full name of another suspect who could confirm or refute it, or none> | ISSUE: <the issue in one sentence>"""


def parse_issue(text, pats):
    """`TARGET: x | WITNESS: y | ISSUE: z` -> (target, witness or None, issue), tolerant of noise."""
    text = (text or "").replace("**", "")
    def part(key):
        m = re.search(r"%s\s*:\s*(.*?)\s*(?=\||\b(?:TARGET|WITNESS|ISSUE)\s*:|$)" % key, text,
                      re.I | re.S)
        return m.group(1).strip() if m else ""
    t, w, issue = part("TARGET"), part("WITNESS"), part("ISSUE")
    th = mentioned(t, pats)
    if len(th) != 1:
        return None
    wh = [s for s in mentioned(w, pats) if s != th[0]]
    issue = issue.split("\n")[0].strip()
    if len(issue) < 8:
        return None
    return th[0], (wh[0] if len(wh) == 1 else None), issue[:300]


class Harness(DCHarness):
    N_FINAL = 5
    STOP_FROM = 12
    STOP_STRIKES = 4
    STOP_LEAD = 3

    def _ask(self, who, question):
        norm = (who, re.sub(r"\W+", " ", question.lower()).strip())
        if norm in self._asked or self.env.remaining <= 0:
            return False
        self._asked.add(norm)
        self.env.interrogate(who, question)
        return True

    def _statements(self):
        return "\n".join("%d. %s (asked: %s) said: %s" % (k + 1, n, q, r)
                         for k, (n, q, r) in enumerate(self.env.history))

    def _issue_log(self):
        return "\n".join("- about %s: %s" % (t, i) for t, _, i in self._issues) or "(none)"

    def _settled(self):
        if len(self.env.history) < self.STOP_FROM or not self._strikes:
            return False
        top = self._strikes.most_common(2)
        lead = top[0][1] - (top[1][1] if len(top) > 1 else 0)
        return top[0][1] >= self.STOP_STRIKES and lead >= self.STOP_LEAD

    def _fallback(self, f):
        counts = Counter(n for n, _, _ in self.env.history)
        for s in sorted(self.env.suspects, key=lambda s: (counts[s], self.env.suspects.index(s))):
            for q in ("Who can confirm where you were around {time}?",
                      "What did you see or hear near {location} around {time}?",
                      "When did you last see {victim}, and what happened?"):
                if self._ask(s, q.format(**f)):
                    return True
        return False

    def solve(self) -> int:
        env = self.env
        f = case_facts(env.init_info)
        self._pats = build_matchers(env.suspects, f["victim"])
        self._asked, self._issues, self._strikes = set(), [], Counter()
        witnessed = set()                                   # (witness, target) pairs already asked

        for s in env.suspects:
            self._ask(s, "Where were you and what were you doing around {time}, and who can "
                         "confirm it?".format(**f))

        while env.remaining > 0 and not self._settled():
            out = self.llm([{"role": "system", "content": env.system_prompt},
                            {"role": "user", "content": AUDIT.format(
                                statements=self._statements(), issues=self._issue_log())}])
            got = parse_issue(out, self._pats)
            done = False
            if got:
                target, witness, issue = got
                self._strikes[target] += 1
                self._issues.append((target, witness, issue))
                if witness and (witness, target) not in witnessed:
                    witnessed.add((witness, target))
                    done = self._ask(witness, "About %s: %s Is that true, as far as you know?"
                                     % (target, issue))
                if not done:
                    done = self._ask(target, "%s How do you explain that?" % issue)
            if not done and not self._fallback(f):
                break

        choices = ", ".join("%s. %s" % (LETTERS[i], s) for i, s in enumerate(env.suspects))
        prompt = (env.system_prompt
                  + "\nInterrogation record:\n" + self._statements()
                  + "\n\nIssues the investigation looked into:\n" + self._issue_log()
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
        strikes = {i: self._strikes[s] for i, s in enumerate(env.suspects)}
        return max(range(len(env.suspects)), key=lambda i: (votes[i], strikes[i], -i))
