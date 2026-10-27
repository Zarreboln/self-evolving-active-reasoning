"""Prior-gated verification: trust the background unless it is split, and then test only the split.

On this task asking nothing (~40%) beats every published interrogation method (31-35%): a long
turn-ordered record of suspects who are free to deflect, defend or mislead pulls the final judgment
off what the background already gave away. This harness makes the background an explicit prior and
spends turns only where that prior is uncertain.

  * prior: five independent background-only accusations (AR-Bench's own final prompt on an empty
    record, sampled at the solver's temperature), read by a parser that does not take a capital
    letter out of the suspect's name for the answer.
  * stopping: if four of the five agree, accuse at once -- zero turns.
  * otherwise, verification: the two (or, on a tie, three) suspects the prior is split between are
    each examined by templated questions -- one alibi question to the contender, and third-party
    questions about the contender (whereabouts, the weapon, a reason) to everyone else.
  * decision: the verification record is judged five more times under a prompt that ranks
    corroborated facts above self-report; prior votes and post votes are added, and ties go to the
    prior.
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


class Harness(DCHarness):
    N_PRIOR = 5
    CONSENSUS = 4
    N_POST = 5

    def _votes(self, prompt, n):
        votes = Counter()
        for _ in range(n):
            idx = parse_answer(self.llm([{"role": "user", "content": prompt}]), self._pats,
                               self.env.suspects)
            if idx is not None:
                votes[idx] += 1
        return votes

    def _ask(self, who, question):
        if self.env.remaining > 0:
            self.env.interrogate(who, question)

    def solve(self) -> int:
        env = self.env
        f = case_facts(env.init_info)
        self._pats = build_matchers(env.suspects, f["victim"])

        prior = self._votes(env.final_prompt, self.N_PRIOR)   # empty record: background only
        if not prior:
            return 0
        ranked = [i for i, _ in prior.most_common()]
        top, top_n = prior.most_common(1)[0]
        if top_n >= self.CONSENSUS or (len(prior) == 1 and top_n >= 3):
            return top

        second_n = prior[ranked[1]] if len(ranked) > 1 else 0
        contenders = [i for i in ranked if prior[i] >= max(second_n, 1)][:3]
        if len(contenders) < 2:                              # one name only: test it against a
            contenders.append(next(i for i in range(len(env.suspects)) if i not in contenders))

        topics = [
            "Around {time}, did you see {c} anywhere near {location}? What were they doing?",
            "Could {c} have gotten hold of {weapon}? How do you know?",
            "Did {c} have any reason to want {victim} dead?",
        ]
        for ci in contenders:
            c = env.suspects[ci]
            self._ask(c, "Where exactly were you around {time}, and who can confirm it?".format(**f))
            witnesses = [s for s in env.suspects if s != c]
            for j, w in enumerate(witnesses):
                self._ask(w, topics[j % len(topics)].format(c=c, **f))

        record = "\n".join("- %s was asked: %s\n  %s answered: %s" % (n, q, n, r)
                           for n, q, r in env.history)
        choices = ", ".join("%s. %s" % (LETTERS[i], s) for i, s in enumerate(env.suspects))
        post_prompt = (
            env.system_prompt
            + "\nInterrogation record:\n" + record + "\n\n" + GUIDE.format(**f) + "\n\n"
            + "Now tell me who is more likely to be the true murderer, %s.\n" % choices
            + "You should only output the index of the candidate suspect like: A, B, C, D or E.\n"
            + "you should strictly follow this answer format:\n"
            + "Reason: [Your inference step by step]\nAnswer: [A, B, C, D or E]\n")
        post = self._votes(post_prompt, self.N_POST)

        def key(i):
            return (prior[i] + post[i], prior[i], post[i])
        return max(range(len(env.suspects)), key=key)
