"""bare + small repairs, everything else is the official AR-Bench loop.

1. Stall nudge: when the last STALL referee replies contain no "Yes", the turn prompt tells the
   model that line of questioning is not working and asks it to switch to a different aspect
   (bare traces show 10-18 turn runs of near-identical questions all answered No/Unknown).
2. Final recap: before AR-Bench's final-answer prompt, the model is shown the referee's Yes/No
   replies grouped as facts and asked to give one concrete explanation consistent with them
   (bare final answers asserted things the referee had answered No to).
3. Mental-state re-ask: a question about what a character feels, thinks, believes, suspects or is
   aware of is re-asked once with a note to ask about what happened instead (parent traces i82/i93
   spent 8-9 questions on Sarah's/Emma's feelings, drawing Unknown or uninformative Yes). The
   re-asked reply is sent unless it is a What/How-style question (the referee rejects those as
   "not a question"), in which case the original question is sent.
4. NO-claim revision: if the final answer reuses most of the distinctive words of a question the
   referee answered No, the model is shown those questions and asked to rewrite its answer without
   them, deleting only that claim (parent final answers i20/i52/i94 copied No-answered questions in
   as true)."""
import re

from sp_harness_base import SPHarness

STALL = 4
MENTAL = re.compile(r"\b(feel|feels|felt|feeling|suspect\w*|believ\w*|think|thinks|thought|consider\w*|"
                    r"wonder\w*|aware|theor\w*)\b", re.I)
WH = re.compile(r"\s*(Q\s*:)?\s*(what|why|how|who|whom|whose|where|when|which)\b", re.I)
STOP = set("a an the of to in on at for by with from and or is was were are be been did does do has had have "
           "it its this that these those his her him she he they them their someone something any some as so but "
           "not no yes could would should can may might possible there what who how why when where which about "
           "into than then just such other ever being because if only own very more most also".split())


def _verdict(reply):
    r = (reply or "").strip().lower()
    if r.startswith("a:"):  # the referee sometimes answers "A: Yes"
        r = r[2:].strip()
    if r.startswith("yes"):
        return "yes"
    if r.startswith("no"):
        return "no"
    return "other"


def _words(text):
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w
            for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in STOP and len(w) > 2}


def _clean(q):
    return q.strip()[2:].strip() if q.strip()[:2].upper() == "Q:" else q.strip()


class Harness(SPHarness):
    def solve(self):
        env = self.env
        msgs = [{"role": "system", "content": env.system_prompt}]
        for t in range(1, env.budget + 1):
            prompt = env.turn_prompt(t)
            recent = [_verdict(a) for _, a in env.history[-STALL:]]
            if len(recent) == STALL and "yes" not in recent:
                prompt += (f"\n(Note: your last {STALL} questions got no \"Yes\". That line of questioning "
                           "is not working -- do not ask another variation of it. Ask about a different, "
                           "more basic aspect of the puzzle, e.g. whether something the puzzle seems to "
                           "state or assume is really true.)")
            msgs.append({"role": "user", "content": prompt})
            q = self.llm(msgs) or ""
            if MENTAL.search(q):
                retry = msgs + [{"role": "assistant", "content": q},
                                {"role": "user", "content": (
                                    "That question asks what someone feels, thinks, believes, suspects or is "
                                    "aware of. The story records what happened, not feelings, so such questions "
                                    "get \"Unknown\" or an uninformative answer. Ask instead about what actually "
                                    "happened: who did what, how, or why -- as one yes/no question. "
                                    "Reply in the format Q: [Your question]")}]
                alt = self.llm(retry) or ""
                if alt.strip() and not WH.match(alt):  # a What/How question is "not a question" to the referee
                    q = alt
            msgs.append({"role": "assistant", "content": q})
            a = env.ask(q)
            msgs.append({"role": "user", "content": a})
        msgs.append({"role": "user", "content": self._recap() + env.final_prompt})
        ans = self.llm(msgs) or ""
        hits = self._no_claims(ans)
        if hits:
            fix = msgs + [{"role": "assistant", "content": ans},
                          {"role": "user", "content": (
                              "The referee answered NO to these questions, so they are false:\n"
                              + "\n".join(f"- {h}" for h in hits)
                              + "\nIf your answer states any of them as true, delete only the words that state it "
                                "and keep everything else as it is. Do not add new explanations and do not mention "
                                "what is false. Otherwise repeat your answer unchanged.\n"
                                "your answer format:\nA: [Your answer]")}]
            ans = self.llm(fix) or ans
        return ans

    def _no_claims(self, ans):
        """No-answered questions whose distinctive words (not in the puzzle text) mostly reappear in ans."""
        aw, surf = _words(ans), _words(self.env.surface)
        hits = []
        for q, a in self.env.history:
            if _verdict(a) != "no":
                continue
            qw = _words(q) - surf
            shared = qw & aw
            if len(shared) >= 3 and len(shared) >= 0.6 * len(qw):
                hits.append(_clean(q))
        return hits

    def _recap(self):
        yes = [q for q, a in self.env.history if _verdict(a) == "yes"]
        no = [q for q, a in self.env.history if _verdict(a) == "no"]
        lines = ["Questions the referee answered YES (these are true):"]
        lines += [f"- {_clean(q)}" for q in yes] or ["- (none)"]
        lines.append("Questions the referee answered NO (these are false):")
        lines += [f"- {_clean(q)}" for q in no] or ["- (none)"]
        lines.append("Your answer must agree with every YES and must not claim anything answered NO. "
                     "Give one concrete explanation of what actually happened, not a list of possibilities.")
        return "\n".join(lines) + "\n"
