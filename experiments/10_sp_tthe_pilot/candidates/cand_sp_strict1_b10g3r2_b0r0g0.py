"""bare + two small repairs, everything else is the official AR-Bench loop.

1. Stall nudge: when the last STALL referee replies contain no "Yes", the turn prompt tells the
   model that line of questioning is not working and asks it to switch to a different aspect
   (bare traces show 10-18 turn runs of near-identical questions all answered No/Unknown).
2. Final recap: before AR-Bench's final-answer prompt, the model is shown the referee's Yes/No
   replies grouped as facts and asked to give one concrete explanation consistent with them
   (bare final answers asserted things the referee had answered No to)."""
from sp_harness_base import SPHarness

STALL = 4


def _verdict(reply):
    r = (reply or "").strip().lower()
    if r.startswith("a:"):  # the referee sometimes answers "A: Yes"
        r = r[2:].strip()
    if r.startswith("yes"):
        return "yes"
    if r.startswith("no"):
        return "no"
    return "other"


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
            msgs.append({"role": "assistant", "content": q})
            a = env.ask(q)
            msgs.append({"role": "user", "content": a})
        msgs.append({"role": "user", "content": self._recap() + env.final_prompt})
        return self.llm(msgs) or ""

    def _recap(self):
        yes = [q for q, a in self.env.history if _verdict(a) == "yes"]
        no = [q for q, a in self.env.history if _verdict(a) == "no"]
        clean = lambda q: q.strip()[2:].strip() if q.strip()[:2].upper() == "Q:" else q.strip()
        lines = ["Questions the referee answered YES (these are true):"]
        lines += [f"- {clean(q)}" for q in yes] or ["- (none)"]
        lines.append("Questions the referee answered NO (these are false):")
        lines += [f"- {clean(q)}" for q in no] or ["- (none)"]
        lines.append("Your answer must agree with every YES and must not claim anything answered NO. "
                     "Give one concrete explanation of what actually happened, not a list of possibilities.")
        return "\n".join(lines) + "\n"
