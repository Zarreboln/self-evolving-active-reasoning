"""Label-free signals of one SP episode, computed by the controller from public information only
(questions asked, referee replies, the final text). No oracle exists for SP."""
from __future__ import annotations

import re


def _norm(q):
    q = re.sub(r"^\s*Q\s*:\s*", "", q or "", flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def compute(env, answer_text, crashed=False):
    qs = [_norm(q) for q, _ in env.history]
    ans = [(a or "").strip().lower() for _, a in env.history]
    n = len(qs)
    dist = {k: sum(1 for a in ans if a.startswith(k)) / n for k in ("yes", "no", "unknown")} if n else {"yes": None, "no": None, "unknown": None}
    not_q = sum(1 for a in ans if "not a question" in a) / n if n else None
    return {"health": (not crashed) and bool((answer_text or "").strip()), "crashed": crashed,
            "parsed_answer": bool((answer_text or "").strip()), "n_questions": n,
            "repeat": (n - len(set(qs))) / n if n else None,
            "yes_rate": dist["yes"], "no_rate": dist["no"], "unknown_rate": dist["unknown"], "not_question_rate": not_q,
            "mean_q_words": sum(len(q.split()) for q in qs) / n if n else None,
            "answer_words": len((answer_text or "").split())}
