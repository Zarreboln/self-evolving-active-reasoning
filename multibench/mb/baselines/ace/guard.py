"""Rules for shared stores (notes/exp_b_design.md, "Rules for all methods"), applied to ACE's playbook.

1. No game or benchmark names: every Curator ADD is passed through `redact_names` before it is applied
   (the name is replaced by "this game"; the redaction is logged).
2. No answers to specific problems: an ADD whose text contains the answer of a problem this phase has
   already played is dropped (`answer_hits`; logged). Only instance ids that are words (letters, at
   least 3) are checked; numeric seeds and file ids cannot be copied usefully and would match numbers.
3. Answer copying, the design's statistical test: `copying` runs mb.evolve.check.leaks on the final
   offline playbook per development task (answers of the search split it saw against the selection
   split it never saw).
"""
from __future__ import annotations

import re

NAMES = [r"wordle", r"hangman", r"baby\s?-?ai", r"science\s?-?world", r"sci\s?world", r"alf\s?-?world",
         r"master\s?-?mind", r"mine\s?-?sweeper", r"text\s?-?arena", r"balrog", r"swift\s?-?sage",
         r"ar\s?-?bench", r"twenty[\s-]+questions", r"20\s?-?q(?:uestions)?", r"detective\s+cases?",
         r"go\s?to\s?local", r"pick_and_place_simple", r"test-conductivity", r"grow-plant"]
NAME_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(NAMES) + r")(?![A-Za-z0-9])", re.I)


def redact_names(text: str):
    hits = NAME_RE.findall(text or "")
    return NAME_RE.sub("this game", text or ""), hits


def has_names(text: str) -> bool:
    return bool(NAME_RE.search(text or ""))


def checkable(answer: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z' -]{2,}", answer or ""))


def answer_hits(text: str, answers) -> list[str]:
    low = (text or "").lower()
    return [a for a in answers if checkable(a) and
            re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", low)]


def clean_operations(ops, played_answers):
    """Returns (kept operations with names redacted, log entries)."""
    kept, log = [], []
    for op in ops:
        if not isinstance(op, dict):
            kept.append(op)
            continue
        op = dict(op)
        for field in ("content", "section"):
            new, hits = redact_names(str(op.get(field, "")))
            if hits:
                log.append({"redacted": hits, "field": field})
                op[field] = new
        bad = answer_hits(str(op.get("content", "")) + " " + str(op.get("section", "")), played_answers)
        if bad:
            log.append({"dropped_for_answers": bad, "content": str(op.get("content", ""))[:200]})
            continue
        kept.append(op)
    return kept, log


def copying(playbook: str, seen, unseen):
    from mb.evolve.check import leaks
    return leaks(playbook, list(seen), list(unseen))
