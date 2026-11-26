"""The shared-store rule of notes/exp_b_design.md for Harness²'s library.

"No game or benchmark names and no answers to specific problems in any shared store." Harness²'s
library stores whole adopted harness trees plus EXPERIENCE.md, so every tree is checked when it is
proposed (a violation is fed back to the editor as a rejection, like any validation error) and again,
with its experience text, before it enters the library (a violation keeps it out).

Answers: mb/evolve/check.leaks, the project's test, comparing how many answers of problems the
method has seen the store contains against answers of a split it never saw (selection). The answer
lists come from an answers file prepared where the environments live (see problems.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from mb.evolve.check import leaks

# Names of the games, environments and suites in experiment B (development and test), and of the
# code bases they come from. Matched case-insensitively as whole words.
NAMES = ["wordle", "hangman", "babyai", "baby ai", "baby-ai", "scienceworld", "science world", "sciworld",
         "alfworld", "alf world", "mastermind", "master mind", "minesweeper", "mine sweeper",
         "twenty questions", "20 questions", "20q", "twentyq", "ar-bench", "arbench", "ar bench",
         "textarena", "text arena", "balrog", "swiftsage", "minigrid", "textworld",
         "detective cases", "situation puzzles", "guess my number"]


def name_hits(text: str, names=NAMES) -> list[str]:
    low = (text or "").lower()
    return sorted({n for n in names if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", low)})


class Answers:
    """Answers of problems the method has seen so far, and of a split it never sees.

    file: {"<cell>": {"<split>": {"<instance id>": "<answer text>"}}}; `unseen` = every selection
    split answer. Without a file only the names check runs (recorded in the run's meta).
    """

    def __init__(self, path: str | Path | None = None, table: dict | None = None):
        self.table = table if table is not None else (json.loads(Path(path).read_text()) if path else None)
        self.seen_keys: set[tuple[str, str, str]] = set()

    @property
    def active(self):
        return self.table is not None

    def mark_seen(self, cell, split, inst):
        self.seen_keys.add((cell, split, str(inst)))

    def seen(self):
        t = self.table or {}
        return sorted({t[c][s][i] for c, s, i in self.seen_keys if i in t.get(c, {}).get(s, {})})

    def unseen(self):
        return sorted({a for c in (self.table or {}).values() for a in c.get("selection", {}).values()})


def violations(texts: dict[str, str], answers: Answers | None) -> list[str]:
    out = []
    for rel, body in sorted(texts.items()):
        hits = name_hits(body) + name_hits(rel)
        if hits:
            out.append(f"{rel} names a specific game or task collection ({', '.join(hits)}); describe the "
                       f"situation by its shape instead")
    if answers is not None and answers.active:
        lk = leaks("\n".join(texts.values()), answers.seen(), answers.unseen())
        if lk:
            out.append("the harness holds answers of specific problems: " + lk)
    return out


def audit_dir(d: Path, answers: Answers | None) -> list[str]:
    """The rule over everything a store directory holds (used on the library at the end of a run)."""
    texts = {str(p.relative_to(d)): p.read_text(errors="replace") for p in sorted(Path(d).rglob("*")) if p.is_file()}
    return violations(texts, answers)
