"""Rules for the meta-skill bank, a shared store (notes/exp_b_design.md, "Rules for all methods"):
no game or benchmark names, and no answers to specific problems.

    names_in(text)                         names of games, benchmarks or task identifiers found in text
    answer_leak(text, seen, unseen)        check.py's `leaks` rule: seen-split answers covered clearly more
                                           often than unseen ones (copying from traces)
    bank_problems(skills, answers)         both, over the when/provide/use text of every skill
"""
from __future__ import annotations

import re

from mb.evolve.check import _hits, leaks

# Benchmarks, games and task identifiers of the suite (mb/benches/), and the frameworks that ship them.
NAMES = [
    "wordle", "hangman", "mastermind", "minesweeper", "mine sweeper", "twenty questions", "20 questions",
    "20q", "twentyq", "guess the number", "guessthenumber", "guess who", "guesswho", "babyai", "baby ai",
    "minigrid", "scienceworld", "science world", "sciworld", "alfworld", "alf world", "textworld",
    "textarena", "text arena", "balrog", "ar-bench", "arbench", "detective case", "detective cases",
    "situation puzzle", "circuit decoding", "boxinggym", "boxing gym", "blicket", "newtonbench",
    "harness-bench", "gotolocal", "go to local", "test-conductivity", "test conductivity", "grow-plant",
    "grow plant", "pick_and_place", "pick and place", "swiftsage",
]
_NAME_RE = [(n, re.compile(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])")) for n in NAMES]


def names_in(text: str) -> list[str]:
    low = (text or "").lower()
    return [n for n, r in _NAME_RE if r.search(low)]


def skill_text(skill: dict) -> str:
    return "\n".join(str(skill.get(k, "")) for k in ("skill_id", "when", "provide", "use"))


def answer_leak(text: str, seen: list[str], unseen: list[str]) -> str | None:
    return leaks(text, seen, unseen)


def bank_problems(skills: list[dict], answers: dict | None) -> list[str]:
    """answers: {cell: {"search": [...], "selection": [...]}} of the development tasks, or None when the
    answer lists are not available (then only the name rule is applied, and the caller records that)."""
    text = "\n".join(skill_text(s) for s in skills)
    out = []
    found = names_in(text)
    if found:
        out.append(f"names a game, benchmark or task: {', '.join(found)}")
    for cell, a in (answers or {}).items():
        lk = answer_leak(text, a.get("search", []), a.get("selection", []))
        if lk:
            out.append(f"answers of {cell}: {lk}")
    return out


def answer_hits(skills: list[dict], answers: dict | None) -> dict:
    """For the record: how many seen and unseen answers of each task appear in the bank."""
    text = "\n".join(skill_text(s) for s in skills).lower()
    return {cell: {"search": len(_hits(text, a.get("search", []))), "selection": len(_hits(text, a.get("selection", [])))}
            for cell, a in (answers or {}).items()}
