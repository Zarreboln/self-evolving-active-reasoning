"""The suite: one uniform handle per bench, imported lazily (ScienceWorld and ALFWorld are heavy).

    from mb.benches import get
    b = get("alfworld")
    b.subtasks, b.dimensions, b.responder
    b.instances(subtask, split, limit=None) -> [instance key]
    b.make(subtask, instance, responder=None) -> TextEnv
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

BENCHES = ["wordle", "twentyq", "gn", "dc", "sp",
           "mastermind", "hangman", "minesweeper", "guessnumber", "guesswho",
           "babyai", "alfworld", "sciworld"]
_AR = {"gn", "dc", "sp"}
_TA = {"mastermind", "hangman", "minesweeper", "guessnumber", "guesswho"}


def get(name):
    if name in _TA:
        m = importlib.import_module("mb.benches.tagames")
        return SimpleNamespace(
            name=name, subtasks=list(m.GAMES[name]), dimensions=m.DIMENSIONS[name],
            responder=m.RESPONDER[name],
            instances=lambda subtask, split, limit=None: m.instances(name, subtask, split, limit),
            make=lambda subtask, instance, responder=None: m.make(name, subtask, instance, responder))
    if name in _AR:
        m = importlib.import_module("mb.benches.arbench")
        return SimpleNamespace(
            name=name, subtasks=list(m.SUBTASKS), dimensions=m.DIMENSIONS[name],
            responder=m.RESPONDER[name],
            instances=lambda subtask, split, limit=None: m.instances(name, subtask, split, limit),
            make=lambda subtask, instance, responder=None: m.make(name, subtask, instance, responder))
    m = importlib.import_module(f"mb.benches.{name}")
    return SimpleNamespace(name=name, subtasks=list(m.SUBTASKS), dimensions=m.DIMENSIONS,
                           responder=m.RESPONDER, instances=m.instances, make=m.make)
