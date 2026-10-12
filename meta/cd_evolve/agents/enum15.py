"""Seed 1: the hand-written belief summary, verbatim from
runner/cd_runner_async.py belief_summary(list_cap=15). Reproduces the
`belief_injected` condition of results/gate_cd.jsonl.

Per turn: count of remaining configurations, the entries every remaining
configuration agrees on (with 'do not ask these again'), and -- once at
most 15 remain -- the full list of remaining joint truth tables."""

from harness_lib import count_line, determined_text, enumeration_text

NAME = "enum15"
PARENT = "bare"
LIST_CAP = 15


def init_state(view):
    return {}


def build_feedback(view, state):
    parts = [count_line(view), determined_text(view), enumeration_text(view, LIST_CAP)]
    return " ".join(p for p in parts if p)


def final_hint(view, state):
    return ""
