"""Ladder rung (c): enum15 plus, for every undetermined entry, how the
remaining configurations split between 0 and 1. Choosing the next query is
now a comparison over at most 24 pairs of numbers instead of a deduction
over circuits. If this restores ex-ante efficiency where enum15 did not,
the compensation boundary sits between 'state' and 'decision table'."""

from harness_lib import count_line, determined_text, enumeration_text, split_table_text

NAME = "splits"
PARENT = "enum15"
LIST_CAP = 15


def init_state(view):
    return {}


def build_feedback(view, state):
    parts = [count_line(view), determined_text(view), split_table_text(view),
             enumeration_text(view, LIST_CAP)]
    return " ".join(p for p in parts if p)


def final_hint(view, state):
    return ""
