"""Ladder rung (d), the ceiling: enum15 plus the max-entropy query spelled
out. A model that simply copies the hint reproduces the oracle policy
(300/300 within budget, mean 11.6 queries). Whatever gap remains below that
is the model failing to copy, not failing to reason."""

from harness_lib import count_line, determined_text, enumeration_text, best_query, fmt

NAME = "oracle_hint"
PARENT = "enum15"
LIST_CAP = 15


def init_state(view):
    return {}


def build_feedback(view, state):
    parts = [count_line(view), determined_text(view), enumeration_text(view, LIST_CAP)]
    q = best_query(view)
    if q is not None:
        parts.append(f"Most informative next query: {fmt(q, view.k)}.")
    return " ".join(p for p in parts if p)


def final_hint(view, state):
    return ""
