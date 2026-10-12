"""Seed 2: enum15 without the enumeration -- count plus determined entries
only. Isolates the symbolic-summary half of the hand-written harness
(GN showed list vs symbolic was a 19.2% vs 73.0% dead-guess difference)."""

from harness_lib import count_line, determined_text

NAME = "symbolic"
PARENT = "enum15"


def init_state(view):
    return {}


def build_feedback(view, state):
    parts = [count_line(view), determined_text(view)]
    return " ".join(p for p in parts if p)


def final_hint(view, state):
    return ""
