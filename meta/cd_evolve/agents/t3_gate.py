"""Seed 3: enum15 content, but only on turns where the T3-style trap proxy
fires (the query just made carried zero information, repeated an earlier
query, or left the feasible set unchanged). Otherwise identical to bare.
The proxy measured precision 0.74 / recall 0.32 on gate_cd.jsonl
(runner/calibrate_t3_proxy.py), so this is expected to under-intervene."""

from harness_lib import count_line, determined_text, enumeration_text

NAME = "t3_gate"
PARENT = "enum15"
LIST_CAP = 15


def init_state(view):
    return {"fired": 0}


def _trapped(view):
    if view.last_bits_expected < 1e-9:
        return True
    if view.history[:-1] and any(q == view.last_q for q, _ in view.history[:-1]):
        return True
    return view.n_before == view.n_after


def build_feedback(view, state):
    if not _trapped(view):
        return ""
    state["fired"] += 1
    parts = [count_line(view), determined_text(view), enumeration_text(view, LIST_CAP)]
    return " ".join(p for p in parts if p)


def final_hint(view, state):
    return ""
