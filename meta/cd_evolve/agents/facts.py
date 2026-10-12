"""Ladder rung (a): restate the model's own observations compactly, with no
inference at all. Everything in this message is already in the transcript;
if this alone moves repeat% or eff, the bottleneck was retrieval from a long
transcript, not reasoning."""

from harness_lib import observations_text

NAME = "facts"
PARENT = "bare"


def init_state(view):
    return {}


def build_feedback(view, state):
    return observations_text(view)


def final_hint(view, state):
    return observations_text(view)
