"""Seed 0: the environment line only. Reproduces the `bare` condition of
results/gate_cd.jsonl (runner/cd_runner_async.py)."""

NAME = "bare"
PARENT = None


def init_state(view):
    return {}


def build_feedback(view, state):
    return ""


def final_hint(view, state):
    return ""
