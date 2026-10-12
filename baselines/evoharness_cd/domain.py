from __future__ import annotations

from typing import Any

from ...domain import Domain, Task
from ...experience import ExperienceDomain
from ...prompts import ACCUMULATED
from .belief import CircuitDecodingBelief
from .env import CircuitDecodingEnv, list_tasks
from .prompts import PROMPT


class CircuitDecodingExperience(ExperienceDomain):
    agent_description = "a circuit-deduction agent"
    task_label = "INVENTORY"
    default_task_type = "general"
    # One family per gate inventory of the released set; keywords are the inventory texts.
    task_types = {
        "and1_or5_not2": ("1 and gate, 5 or gates, 2 not gates",),
        "and3_or3_not0": ("3 and gates, 3 or gates",),
        "and3_or3_not1": ("3 and gates, 1 not gate, 3 or gates",),
    }
    first_match = True
    priority_rule = "CircuitDecoding keeps no search priorities; leave search_priorities empty."
    priority_example = ('"(unused)"', "[]")


class CircuitDecodingDomain(Domain):
    name = "circuitdecoding"
    prompt = PROMPT
    experience = CircuitDecodingExperience()
    # The paper's protocol is one growing conversation (the model sees every
    # earlier query and answer), so the accumulated regime is the faithful one.
    context = ACCUMULATED
    hints_label = "RECALLED HINTS"

    def make_belief(self) -> CircuitDecodingBelief:
        return CircuitDecodingBelief()

    def make_env(self, worker: int = 0, data_path: str | None = None, **_: Any) -> CircuitDecodingEnv:
        return CircuitDecodingEnv(data_path=data_path)

    def list_tasks(self, split: str, split_file: str | None = None, data_path: str | None = None,
                   **_: Any) -> list[Task]:
        return list_tasks(split, split_file=split_file, data_path=data_path)

    def format_actions(self, actions) -> str:
        # 24 query strings every turn is noise; show the family once.
        if len(actions) > 6 and all(a.startswith("query") for a in actions):
            return "query X(a, b, c) for X in {A, B, C} and a, b, c in {0, 1}"
        return ", ".join(f"'{a}'" for a in actions)
