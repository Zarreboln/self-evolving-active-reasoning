"""Turn the Builder's components file into one harness file of our interface, and check its interface.

    harness.py = components.py (written by the Builder) + runner (fixed, runner_template.py)

Interface failures (interface_check.py) are what an interface repair is for (paper B.2: at most two).
check.py's static rules then apply to the whole file, as to every harness in experiment B.
"""
from __future__ import annotations

import json

from mb.baselines.metaskill import official
from mb.baselines.metaskill.config import HERE
from mb.baselines.metaskill.interface_check import component_problems
from mb.evolve.check import problems as static_problems


def neutral_components() -> str:
    """H0: every optional component disabled, the released neutral instruction."""
    src = (HERE / "components_template.py").read_text()
    return src.replace('"@@NEUTRAL_INSTRUCTION@@"', json.dumps(official.neutral_instruction()))


def runner_source() -> str:
    src = (HERE / "runner_template.py").read_text()
    return src.replace('"@@BOUNDARY@@"', json.dumps(official.runner_boundary()))


def assemble(components_src: str) -> str:
    return components_src.rstrip() + "\n\n\n" + runner_source()


def harness_problems(components_src: str) -> list[str]:
    """Interface problems of the components, then check.py's static rules on the assembled file."""
    p = component_problems(components_src)
    if p:
        return p
    return static_problems(assemble(components_src))
