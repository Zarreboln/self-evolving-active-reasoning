"""Static invariant audit for candidate harnesses (TTHE's FROZEN SOLVER + LABEL-FREE), plus the
arm-specific rule on the public hypothesis space.

    python audit.py agents/<name>.py [--arm scratch|lib]
"""

from __future__ import annotations

import ast
import re
import sys

from tasks import SPECS  # noqa: E402

FORBIDDEN = [
    (r"\bopen\(|\bPath\(|json\.load|\bglob\b|pickle|shelve|os\.|sys\.|subprocess|socket|urllib|requests|httpx",
     "file / process / network access"),
    (r"OpenAI\(|base_url|api_key|_solver\b|_client|bridge\.", "touches the solver client or the bridge directly"),
    (r"\beval\(|\bexec\(|__import__|importlib", "dynamic code execution"),
]
ALLOWED_IMPORTS = {"re", "math", "itertools", "collections", "functools", "random", "typing", "dataclasses",
                   "statistics", "heapq", "numpy", "np", "harness_base",
                   # the context arm: a candidate subclasses the fixed loop and changes only PLAYBOOK
                   # (ctx_audit.text_only enforces that separately)
                   "ctx_harness"}
LIB_ONLY = {"cd_oracle"}


def audit(src, arm="scratch", task="cd"):
    problems = []
    spec = SPECS[task]
    for pat, why in [(spec["audit_forbidden"], "reads the truth / instance data")] + FORBIDDEN:
        if re.search(pat, src):
            problems.append(why)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        for n in names:
            if n in ALLOWED_IMPORTS or n == spec["allowed_import"]:
                continue
            if n in LIB_ONLY and arm == "lib":
                continue
            problems.append(f"import of {n} not allowed in arm '{arm}'")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Harness" for n in tree.body):
        problems.append("no class named Harness")
    return problems


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--arm", default="scratch"); p.add_argument("--task", default="cd")
    a = p.parse_args()
    probs = audit(open(a.file).read(), a.arm, a.task)
    print("OK" if not probs else "INVALID: " + "; ".join(probs))
    sys.exit(0 if not probs else 1)
