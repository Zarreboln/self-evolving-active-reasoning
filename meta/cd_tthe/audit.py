"""Static invariant audit for candidate harnesses (TTHE's FROZEN SOLVER + LABEL-FREE), plus the
arm-specific rule on the public hypothesis space.

    python audit.py agents/<name>.py [--arm scratch|lib]
"""

from __future__ import annotations

import ast
import re
import sys

FORBIDDEN = [
    (r"_truth|\btruth\b|INSTANCES|load_instances|cd_test|circuit_joint_lookup", "reads the truth / instance data"),
    (r"\bopen\(|\bPath\(|json\.load|\bglob\b|pickle|shelve|os\.|sys\.|subprocess|socket|urllib|requests|httpx",
     "file / process / network access"),
    (r"OpenAI\(|base_url|api_key|_solver\b|_client|bridge\.", "touches the solver client or the bridge directly"),
    (r"\beval\(|\bexec\(|__import__|importlib", "dynamic code execution"),
]
ALLOWED_IMPORTS = {"re", "math", "itertools", "collections", "functools", "random", "typing", "dataclasses",
                   "statistics", "heapq", "numpy", "np", "harness_base"}
LIB_ONLY = {"cd_oracle"}


def audit(src, arm="scratch"):
    problems = []
    for pat, why in FORBIDDEN:
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
            if n in ALLOWED_IMPORTS:
                continue
            if n in LIB_ONLY and arm == "lib":
                continue
            problems.append(f"import of {n} not allowed in arm '{arm}'")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Harness" for n in tree.body):
        problems.append("no class named Harness")
    return problems


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--arm", default="scratch")
    a = p.parse_args()
    probs = audit(open(a.file).read(), a.arm)
    print("OK" if not probs else "INVALID: " + "; ".join(probs))
    sys.exit(0 if not probs else 1)
