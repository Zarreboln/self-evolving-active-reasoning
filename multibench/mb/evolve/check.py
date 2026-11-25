"""Static check of a proposed harness before it is run anywhere.

A candidate is one Python file defining `run_episode(env, chat)`. It runs in a child process that
only exchanges text with the episode (mb/evolve/child.py), so it cannot reach the environment
object or the secret it holds. This check closes the other ways out of that child: imports are
limited to a small standard-library allowlist plus harness_lib, the builtins that open files,
import or execute code, or walk the object graph are banned, and no attribute or name may start
with an underscore (that blocks `().__class__.__subclasses__()` and the like).

The content rule is HW2's: nothing about specific instances may be hard-coded. General knowledge
(a vocabulary, a strategy) is allowed. Since lost games reveal their answer in the trace, `leaks`
checks that no answer of the split the proposer has seen appears in the file; it runs where the
instance lists live (bath), right before evaluation.

    python -m mb.evolve.check path/to/harness.py
"""
from __future__ import annotations

import ast
import re
import sys

ALLOWED_IMPORTS = {"re", "json", "math", "itertools", "collections", "random", "string",
                   "functools", "heapq", "typing", "dataclasses", "harness_lib"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview", "help",
                "exit", "quit", "dir", "type", "object", "super", "classmethod", "staticmethod"}
MAX_BYTES = 200_000


def problems(src: str) -> list[str]:
    out = []
    if len(src.encode()) > MAX_BYTES:
        out.append(f"file is larger than {MAX_BYTES} bytes")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return out + [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    out.append(f"line {node.lineno}: import {a.name} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                out.append(f"line {node.lineno}: from {node.module} import is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            out.append(f"line {node.lineno}: name {node.id!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("_") and node.id != "_":
            out.append(f"line {node.lineno}: names starting with '_' are not allowed ({node.id})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            out.append(f"line {node.lineno}: attributes starting with '_' are not allowed (.{node.attr})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name.startswith("__"):
            out.append(f"line {node.lineno}: dunder definitions are not allowed ({node.name})")
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and any(n.startswith("_") for n in node.names):
            out.append(f"line {node.lineno}: names starting with '_' are not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run_episode" for n in tree.body):
        out.append("no top-level `def run_episode(env, chat)`")
    return out


def _hits(low, answers):
    return [a for a in answers if re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", low)]


def leaks(src: str, seen, unseen) -> str | None:
    """Whether the file holds answers of the seen split more than a general vocabulary would.

    A vocabulary contains many answers of every split, so a hit alone proves nothing. What copying
    from the traces leaves is an excess: the seen answers covered at a clearly higher rate than
    answers of a split the proposer never saw (`unseen`, e.g. selection). Flagged when the seen
    hits exceed their expectation under the unseen rate by more than 2 + 2 sd.
    """
    low = src.lower()
    hs, hu = _hits(low, seen), _hits(low, unseen)
    rate = len(hu) / max(len(unseen), 1)
    exp = rate * len(seen)
    sd = (len(seen) * rate * (1 - rate)) ** 0.5
    if len(hs) > exp + 2 + 2 * sd:
        return (f"holds {len(hs)}/{len(seen)} answers of the seen split against {exp:.1f} expected from "
                f"its coverage of unseen answers ({len(hu)}/{len(unseen)}): hard-coded answers")
    return None


if __name__ == "__main__":
    p = problems(open(sys.argv[1]).read())
    print("\n".join(p) or "ok")
    sys.exit(1 if p else 0)
