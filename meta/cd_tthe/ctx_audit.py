"""Enforce that a context-arm candidate changes text and nothing else.

The context arm is only a controlled comparison against the code arm if the proposer cannot
smuggle logic into it. This check parses the candidate and accepts exactly one shape: a module
docstring, one import of `ctx_harness`, and one class holding a single string assignment named
PLAYBOOK. Anything that could compute — a function, a loop, an f-string, a second assignment —
is a violation, and the candidate is discarded like any audit failure.

    python ctx_audit.py agents_sp/<name>.py
"""

from __future__ import annotations

import ast
import sys

BASES = {"SPCtxHarness", "CDCtxHarness"}


def text_only(src: str):
    """Return a list of violations; empty means the candidate changes only the playbook."""
    bad = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"does not parse: {e}"]

    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]                                   # module docstring is fine

    cls = None
    for node in body:
        if isinstance(node, ast.ImportFrom):
            if node.module != "ctx_harness":
                bad.append(f"imports {node.module}, only ctx_harness is allowed")
            for a in node.names:
                if a.name not in BASES:
                    bad.append(f"imports {a.name}, only {' / '.join(sorted(BASES))} are allowed")
        elif isinstance(node, ast.Import):
            bad.append(f"plain import of {', '.join(a.name for a in node.names)}; not allowed")
        elif isinstance(node, ast.ClassDef):
            if cls is not None:
                bad.append("more than one class")
            cls = node
        else:
            bad.append(f"top-level {type(node).__name__}; only the docstring, the import "
                       f"and the class are allowed")

    if cls is None:
        return bad + ["no class defined"]
    if not any(isinstance(b, ast.Name) and b.id in BASES for b in cls.bases):
        bad.append(f"class must inherit from one of {' / '.join(sorted(BASES))}")

    members = list(cls.body)
    if members and isinstance(members[0], ast.Expr) and isinstance(members[0].value, ast.Constant) \
            and isinstance(members[0].value.value, str):
        members = members[1:]                             # class docstring is fine

    assigns = 0
    for node in members:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PLAYBOOK":
            assigns += 1
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                bad.append("PLAYBOOK must be a plain string literal, not a computed value")
        elif isinstance(node, ast.Pass):
            continue
        else:
            bad.append(f"class body contains {type(node).__name__}; only PLAYBOOK = \"...\" is allowed")
    if assigns == 0:
        bad.append("no PLAYBOOK assignment")
    if assigns > 1:
        bad.append("PLAYBOOK assigned more than once")
    return bad


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    v = text_only(open(sys.argv[1]).read())
    if v:
        print("REJECTED — the context arm may change only the playbook:")
        for x in v:
            print("  -", x)
        sys.exit(1)
    print("ok: changes only the playbook")


if __name__ == "__main__":
    main()
