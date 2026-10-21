"""A two-line playbook, to show the arm end to end: ask about entries that are still open."""
from ctx_harness import CDCtxHarness


class Harness(CDCtxHarness):
    PLAYBOOK = """
- Before each query, restate which output entries you have already determined, and never query one of those again.
- Prefer a query whose answer you cannot predict from what you already know.
"""
