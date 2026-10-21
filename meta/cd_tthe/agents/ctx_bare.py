"""The context arm's seed for Circuit Decoding: an empty playbook, so it is the plain loop."""
from ctx_harness import CDCtxHarness


class Harness(CDCtxHarness):
    PLAYBOOK = ""
