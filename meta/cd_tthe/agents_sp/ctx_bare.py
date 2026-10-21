"""The context arm's seed: an empty playbook, so it is exactly AR-Bench's official zero-shot loop."""
from ctx_harness import SPCtxHarness


class Harness(SPCtxHarness):
    PLAYBOOK = ""
