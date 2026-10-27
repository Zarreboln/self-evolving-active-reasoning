"""What the loop shells out to. Fixes the task and hands off to the shared implementation.

`meta_harness.py` calls `uv run python benchmark.py --logs-dir … --results-dir … <flags>` with no
task argument, because upstream one run directory means one domain. Ours means one AR-Bench task,
so the task is pinned here rather than passed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mh_benchmark import load_results  # noqa: F401,E402  (the loop imports this from here)

TASK = "dc"


def get_model_short_name(model_id: str) -> str:
    """Upstream lowercases; ours keeps the case the result directories were written with."""
    return model_id.split("/")[-1]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--task", TASK] + sys.argv[1:]
    from mh_benchmark import main
    main()
