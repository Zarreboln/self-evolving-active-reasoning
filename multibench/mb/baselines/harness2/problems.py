"""Build the problem file and the answers file the Mac driver needs (run where the environments are
installed, e.g. on ORCD with mb/env.sh sourced; the Mac cannot list instances).

problems.json: {"dev": [[cell, "search", instance], ...]       # development stream, in stream order
                "test": {"<task>": [[cell, "test", instance], ...]}}   # the fixed test order
answers.json:  {"<cell>": {"<split>": {"<instance>": "<answer>"}}}  # for the shared-store leak check

Development stream: the first 20 search problems of each development task (the search split every
method uses), interleaved in one seeded random order (Harness²'s stream_order shuffles tasks with a
seed). Test: when the experiment's shared fixed order exists, pass it with --test-order (same JSON
shape as "test") so every method plays the same order; otherwise a seeded shuffle is written.

    python -m mb.baselines.harness2.problems --out ~/h2_problems [--test-order FILE]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

DEV_CELLS = ["wordle:basic5", "wordle:basic7", "hangman:basic", "babyai:GoToLocal", "babyai:Open",
             "sciworld:test-conductivity", "dc:all"]
TEST_TASKS = {"mastermind_basic": ["mastermind:basic"],
              "alfworld_pick_and_place_simple": ["alfworld:pick_and_place_simple"],
              "twentyq_basic_pooled": ["twentyq:basic-people", "twentyq:basic-places", "twentyq:basic-things"],
              "minesweeper_basic": ["minesweeper:basic"],
              "wordle_hard5": ["wordle:hard5"], "hangman_hard": ["hangman:hard"],
              "sciworld_grow_plant": ["sciworld:grow-plant"]}
N_TEST = {"alfworld_pick_and_place_simple": 24, "sciworld_grow_plant": 33}


def instances(cell, split, n=None):
    from mb.benches import get
    bench, sub = cell.split(":", 1)
    return [str(i) for i in get(bench).instances(sub, split, limit=n)]


def answer(cell, inst):
    """The instance's answer as one short string, from its oracle plan; None where it has none."""
    from mb.benches import get
    bench, sub = cell.split(":", 1)
    try:
        plan = get(bench).make(sub, inst, None)._oracle_plan() or []
    except Exception:
        return None
    if not plan or len(plan) > 3:
        return None
    a = re.sub(r"^(answer|guess)\s*:\s*", "", str(plan[-1]).strip(), flags=re.I).strip("[] ")
    return a if 0 < len(a) <= 60 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-order", help="the experiment's shared fixed test order (JSON like problems['test'])")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dev = [[c, "search", i] for c in DEV_CELLS for i in instances(c, "search", a.n_dev)]
    random.Random(a.seed).shuffle(dev)
    if a.test_order:
        test = json.loads(Path(a.test_order).read_text())
    else:
        test = {}
        for name, cells in TEST_TASKS.items():
            items = [[c, "test", i] for c in cells for i in instances(c, "test")]
            random.Random(f"{a.seed}/{name}").shuffle(items)
            test[name] = items[:N_TEST.get(name, 50)]
    (out / "problems.json").write_text(json.dumps({"dev": dev, "test": test}, indent=1))
    ans = {}
    for c in DEV_CELLS:
        for split in ("search", "selection"):
            for i in instances(c, split, a.n_dev if split == "search" else None):
                if (x := answer(c, i)) is not None:
                    ans.setdefault(c, {}).setdefault(split, {})[i] = x
    for items in test.values():
        for c, split, i in items:
            if (x := answer(c, i)) is not None:
                ans.setdefault(c, {}).setdefault(split, {})[i] = x
    (out / "answers.json").write_text(json.dumps(ans, indent=1))
    print(f"dev {len(dev)} problems; test {{{', '.join(f'{k}: {len(v)}' for k, v in test.items())}}}")


if __name__ == "__main__":
    main()
