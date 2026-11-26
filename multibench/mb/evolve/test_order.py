"""The fixed test order shared by every method in experiment B (notes/exp_b_design.md).

Each test task's problems are the first N of its test split, in the split's own order: the same problems
`broker run --n N` plays, so the batch-evaluated methods (Meta-Harness, ours before deployment) and the
methods that play one problem at a time (ACE, Harness², Meta-Skills, our deployment) see the same
problems in the same order. 20Q basic pools three themes: their first problems, taken in turn
(people, places, things, people, ...), 50 in all.

Run where the environments are installed (ORCD, mb/env.sh sourced):
    python -m mb.evolve.test_order --out ~/expb_order
writes
    order.json            {task: [[cell, instance], ...]}          the canonical file; ACE reads this
    order_harness2.json   {task: [[cell, "test", instance], ...]}  for harness2/problems.py --test-order
    order_metaskill.json  {task: [{"cell", "instance"}, ...]}      for metaskill/driver.py --order
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEST_TASKS = {"mastermind_basic": ["mastermind:basic"],
              "alfworld_pick_and_place_simple": ["alfworld:pick_and_place_simple"],
              "twentyq_basic_pooled": ["twentyq:basic-people", "twentyq:basic-places", "twentyq:basic-things"],
              "minesweeper_basic": ["minesweeper:basic"],
              "wordle_hard5": ["wordle:hard5"], "hangman_hard": ["hangman:hard"],
              "sciworld_grow_plant": ["sciworld:grow-plant"]}
N_TEST = {"alfworld_pick_and_place_simple": 24, "sciworld_grow_plant": 33}      # others: 50


def build() -> dict:
    from mb.benches import get
    order = {}
    for task, cells in TEST_TASKS.items():
        n = N_TEST.get(task, 50)
        lists = [[str(i) for i in get(c.split(":", 1)[0]).instances(c.split(":", 1)[1], "test")] for c in cells]
        items, k = [], 0
        while len(items) < n and any(k < len(l) for l in lists):
            items += [[c, l[k]] for c, l in zip(cells, lists) if k < len(l)]
            k += 1
        if len(items) < n:
            raise SystemExit(f"{task}: only {len(items)} test problems, the design needs {n}")
        order[task] = items[:n]
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    order = build()
    (out / "order.json").write_text(json.dumps(order, indent=1))
    (out / "order_harness2.json").write_text(json.dumps(
        {t: [[c, "test", i] for c, i in v] for t, v in order.items()}, indent=1))
    (out / "order_metaskill.json").write_text(json.dumps(
        {t: [{"cell": c, "instance": i} for c, i in v] for t, v in order.items()}, indent=1))
    print({t: len(v) for t, v in order.items()}, "total", sum(map(len, order.values())))


if __name__ == "__main__":
    main()
