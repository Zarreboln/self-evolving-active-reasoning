"""Cluster side, CPU only, once: what the Mac-side driver needs from the environments.

    cards/<cell>/first_search.json   public card of the first search problem of each development task
    cards/<cell>/<id>.json           public card of each test problem (id = the opaque TextEnv.instance_id)
    answers.json                     {cell: {"search": first 20 keys, "selection": all keys}} of the development
                                     tasks, for the no-answers rule on the bank (check.py's `leaks`)
    test_order.json                  {task: [{"cell", "instance", "id"}, ...]}: only if --order is not given

A card is what a harness sees before its first action (the init message of child.py), obtained by
constructing the environment without acting. The test order must be the one shared by every method;
pass it with --order when it exists. The fallback here (a seeded shuffle of the pooled test problems,
cut to the design's count) is only for dry runs until then.

    cd ~/code/multibench && source mb/env.sh
    python -m mb.baselines.metaskill.prep_cluster --out ~/ms_cards [--order shared_order.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from mb.baselines.metaskill.config import DEV_CELLS, SEARCH_N, TEST_PROBLEMS, TEST_TASKS

ORDER_SEED = "exp_b_test_order_v1"


def _no_responder(*a, **k):
    raise RuntimeError("a card is read before the first action; no suspect is asked")


def opaque(cell: str, instance: str) -> str:
    """TextEnv.instance_id (mb/core.py)."""
    bench, sub = cell.split(":", 1)
    return hashlib.md5(f"{bench}/{sub}/{instance}".encode()).hexdigest()[:10]


def card(cell, instance):
    from mb.benches import get
    bench, sub = cell.split(":", 1)
    env = get(bench).make(sub, instance, _no_responder)   # nothing is asked before the first action
    try:
        return {"instructions": env.instructions, "observation": env.observation, "max_steps": env.max_steps,
                "reads_full_reply": env.reads_full_reply, "valid_actions": env.admissible()}
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main():
    from mb.evolve.broker import instances
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--order", type=Path, help="the shared test order {task: [{cell, instance}]}")
    a = ap.parse_args()
    out = a.out.expanduser()
    answers = {}
    for cell in DEV_CELLS:
        search = instances(cell, "search", SEARCH_N)
        answers[cell] = {"search": [str(x) for x in search],
                         "selection": [str(x) for x in instances(cell, "selection", None)]}
        p = out / "cards" / cell.replace(":", "_") / "first_search.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(card(cell, search[0]), indent=1))
        print(cell, "card + answers", flush=True)
    (out / "answers.json").write_text(json.dumps(answers, indent=1))
    if a.order:
        order = json.loads(a.order.read_text())
    else:
        order = {}
        for task, cells in TEST_TASKS.items():
            pool = [(c, str(i)) for c in cells for i in instances(c, "test", None)]
            random.Random(f"{ORDER_SEED}:{task}").shuffle(pool)
            order[task] = [{"cell": c, "instance": i} for c, i in pool[:TEST_PROBLEMS.get(task, 50)]]
        print("WARNING: fallback test order written; use the shared order for the real run", flush=True)
    for task, probs in order.items():
        for p in probs:
            p["id"] = opaque(p["cell"], p["instance"])
            q = out / "cards" / p["cell"].replace(":", "_") / f"{p['id']}.json"
            q.parent.mkdir(parents=True, exist_ok=True)
            if not q.exists():
                q.write_text(json.dumps(card(p["cell"], p["instance"]), indent=1))
        print(task, len(probs), "test cards", flush=True)
    (out / "test_order.json").write_text(json.dumps(order, indent=1))


if __name__ == "__main__":
    main()
