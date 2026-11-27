"""The official default agents on experiment B's fixed test problems (~/expb_inputs/order.json), under the
current episode rules (8192-token replies, runaway = 0, wall-clock timeout = rerun). Decided 2026-11-27
(user): results measured under the old rules are not reused.

    python -m mb.evolve.official_test --remote mailbox:$HOME/evo_b_mailbox            # submit and wait
writes ~/evo_b_runs/official_test/results.json.

ALFWorld is not rerun: its official ReAct agent (mb/official/run_react_alfworld.py) uses the notebook's
max_tokens 100, so the reply cap never binds, and its run on the same 24 problems had no failures
(0.5833, mb/official results); it is reused as is.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from mb.evolve import remote

SEEDS = Path(__file__).parent / "seeds"
RUNS = Path.home() / "evo_b_runs" / "official_test"
# test task -> (official agent seed, cells, n per cell)
PLAN = {"mastermind_basic": ("textarena_standard", ["mastermind:basic"], 50),
        "minesweeper_basic": ("textarena_standard", ["minesweeper:basic"], 50),
        "wordle_hard5": ("textarena_standard", ["wordle:hard5"], 50),
        "hangman_hard": ("textarena_standard", ["hangman:hard"], 50),
        "twentyq_basic_pooled": ("textarena_standard",
                                 ["twentyq:basic-people", "twentyq:basic-places", "twentyq:basic-things"], 17),
        "sciworld_grow_plant": ("swiftsage_react", ["sciworld:grow-plant"], 33)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    a = ap.parse_args()
    box = remote.make(a.remote)
    order = json.loads((Path.home() / "expb_inputs" / "order.json").read_text())
    keys = {}
    for task, (seed, cells, n) in PLAN.items():
        for c in cells:
            key = f"official_test/{task}/test_{c.replace(':', '_')}"
            box.submit(key, SEEDS / f"{seed}.py", {"cell": c, "split": "test", "n": n, "export": True,
                                                   "smoke": False})
            keys[key] = (task, c)
            print("submitted", key, flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    while True:
        res = {k: box.result(k) for k in keys}
        if all(r is not None for r in res.values()):
            break
        time.sleep(60)
    # exports hash episode ids: match the raw records on the cluster by real instance id
    code = (Path(__file__).parent / "official_score_remote.py").read_text()
    remote_cmd = "cat > /tmp/official_score_remote.py <<'PYEOF'\n" + code + "\nPYEOF\npython3 /tmp/official_score_remote.py"
    out = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "engaging", remote_cmd], input=json.dumps(order),
                                    capture_output=True, text=True, check=True).stdout)
    for task in out:
        out[task]["agent"] = PLAN[task][0]
        print(task, out[task], flush=True)
    out["alfworld_pick_and_place_simple"] = {"agent": "ReAct (reused, see docstring)", "episodes": 24,
                                             "score": 0.5833}
    (RUNS / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
