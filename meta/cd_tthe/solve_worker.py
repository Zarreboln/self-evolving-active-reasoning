"""Run ONE harness on ONE instance in an isolated process and write a full result record.

    python solve_worker.py --task task.json
task.json: {"harness": "<name in agents/>", "index": <instance index>, "output": "<path>",
            "solver": {"base_url", "model", "cache_dir", "mock", "seed"}, "arm": "scratch|lib"}

The parent enforces the wall-clock limit (process group kill); here we only run the episode,
score it (gold is used for the record only, never shown to the harness), and compute the
label-free signals.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bridge          # noqa: E402
import signals         # noqa: E402
from audit import audit  # noqa: E402


def load_harness(name):
    path = HERE / "agents" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agents_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Harness


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True)
    task = json.loads(Path(ap.parse_args().task).read_text())
    out = {"harness": task["harness"], "index": task["index"], "status": "ok", "error": None}
    t0 = time.time()
    try:
        src = (HERE / "agents" / f"{task['harness']}.py").read_text()
        problems = audit(src, task.get("arm", "scratch"))
        if problems:
            raise RuntimeError("audit: " + "; ".join(problems))
        env = bridge.CDEnv(task["index"], budget=int(task.get("budget", bridge.BUDGET)))
        sv = task.get("solver", {})
        solver = bridge.Solver(base_url=sv.get("base_url"), model=sv.get("model"), cache_dir=sv.get("cache_dir"),
                               mock=sv.get("mock"), mock_game=bridge.game_for(env.inventory), seed=int(sv.get("seed", 0)))
        H = load_harness(task["harness"])
        h = H(env, solver)
        guess_text, crashed = "", False
        try:
            guess_text = h.solve() or ""
        except Exception:
            crashed = True
            out["status"] = "crash"
            out["error"] = traceback.format_exc()[-1500:]
        # What leaves this worker depends on task["reveal"] (label isolation, 2026-10-12):
        #   none       public run status only (the proposer's own check runs in the strict arm)
        #   outcome    + the episode's gold outcome (labeled arm: labels are shown by design)
        #   controller + oracle-derived signals and the full gold record, for the controller's private store
        reveal = task.get("reveal", "none")
        parsed = bool(re.findall(r"[01]{24}", re.sub(r"\s", "", guess_text or "")))
        seen, rep = set(), 0
        for q, _ in env.history:
            rep += q in seen; seen.add(q)
        out["signals"] = {"health": (not crashed) and parsed, "crashed": crashed, "parsed_guess": parsed,
                          "n_queries": len(env.history), "repeat": (rep / len(env.history)) if env.history else None}
        if reveal == "controller":
            out["signals"] = signals.compute(env, guess_text, crashed)
            out["gold"] = env.score(guess_text)
        elif reveal == "outcome":
            g = env.score(guess_text)
            out["gold"] = {k: g[k] for k in ("parsed_guess", "joint_truth_table_correct", "circuit_wise_accuracy")}
        out["history"] = [[list(q), b] for q, b in env.history]
        out["trace"] = h._trace
        out["guess_text"] = (guess_text or "")[-600:]
        out["solver_calls"] = solver.calls
        out["tokens"] = {"in": solver.tokens_in, "out": solver.tokens_out}
    except Exception:
        out["status"] = "harness_error"
        out["error"] = traceback.format_exc()[-1500:]
    out["duration_s"] = round(time.time() - t0, 2)
    Path(task["output"]).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
