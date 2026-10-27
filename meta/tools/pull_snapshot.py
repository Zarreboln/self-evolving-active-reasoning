"""Pull the three run directories' current state off the cluster into one flat snapshot.

The figures read the snapshot rather than the cluster, so they can be redrawn offline and the
numbers behind a published figure stay with it. Run this, then plot_evolution.py.

    python meta/tools/pull_snapshot.py notes/arbench_mh/evolution_run1_snapshot.json
"""
from __future__ import annotations

import json
import subprocess
import sys

REMOTE = r'''
import json, glob, os, pathlib

# val.json is the scored result the loop itself wrote: accuracy as a fraction, total_tokens, and
# for Situation Puzzles the judge's three tiers. It is the only place a system that has fallen off
# the frontier still has its number, which is exactly the no-questions anchor's situation once the
# search passes it -- reading the frontier instead drops the anchor off the figure.
out = {}
for t in ("gn", "dc", "sp"):
    R = pathlib.Path.home() / ("meta/arbench_mh/runs/%s/logs/run1" % t)
    task = {"iterations": [], "frontier": None, "systems": {}}
    f = R / "evolution_summary.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                task["iterations"].append(json.loads(line))
    fr = R / "frontier_val.json"
    if fr.exists():
        task["frontier"] = json.load(open(fr))
    for q in glob.glob(str(R / "**/val.json"), recursive=True):
        v = json.load(open(q))
        name = pathlib.Path(q).parts[-3]
        task["systems"][name] = {
            "accuracy": 100.0 * v["accuracy"], "tokens": v.get("total_tokens"),
            "calls": v.get("llm_calls"), "n": v.get("n"), "tiers": v.get("tiers"),
        }
    out[t] = task
print(json.dumps(out))
'''

TASK_NAME = {"gn": "Guess Number", "dc": "Detective Cases", "sp": "Situation Puzzles"}
SEEDS = {"baseline_no_questions", "template_zero_shot",
         "baseline_zero_shot", "baseline_few_shot_inst"}


def main(out_path):
    raw = subprocess.run(["ssh", "-o", "BatchMode=yes", "bath", "~/venv/bin/python -I -"],
                         input=REMOTE, capture_output=True, text=True, check=True).stdout
    evo = json.loads(raw)
    rows = []
    for t, v in evo.items():
        pareto = {p["system"] for p in (v["frontier"] or {}).get("_pareto", [])}
        meta = {it["system"]: it for it in v["iterations"]}
        for name, s in sorted(v["systems"].items()):
            m = meta.get(name, {})
            rows.append({
                "task": TASK_NAME[t], "task_id": t,
                "iteration": m.get("iteration", 0), "system": name,
                "kind": "seed" if name in SEEDS else "candidate",
                "accuracy": s["accuracy"], "tokens": s["tokens"],
                "calls": s.get("calls"), "episodes": s.get("n"), "tiers": s.get("tiers"),
                "axis": m.get("axis", ""), "delta": m.get("delta"),
                "frontier": "yes" if name in pareto else "no",
                "hypothesis": (m.get("hypothesis") or "")[:300],
            })
    json.dump(rows, open(out_path, "w"), indent=0)
    for t in ("gn", "dc", "sp"):
        rs = [r for r in rows if r["task_id"] == t and r["accuracy"] is not None]
        best = max(rs, key=lambda r: r["accuracy"])
        print(f"  {t}: {len(rs)} systems, iterations 0-{max(r['iteration'] for r in rs)}, "
              f"best {best['system']} {best['accuracy']:.1f}% @ {best['tokens']:,.0f} tok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "notes/arbench_mh/evolution_run1_snapshot.json")
