"""Take a task's own evolution in a soft_tree3 run from a finished Meta-Harness run (user, 2026-11-27).

A task that starts from the base agent goes through exactly Meta-Harness's inner loop (same start, same TASK
template and archive, same search problems), so its evolution is Meta-Harness's own run: the first task of
every version, and every task of "flat". This copies the Meta-Harness archive (c000..cNNN) as the task's
candidates, makes the leaf the candidate with the highest search score (the earliest of ties; the rule fixed
before looking), sets the start to Meta-Harness's c000 (the base agent on the same 20 problems), and writes the
edit records, so the run continues after this task's edits. The task's own probe, description and required
capabilities stay as they are (the task must already have joined). Opus time and tokens of the copied
candidates stay in the Meta-Harness run's ledger; the records name the source.

    python -m mb.evolve.import_mh_task --run ours3 --task dc:all --mh mh_pbta_dc
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

RUNS = Path.home() / "evo_b_runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--mh", required=True, help="the Meta-Harness run, e.g. mh_pbta_dc")
    ap.add_argument("--k", type=int, default=0, help="the task's position (1-based); default: its index in the graph")
    a = ap.parse_args()
    run, src = RUNS / a.run, RUNS / a.mh / "archive"
    slug = a.task.replace(":", "_")
    v = sorted(p for p in (run / "state").iterdir() if (p / "graph.json").exists())[-1]
    g, meta = json.loads((v / "graph.json").read_text()), json.loads((v / "meta.json").read_text())
    assert a.task in g["order"], "the task must have joined (probe, description) before its edits are imported"
    k = a.k or g["order"].index(a.task) + 1
    cands = sorted(d.name for d in src.iterdir() if (d / "meta.json").exists())
    metas = {c: json.loads((src / c / "meta.json").read_text()) for c in cands}
    scored = [(m["scores"]["mean_score"], c) for c, m in metas.items() if m.get("valid") and m.get("scores")]
    best = max(scored, key=lambda t: (t[0], -int(t[1][1:])))[1]

    arc = run / "archive" / slug
    shutil.rmtree(arc, ignore_errors=True)
    for c in cands:
        d, m = arc / c, metas[c]
        shutil.copytree(src / c / "harness", d / "harness")
        for f in ("proposal.md", "scores.json"):
            if (src / c / f).exists():
                shutil.copy(src / c / f, d / f)
        if (src / c / "episodes").exists():
            shutil.copytree(src / c / "episodes", d / "episodes")
        pub = {"id": c, "parent": m.get("parent"), "valid": m.get("valid"), "reason": m.get("reason")}
        (d / "meta.json").write_text(json.dumps(pub, indent=1))
        (d / "full.json").write_text(json.dumps({**pub, "scores": m.get("scores"), "source": f"{a.mh}/{c}"}, indent=1))
    for c in (cands[0], best):                       # exports the run reads as evidence
        e = run / "exports" / f"t{k}_mh_{c}"
        shutil.rmtree(e, ignore_errors=True)
        e.mkdir(parents=True)
        shutil.copy(src / c / "scores.json", e / "scores.json")
        if (src / c / "episodes").exists():
            shutil.copytree(src / c / "episodes", e / "episodes")

    g["leaves"][a.task] = {"code": (src / best / "harness" / "harness.py").read_text(), "doc": ""}
    s0, sb = metas[cands[0]]["scores"], metas[best]["scores"]
    meta["start"][a.task] = s0["mean_score"]
    meta["scores"][a.task] = sb["mean_score"]
    meta["per_episode"][a.task] = sb["per_episode"]
    meta["last_export"][a.task] = f"t{k}_mh_{best}"
    meta.setdefault("leaf_cand", {})[a.task] = best
    d = run / "state" / f"v{len(list((run / 'state').iterdir())):03d}"
    d.mkdir()
    (d / "graph.json").write_text(json.dumps(g, indent=1))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rounds = run / "rounds"
    rec = json.loads((rounds / f"t{k}_start.json").read_text()) if (rounds / f"t{k}_start.json").exists() else {}
    rec.update(start_score=s0["mean_score"], imported_from=f"{a.mh}/{cands[0]}",
               outcome=f"joined; start {s0['mean_score']} (the base agent: {a.mh}/{cands[0]})", finished_at=now)
    (rounds / f"t{k}_start.json").write_text(json.dumps(rec, indent=1))
    for e, c in enumerate(cands[1:], 1):
        m = metas[c]
        (rounds / f"t{k}_e{e}.json").write_text(json.dumps({
            "task": a.task, "kind": "edit", "candidate": c, "parent": m.get("parent"),
            "imported_from": f"{a.mh}/{c}", "search_score": (m.get("scores") or {}).get("mean_score"),
            "outcome": ("kept" if c == best else "imported") + f" ({a.mh}/{c} {(m.get('scores') or {}).get('mean_score')})",
            "finished_at": now}, indent=1))
    print(f"{a.run}: {a.task} <- {a.mh} {cands}; leaf {best} ({sb['mean_score']}); start {s0['mean_score']}; state {d.name}")


if __name__ == "__main__":
    main()
