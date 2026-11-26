"""Experiment B, the outer loop on the Mac: proposer in the Docker sandbox, evaluation on bath.

The loop is MAS.S62 HW2's (repo/evolve.py), with the user's own choices there kept:

    1. select_parent()  RRSI + DGM: build only on children that beat their parent by more than the
                        combined noise; among those, sample by score and by how few children each has
    2. propose()        Claude Code (Opus 5.5) edits a copy of the parent in its own workspace, run in
                        the sandbox (ar_harness/sandbox/run_proposer.py), which mounts that workspace only
    3. check()          the child must change something and pass the static rules locally; on bath it
                        must not hold the archive's answers and must survive one smoke episode
    4. evaluate         N search episodes on bath (mb/evolve/eval_cand.sbatch)

What a workspace holds is decided here (`fill_workspace`) and nowhere else: the parent's harness, the
helpers, and for every candidate its code, proposal, public meta, scores and episode traces. Never the
proposer logs, our action-source labels (kept under private/), or anything from the selection and
test splits, which only `final` touches.

    python -m mb.evolve.orchestrate --run pilot_wordle --rounds 8
    python -m mb.evolve.orchestrate --run pilot_wordle --final
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from mb.evolve import remote
from mb.evolve.check import problems

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "ws_template"
SANDBOX = HERE.parents[2] / "sandbox"
RUNS = Path.home() / "evo_b_runs"                 # outside every repo and workspace
CELLS = ["wordle:basic5"]     # set in main(); one harness is scored on all of them (mean over tasks)
MULTI_NOTE = """
## Several tasks

This harness is scored on {k} tasks at once, and its score is the mean over the tasks. For a candidate,
`scores.json` holds the mean and each task's score, and `cells/<task>/` holds that task's `scores.json` and
`episodes/`, laid out as described above. One harness file serves every task.
"""
MODEL = "Qwen3.5-9B"
TEMPERATURE = 0.05            # HW2: score difference that changes a parent's sampling weight by e
CHAT_CAP = 8
PUBLIC_META = ("id", "parent", "valid", "reason")


def sh(cmd, check=True, **kw):
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=True, **kw)


class Archive:
    def __init__(self, run):
        self.dir = RUNS / run / "archive"
        self.dir.mkdir(parents=True, exist_ok=True)

    def all(self):
        return [json.loads(p.read_text()) for p in sorted(self.dir.glob("c*/meta.json"))]

    def save(self, m):
        d = self.dir / m["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(m, indent=1))


def noise(m):
    """Standard error of a candidate's search score, over episodes (one per instance)."""
    s = [e["score"] for e in m["scores"]["per_episode"].values() if e["score"] is not None]
    return statistics.stdev(s) / math.sqrt(len(s)) if len(s) > 1 else 0.0


def select_parent(archive):
    """HW2's select_parent, unchanged in substance (RRSI gate + DGM-weighted draw, seeded by size)."""
    everyone = {m["id"]: m for m in archive.all()}
    scored = {i: (m["scores"]["mean_score"], noise(m)) for i, m in everyone.items() if m["valid"] and m.get("scores")}
    children = Counter(m.get("parent") for m in everyone.values())

    def is_progress(cid):
        p = everyone[cid].get("parent")
        if p is None or p not in scored:
            return True
        return scored[cid][0] - scored[p][0] > math.hypot(scored[cid][1], scored[p][1])

    pool = [c for c in scored if is_progress(c)] or list(scored)
    top = max(scored[c][0] for c in pool)
    w = [math.exp((scored[c][0] - top) / TEMPERATURE) / (1 + children[c]) for c in pool]
    return everyone[random.Random(len(everyone)).choices(pool, weights=w, k=1)[0]]


def summary_md(archive):
    rows = ["| id | parent | valid | mean score | success | reason |", "|---|---|---|---|---|---|"]
    for m in archive.all():
        s = m.get("scores") or {}
        rows.append(f"| {m['id']} | {m.get('parent') or '-'} | {m['valid']} | {s.get('mean_score', '-')} | "
                    f"{s.get('success_rate', '-')} | {m.get('reason') or ''} |")
    return "\n".join(rows) + "\n"


def fill_workspace(ws, archive, parent, n):
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for f in ("harness_lib.py", "mock_env.py"):
        shutil.copy(TEMPLATE / f, ws / f)
    shutil.copy(HERE / "check.py", ws / "check_harness.py")
    (ws / "harness").mkdir()
    shutil.copy(archive.dir / parent["id"] / "harness" / "harness.py", ws / "harness" / "harness.py")
    task = (TEMPLATE / "TASK_template.md").read_text().format(
        model=MODEL, parent=parent["id"], n=n, noise=noise(parent), chat_cap=CHAT_CAP)
    (ws / "TASK.md").write_text(task)
    arc = ws / "archive"
    arc.mkdir()
    (arc / "summary.md").write_text(summary_md(archive))
    for m in archive.all():
        src, dst = archive.dir / m["id"], arc / m["id"]
        dst.mkdir()
        shutil.copytree(src / "harness", dst / "harness")
        if (src / "proposal.md").exists():
            shutil.copy(src / "proposal.md", dst / "proposal.md")
        (dst / "meta.json").write_text(json.dumps({k: m.get(k) for k in PUBLIC_META}, indent=1))
        if (src / "scores.json").exists():
            shutil.copy(src / "scores.json", dst / "scores.json")
        if (src / "episodes").exists():
            shutil.copytree(src / "episodes", dst / "episodes")
        if (src / "cells").exists():
            shutil.copytree(src / "cells", dst / "cells")
    if len(CELLS) > 1:
        (ws / "TASK.md").write_text((ws / "TASK.md").read_text() + MULTI_NOTE.format(k=len(CELLS)))


def proposer_cost(log):
    total = None
    for line in log.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            total = e.get("total_cost_usd")
    return total


def propose(ws, log):
    sys.path.insert(0, str(SANDBOX))
    import run_proposer
    rc = run_proposer.run(ws, log)
    return rc, proposer_cost(log) if log.exists() else None


REMOTE = None                 # set in main(): remote.make("ssh:bath") or remote.make("mailbox:<clone>")


def evaluate(run, cid, harness, split="search", n=20, export=True, smoke=True, poll=30, cells=None):
    """Score one harness on the cluster; returns its summary or {"rejected": [...]}."""
    key = f"{run}/{cid}/{split}" + (f"_{'+'.join(c.replace(':', '_') for c in cells)}" if cells else "")
    params = {"cell": ",".join(cells or CELLS), "split": split, "n": n, "export": export, "smoke": smoke}
    while True:
        REMOTE.submit(key, harness, params)
        print(f"  {key}: submitted", flush=True)
        while (r := REMOTE.result(key)) is None:
            time.sleep(poll)
        if "rejected" in r:
            return r
        if "summary" in r and r["summary"]["infra_errors"] == 0:
            return r["summary"]
        print(f"  {key}: incomplete ({str(r)[:120]}), resubmitting", flush=True)   # ssh backend only
        params["smoke"] = False


def pull_export(run, cid, archive):
    REMOTE.fetch_export(f"{run}/{cid}/search", archive.dir / cid)


def score_candidate(run, archive, m, n):
    cid = m["id"]
    r = evaluate(run, cid, archive.dir / cid / "harness" / "harness.py", n=n)
    priv = RUNS / run / "private" / cid
    priv.mkdir(parents=True, exist_ok=True)
    (priv / "search_summary.json").write_text(json.dumps(r, indent=1))
    if "rejected" in r:
        m.update(valid=False, reason="; ".join(r["rejected"])[:300])
    else:
        pull_export(run, cid, archive)
        d = archive.dir / cid
        if len(CELLS) > 1:                  # mean over tasks; episodes keyed task/instance for the noise
            per = {c.name: json.loads((c / "scores.json").read_text()) for c in sorted((d / "cells").iterdir())}
            comb = {"mean_score": statistics.mean(v["mean_score"] for v in per.values()),
                    "success_rate": statistics.mean((v.get("success_rate") or 0) for v in per.values()),
                    "per_task": {k: v["mean_score"] for k, v in per.items()},
                    "per_episode": {f"{k}/{i}": e for k, v in per.items() for i, e in v["per_episode"].items()}}
            (d / "scores.json").write_text(json.dumps(comb, indent=1))
        m.update(valid=True, scores=json.loads((d / "scores.json").read_text()))
    archive.save(m)
    print(f"  {cid}: {m.get('reason') or m['scores']['mean_score']}", flush=True)


def loop(a):
    archive = Archive(a.run)
    REMOTE.sync_code(HERE)
    if not (archive.dir / "c000" / "meta.json").exists():
        (archive.dir / "c000" / "harness").mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE / "seeds" / f"{a.seed}.py", archive.dir / "c000" / "harness" / "harness.py")
        (archive.dir / "c000" / "proposal.md").write_text("The seed: the benchmark's official default agent.\n")
        print("c000 (seed)", flush=True)
        score_candidate(a.run, archive, {"id": "c000", "parent": None, "valid": False}, a.n)
    while True:
        done = len(archive.all()) - 1
        if done >= a.rounds:
            break
        if stalled(archive, a.patience):
            print(f"early stop: {a.patience} rounds in a row without a gain beyond noise", flush=True)
            break
        cid = f"c{done + 1:03d}"
        parent = select_parent(archive)
        print(f"{cid}: parent {parent['id']} ({parent['scores']['mean_score']})", flush=True)
        ws = RUNS / a.run / "workspaces" / cid
        d = archive.dir / cid
        done_p = RUNS / a.run / "private" / cid / "proposed.json"
        if done_p.exists():          # resume: the proposer already finished for this candidate
            m = json.loads(done_p.read_text())
            parent = next(x for x in archive.all() if x["id"] == m["parent"])
            rc, cost = m["proposer_rc"], m["proposer_cost_usd_equiv"]
            print(f"  {cid}: resuming after the proposer (scoring only)", flush=True)
        else:
            fill_workspace(ws, archive, parent, a.n)
            log = RUNS / a.run / "private" / cid / "proposer.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            rc, cost = propose(ws, log)
            m = {"id": cid, "parent": parent["id"], "valid": False, "proposer_rc": rc,
                 "proposer_cost_usd_equiv": cost, "proposer_seconds": round(time.time() - t0)}
            (d / "harness").mkdir(parents=True, exist_ok=True)
            shutil.copy(ws / "harness" / "harness.py", d / "harness" / "harness.py")
            if (ws / "proposal.md").exists():
                shutil.copy(ws / "proposal.md", d / "proposal.md")
            done_p.write_text(json.dumps(m))
        child = d / "harness" / "harness.py"
        src = child.read_text()
        print(f"  proposer rc={rc} cost≈${cost} in {m['proposer_seconds']}s", flush=True)
        if src == (archive.dir / parent["id"] / "harness" / "harness.py").read_text():
            m["reason"] = "no change"
        elif bad := problems(src):
            m["reason"] = "static check: " + "; ".join(bad)[:300]
        if m.get("reason"):
            archive.save(m)
            print(f"  {cid}: rejected locally: {m['reason']}", flush=True)
            continue
        score_candidate(a.run, archive, m, a.n)


def stalled(archive, patience):
    """True if each of the last `patience` candidates failed to beat the best score before it by more
    than the combined noise (invalid candidates count as no gain)."""
    ms = archive.all()
    if len(ms) <= patience:
        return False
    for i in range(len(ms) - patience, len(ms)):
        before = [m for m in ms[:i] if m["valid"] and m.get("scores")]
        m = ms[i]
        if before and m["valid"] and m.get("scores"):
            best = max(before, key=lambda x: x["scores"]["mean_score"])
            if m["scores"]["mean_score"] - best["scores"]["mean_score"] > math.hypot(noise(m), noise(best)):
                return False
    return True


def final(a):
    """After the loop (settings of notes/exp_b_design.md): the 3 valid candidates with the best search
    score go to the selection split; the best of them is then run on the test tasks. Nothing here goes
    back into any workspace."""
    archive = Archive(a.run)
    priv = RUNS / a.run / "private"
    valid = [m for m in archive.all() if m["valid"] and m.get("scores")]
    top = sorted(valid, key=lambda m: m["scores"]["mean_score"], reverse=True)[:3]
    sel = {}
    for m in top:
        r = evaluate(a.run, m["id"], archive.dir / m["id"] / "harness" / "harness.py",
                     split="selection", n=a.n_selection, export=False, smoke=False)
        sel[m["id"]] = r
        (priv / m["id"]).mkdir(parents=True, exist_ok=True)
        (priv / m["id"] / "selection_summary.json").write_text(json.dumps(r, indent=1))
    best = max(sel, key=lambda c: sel[c]["mean_score"])
    print("selection:", {c: r["mean_score"] for c, r in sel.items()}, "best", best, flush=True)
    (RUNS / a.run / "chosen.json").write_text(json.dumps({"best": best, "selection": {
        c: r["mean_score"] for c, r in sel.items()}}, indent=1))
    for cell in (a.test_cells.split(",") if a.test_cells else []):
        r = evaluate(a.run, best, archive.dir / best / "harness" / "harness.py", split="test",
                     n=a.n_test, export=False, smoke=False, cells=[cell])
        (priv / best / f"test_{cell.replace(':', '_')}.json").write_text(json.dumps(r, indent=1))
        print("test", best, cell, r.get("mean_score", r), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n", type=int, default=20, help="search episodes per task")
    ap.add_argument("--n-selection", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--test-cells", default="", help="comma-separated new tasks the chosen harness is tested on")
    ap.add_argument("--patience", type=int, default=3, help="early stop after this many rounds without a gain")
    ap.add_argument("--seed", default="textarena_standard", help="file in mb/evolve/seeds/ (without .py)")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--remote", default="ssh:bath", help="ssh:<host> or mailbox:<path to clone>")
    ap.add_argument("--cells", default="wordle:basic5", help="comma-separated development tasks")
    a = ap.parse_args()
    global REMOTE, CELLS
    REMOTE, CELLS = remote.make(a.remote), a.cells.split(",")
    final(a) if a.final else loop(a)


if __name__ == "__main__":
    main()
