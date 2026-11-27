"""Long-lived episode player on the cluster, for methods that adapt between problems at test time.

The method's own loop runs on the Mac (its LLM is Claude Code on the user's subscription, which stays
there). It asks for one problem at a time through the git mailbox; this worker, a GPU job that keeps one
vLLM server up, plays it at once with mb/evolve/broker.py and pushes the result back. No job has to be
queued per problem.

Mailbox layout (key = "<run>/<task>/<step>"):
    play_requests/<key>/harness.py, request.json      {"cell", "split", "instances": [...] (or "n": the
                                                       first n of the split), "visible_only"}
    play_results/<key>/visible/...                     what the method may read
    play_results/<key>/private.json                    scores, for our reports only (the method's
                                                       play() never reads it)
With visible_only (test time), visible/ holds only each episode's trace.md: what the model saw and did,
and what the environment said. Without it (before deployment), it holds the broker's normal export
(scores.json and, per episode, trace.md and result.json).

    python -m mb.evolve.worker --mailbox ~/mailbox_worker --runs ~/results/mb/play --until UNIX_TIME
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mb.evolve.remote import Mailbox


ME = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"


def alive(owner: str) -> bool:
    if owner.startswith("pid"):
        try:
            os.kill(int(owner[3:]), 0)
            return True
        except Exception:
            return False
    r = subprocess.run(["squeue", "-h", "-j", owner], capture_output=True, text=True)
    if r.returncode != 0:          # slurm controller unreachable: assume alive rather than steal the request
        return True
    return bool(r.stdout.strip())


def claim(work: Path) -> bool:
    """Atomic claim of one request on the shared filesystem (work/claim/job), so that several workers
    (and the eval jobs of race.sh) never play the same request at once. A claim whose owner is no longer
    queued or running is taken over; the broker then resumes from the episodes already on disk."""
    work.mkdir(parents=True, exist_ok=True)
    d = work / "claim"
    try:
        d.mkdir()
    except FileExistsError:
        owner = (d / "job").read_text().strip() if (d / "job").exists() else ""
        if owner and owner != ME and alive(owner):
            return False
    (d / "job").write_text(ME)
    return True


def _broker_cell(cand, cell, split, n, sub, dst, smoke, workers):
    subprocess.run([sys.executable, "-m", "mb.evolve.broker", "run", "--cand", str(cand), "--cell", cell,
                    "--split", split, "--n", str(n), "--out", str(sub), "--workers", str(workers)]
                   + (["--smoke"] if smoke else []), capture_output=True, text=True)
    if dst:
        subprocess.run([sys.executable, "-m", "mb.evolve.broker", "export", "--out", str(sub), "--dest", str(dst)],
                       capture_output=True, text=True)


def _combine(out: Path):
    """eval_cand.sbatch's combined OUT/summary.json (or rejected.json) over the per-task sub-runs."""
    import statistics
    cells = sorted(p.name for p in (out / "cells").iterdir())
    rej, summ = [], {}
    for c in cells:
        d = out / "cells" / c
        if (d / "rejected.json").exists():
            rej += [f"{c}: {r}" for r in json.loads((d / "rejected.json").read_text())["rejected"]]
        elif (d / "summary.json").exists():
            summ[c] = json.loads((d / "summary.json").read_text())
    if rej:
        (out / "rejected.json").write_text(json.dumps({"rejected": rej}, indent=1))
    elif len(summ) == len(cells):
        (out / "summary.json").write_text(json.dumps(
            {"cells": summ, "infra_errors": sum(x["infra_errors"] for x in summ.values()),
             "mean_score": statistics.mean(x["mean_score"] for x in summ.values())}, indent=1))


def serve_eval(box: Mailbox, eval_runs: Path, key: str, log):
    """One evaluation request (requests/<key>, the format eval_cand.sbatch serves) on this worker's
    server: same broker calls, same files, published to results/<key>/ as the driver would."""
    req = box.repo / "requests" / key
    params = json.loads((req / "request.json").read_text())
    work = eval_runs / key
    out, dest = work / "out", work / "export"
    stamp = work / "submitted"
    if not stamp.exists() or stamp.read_text() != str(params["submitted"]):   # a new request starts clean
        for d in (out, dest, work / "harness"):
            shutil.rmtree(d, ignore_errors=True)
        stamp.write_text(str(params["submitted"]))
    (work / "harness").mkdir(parents=True, exist_ok=True)
    cells = params["cell"].split(",")
    if params.get("per_cell"):
        for h in req.glob("harness_*.py"):
            d = work / "harness" / h.stem[len("harness_"):]
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(h, d / "harness.py")
    else:
        shutil.copy(req / "harness.py", work / "harness" / "harness.py")
    workers = max(8, 48 // len(cells))
    for attempt in range(4):
        jobs = []
        for c in cells:
            slug = c.replace(":", "_")
            sub = out if len(cells) == 1 else out / "cells" / slug
            dst = (dest if len(cells) == 1 else dest / "cells" / slug) if params.get("export") else None
            cand = work / "harness" / slug if params.get("per_cell") else work / "harness"
            jobs.append((cand, c, params["split"], params["n"], sub, dst, params.get("smoke") and attempt == 0,
                         workers))
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(lambda j: _broker_cell(*j), jobs))
        if len(cells) > 1:
            _combine(out)
        summ = out / "summary.json"
        if (out / "rejected.json").exists() or (summ.exists() and json.loads(summ.read_text())["infra_errors"] == 0):
            break
    with box.lock():
        res = box.repo / "results" / key
        shutil.rmtree(res, ignore_errors=True)
        res.mkdir(parents=True)
        if (out / "rejected.json").exists():
            shutil.copy(out / "rejected.json", res / "rejected.json")
        elif (out / "summary.json").exists() and json.loads((out / "summary.json").read_text())["infra_errors"] == 0:
            shutil.copy(out / "summary.json", res / "summary.json")
            if params.get("export") and dest.exists():
                shutil.copytree(dest, res / "export", dirs_exist_ok=True)
        else:
            (res / "failed.json").write_text(json.dumps({"worker": ME, "tries": 4}))
    log(f"eval {key}: published")
    return "eval:" + key


def assist_eval(box: Mailbox, eval_runs: Path, key: str, log):
    """Help another worker's evaluation: play its still unclaimed episodes on this worker's server
    (broker run --assist). The owner waits for them, summarizes and publishes; scores do not depend on
    which worker played an episode."""
    params = json.loads((box.repo / "requests" / key / "request.json").read_text())
    work = eval_runs / key
    stamp = work / "submitted"
    if not stamp.exists() or stamp.read_text() != str(params["submitted"]):
        return "assist:" + key                     # the owner has not set this request up yet
    cells = params["cell"].split(",")
    jobs = []
    for c in cells:
        slug = c.replace(":", "_")
        sub = work / "out" if len(cells) == 1 else work / "out" / "cells" / slug
        cand = work / "harness" / slug if params.get("per_cell") else work / "harness"
        if not (cand / "harness.py").exists() or not (sub / ".sharded").exists():
            continue
        jobs.append((cand, c, sub))
    def one(j):
        cand, c, sub = j
        subprocess.run([sys.executable, "-m", "mb.evolve.broker", "run", "--assist", "--cand", str(cand),
                        "--cell", c, "--split", params["split"], "--n", str(params["n"]), "--out", str(sub),
                        "--workers", "8"], capture_output=True, text=True)
    if jobs:
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(one, jobs))
        log(f"assist {key}: done")
    return "assist:" + key


def serve_one(box: Mailbox, runs: Path, key: str, log):
    req = box.repo / "play_requests" / key
    params = json.loads((req / "request.json").read_text())
    work = runs / key
    (work / "harness").mkdir(parents=True, exist_ok=True)
    shutil.copy(req / "harness.py", work / "harness" / "harness.py")
    out = work / "out"
    for attempt in range(4):                       # reruns only episodes marked `infra`
        subprocess.run([sys.executable, "-m", "mb.evolve.broker", "run", "--cand", str(work / "harness"),
                            "--cell", params["cell"], "--split", params["split"],
                            *(["--instances", ",".join(params["instances"])] if params.get("instances")
                              else ["--n", str(params["n"])]), "--out", str(out),
                            "--workers", str(min(16, len(params.get("instances") or []) or int(params.get("n", 16))))],
                           capture_output=True, text=True)
        summ = out / "summary.json"
        if (out / "rejected.json").exists() or (summ.exists() and json.loads(summ.read_text())["infra_errors"] == 0):
            break
    with box.lock():                               # git add -A must not see a half-written result
        return _publish(box, work, out, params, key, log)


def _publish(box, work, out, params, key, log):
    res = box.repo / "play_results" / key
    shutil.rmtree(res, ignore_errors=True)
    (res / "visible").mkdir(parents=True)
    if (out / "rejected.json").exists():
        shutil.copy(out / "rejected.json", res / "visible" / "rejected.json")
        (res / "private.json").write_text((out / "rejected.json").read_text())
        return key
    exp = work / "export"
    subprocess.run([sys.executable, "-m", "mb.evolve.broker", "export", "--out", str(out), "--dest", str(exp)],
                   capture_output=True, text=True)
    if params.get("visible_only"):
        for trace in exp.glob("episodes/*/trace.md"):
            dst = res / "visible" / "episodes" / trace.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            lines = trace.read_text().splitlines()
            # the header's "score .. success .." line is ours, not the environment's: drop it
            lines = [l for k, l in enumerate(lines) if not (k < 4 and l.startswith("score "))]
            (dst / "trace.md").write_text("\n".join(lines) + "\n")
    else:
        shutil.copytree(exp, res / "visible", dirs_exist_ok=True)
    if params.get("raw"):                          # full episode records (calls, steps_log, score)
        from mb.screen import _path
        bench, sub = params["cell"].split(":", 1)
        (res / "raw").mkdir()
        for inst in params["instances"]:
            p = Path(_path(str(out), bench, sub, inst))
            if p.exists():
                shutil.copy(p, res / "raw" / p.name)
    priv = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    if (exp / "scores.json").exists():
        priv["per_episode"] = json.loads((exp / "scores.json").read_text()).get("per_episode")
    (res / "private.json").write_text(json.dumps(priv, indent=1))
    log(f"{key}: done ({priv.get('mean_score')})")
    return "play:" + key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailbox", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--until", type=float, default=0)
    ap.add_argument("--parallel", type=int, default=8, help="requests served at once")
    ap.add_argument("--poll", type=int, default=10)
    ap.add_argument("--eval-runs", default="", help="also serve evaluation requests (requests/), working here")
    ap.add_argument("--name", default="play_worker", help="heartbeat file name")
    ap.add_argument("--drain", type=int, default=5400, help="stop taking new requests this many seconds before "
                    "--until, so long episodes end before the job's time limit (the next worker takes over)")
    ap.add_argument("--only", default="", help="serve only requests whose key starts with one of these "
                    "comma-separated prefixes (a dedicated worker); the other workers skip them")
    a = ap.parse_args()
    box, runs = Mailbox(Path(a.mailbox)), Path(a.runs).expanduser()
    eval_runs = Path(a.eval_runs).expanduser() if a.eval_runs else None
    stop = Path.home() / "mailbox_STOP"

    def log(msg):
        print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)

    busy = {}
    assisted = {}                                         # key -> when this worker last helped it
    last_hb = 0.0
    with ThreadPoolExecutor(a.parallel) as ex:
        while not stop.exists() and not (a.until and time.time() > a.until):
            try:
                box.pull()
                done = [k for k, f in busy.items() if f.done()]
                for k in done:
                    try:
                        busy.pop(k).result()
                    except Exception as e:
                        log(f"{k}: error {e!r}")
                if [k for k in done if not k.startswith("assist:")]:
                    box.push("play: " + ", ".join(done))
                todo = []                                     # (submitted, kind, key), oldest first
                for rj in (box.repo / "play_requests").rglob("request.json"):
                    key = str(rj.parent.relative_to(box.repo / "play_requests"))
                    if ("play:" + key) in busy or (box.repo / "play_results" / key / "private.json").exists():
                        continue
                    todo.append((json.loads(rj.read_text()).get("submitted", 0), "play", key))
                if eval_runs:
                    for rj in (box.repo / "requests").rglob("request.json"):
                        key = str(rj.parent.relative_to(box.repo / "requests"))
                        res = box.repo / "results" / key
                        if ("eval:" + key) in busy or any((res / f).exists() for f in
                                                          ("summary.json", "rejected.json", "failed.json")):
                            continue
                        todo.append((json.loads(rj.read_text()).get("submitted", 0), "eval", key))
                only = [x for x in a.only.split(",") if x]
                reserved = []
                rf = Path.home() / "results" / "mb" / "reserved_prefixes.txt"   # prefixes a dedicated worker owns
                if rf.exists():
                    reserved = [x.strip() for x in rf.read_text().split() if x.strip()]
                if only:
                    todo = [t for t in todo if any(t[2].startswith(x) for x in only)]
                else:
                    todo = [t for t in todo if not any(t[2].startswith(x) for x in reserved)]
                draining = a.until and time.time() > a.until - a.drain
                for _, kind, key in ([] if draining else sorted(todo)):
                    if len(busy) >= a.parallel:
                        break
                    if not claim((runs if kind == "play" else eval_runs) / key):
                        continue
                    if kind == "play":
                        busy["play:" + key] = ex.submit(serve_one, box, runs, key, log)
                    else:
                        busy["eval:" + key] = ex.submit(serve_eval, box, eval_runs, key, log)
                    log(f"{kind} {key}: started")
                # idle capacity helps other workers' evaluations still running (user, 2026-11-27)
                if eval_runs and not draining and len(busy) < a.parallel - 2:   # keep 2 slots for own requests
                    for rj in sorted((box.repo / "requests").rglob("request.json")):
                        if len(busy) >= a.parallel - 2:
                            break
                        key = str(rj.parent.relative_to(box.repo / "requests"))
                        res = box.repo / "results" / key
                        if (("eval:" + key) in busy or ("assist:" + key) in busy
                                or time.time() - assisted.get(key, 0) < 300
                                or any((res / f).exists() for f in ("summary.json", "rejected.json", "failed.json"))
                                or (only and not any(key.startswith(x) for x in only))
                                or (not only and any(key.startswith(x) for x in reserved))):
                            continue
                        cl = eval_runs / key / "claim" / "job"
                        owner = cl.read_text().strip() if cl.exists() else ""
                        if not owner or owner == ME or not alive(owner):
                            continue
                        assisted[key] = time.time()
                        busy["assist:" + key] = ex.submit(assist_eval, box, eval_runs, key, log)
                hb = box.repo / "heartbeat" / f"{a.name}.json"
                hb.parent.mkdir(exist_ok=True)
                hb.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "busy": sorted(busy)}))
                if time.time() - last_hb > 300:               # commit it, or the next pull is blocked
                    box.push(f"{a.name}: heartbeat")
                    last_hb = time.time()
            except Exception as e:                            # git hiccup: try again next loop
                log(f"loop error: {e!r}")
            time.sleep(a.poll)
        for f in busy.values():
            f.result()
        if busy:
            box.push("play: final")
    log("worker exiting")


if __name__ == "__main__":
    main()
