"""Long-lived episode player on the cluster, for methods that adapt between problems at test time.

The method's own loop runs on the Mac (its LLM is Claude Code on the user's subscription, which stays
there). It asks for one problem at a time through the git mailbox; this worker, a GPU job that keeps one
vLLM server up, plays it at once with mb/evolve/broker.py and pushes the result back. No job has to be
queued per problem.

Mailbox layout (key = "<run>/<task>/<step>"):
    play_requests/<key>/harness.py, request.json      {"cell", "split", "instances": [...], "visible_only"}
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
                            "--instances", ",".join(params["instances"]), "--out", str(out),
                            "--workers", str(min(16, len(params["instances"])))],
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
    priv = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    if (exp / "scores.json").exists():
        priv["per_episode"] = json.loads((exp / "scores.json").read_text()).get("per_episode")
    (res / "private.json").write_text(json.dumps(priv, indent=1))
    log(f"{key}: done ({priv.get('mean_score')})")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailbox", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--until", type=float, default=0)
    ap.add_argument("--parallel", type=int, default=8, help="requests served at once")
    ap.add_argument("--poll", type=int, default=10)
    a = ap.parse_args()
    box, runs = Mailbox(Path(a.mailbox)), Path(a.runs).expanduser()
    stop = Path.home() / "mailbox_STOP"

    def log(msg):
        print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)

    busy = {}
    with ThreadPoolExecutor(a.parallel) as ex:
        while not stop.exists() and not (a.until and time.time() > a.until):
            try:
                box.pull()
                done = [k for k, f in busy.items() if f.done()]
                for k in done:
                    busy.pop(k).result()
                if done:
                    box.push("play: " + ", ".join(done))
                for rj in sorted((box.repo / "play_requests").rglob("request.json")):
                    key = str(rj.parent.relative_to(box.repo / "play_requests"))
                    if key in busy or (box.repo / "play_results" / key / "private.json").exists():
                        continue
                    if len(busy) < a.parallel:
                        busy[key] = ex.submit(serve_one, box, runs, key, log)
                        log(f"{key}: started")
                hb = box.repo / "heartbeat" / "play_worker.json"
                hb.parent.mkdir(exist_ok=True)
                hb.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "busy": sorted(busy)}))
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
