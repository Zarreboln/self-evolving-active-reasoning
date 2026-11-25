"""Difficulty screen: the plain react seed, zero-shot, on N search instances of every subtask.

Resumable, because it runs as lowprio and a preempted job is requeued under the same id: every
episode writes its own file (atomically, via rename), and a rerun skips episodes whose file holds
a clean result (one that errored, e.g. on a dead server, runs again). `--merge` writes the
per-subtask table once everything is in.

Every episode runs in its own subprocess with a timeout (EPISODE_TIMEOUT), launched from a thread
pool. The first version used a ProcessPoolExecutor with max_tasks_per_child; on bath it lost its
workers and hung with both GPUs idle for two hours (job 11361). A subprocess per episode cannot
take the run down with it: a hang or crash is killed, recorded as an error, and retried on the
next resume. It also keeps ScienceWorld's JVMs out of a shared process. Episodes are spread
round-robin over the servers in POLICY_BASE_URLS; every LLM-played role uses the same model.

    python -m mb.screen --out DIR [--n 20] [--workers 64]      # run / resume
    python -m mb.screen --out DIR --merge                       # summary.json + summary.tsv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mb.benches import BENCHES, get

EPISODE_TIMEOUT = 2400      # s; the longest clean episodes (ScienceWorld at 250 steps) take ~20 min


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _path(out, bench, sub, inst):
    return os.path.join(out, bench, _slug(sub), _slug(inst) + ".json")


def _episode(args):
    out, bench, sub, inst, url = args
    from mb.harness.react import run_episode
    from mb.llm import Counter, make_chat
    model = os.environ["POLICY_MODEL"]
    b = get(bench)
    pc, rc = Counter(), Counter()
    t0 = time.time()
    jev = None
    try:
        responder = None
        if b.responder:
            # Jev plays SP / 20Q / Guess Who unless MB_USE_JEV=0, in which case every LLM-played
            # role is the run's chat model. Never a silent mix: a Jev failure is an error, not a
            # fallback.
            if os.environ.get("MB_USE_JEV", "1") == "1":
                from mb.jev import Jev, responder_for
                jev = Jev()
                responder = responder_for(bench, jev)
            responder = responder or make_chat(url, model, rc)
        env = b.make(sub, inst, responder)
        r = run_episode(env, make_chat(url, model, pc))
    except Exception as e:
        r = {"bench": bench, "subtask": sub, "instance": inst, "score": float("nan"),
             "success": None, "error": "setup/run: " + repr(e)}
    r.update(policy_tokens=pc.as_dict(), responder_tokens=rc.as_dict(), seconds=round(time.time() - t0, 1))
    if jev is not None and jev.calls:
        r["jev"] = jev.usage()
    _write(out, bench, sub, inst, r)
    return bench, sub, r.get("score"), r.get("error")


def _write(out, bench, sub, inst, r):
    p = _path(out, bench, sub, inst)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(r, open(tmp, "w"))
    os.replace(tmp, p)


def _episode_subprocess(job):
    """Run one episode in a fresh interpreter; a timeout or crash becomes an error record."""
    out, bench, sub, inst, url = job
    cmd = [sys.executable, "-m", "mb.screen", "--one", json.dumps(job)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=EPISODE_TIMEOUT)
        if cp.returncode == 0:
            r = json.load(open(_path(out, bench, sub, inst)))
            return bench, sub, r.get("score"), r.get("error")
        err = f"exit {cp.returncode}: {cp.stderr.strip()[-300:]}"
    except subprocess.TimeoutExpired:
        err = f"timeout after {EPISODE_TIMEOUT}s"
    _write(out, bench, sub, inst, {"bench": bench, "subtask": sub, "instance": inst,
                                   "score": float("nan"), "success": None, "error": err})
    return bench, sub, None, err


def run(a):
    urls = [u.strip().rstrip("/") for u in os.environ["POLICY_BASE_URLS"].split(",") if u.strip()]
    jobs, done = [], 0
    for bench in a.benches.split(","):
        b = get(bench)
        for sub in b.subtasks:
            for inst in b.instances(sub, "search", limit=a.n):
                p = _path(a.out, bench, sub, inst)
                if os.path.exists(p) and not json.load(open(p)).get("error"):
                    done += 1          # finished cleanly; an episode that errored is run again
                    continue
                jobs.append((a.out, bench, sub, inst, urls[len(jobs) % len(urls)]))
    print(f"{done} episodes already on disk, {len(jobs)} to run, {len(urls)} servers", flush=True)
    # interleave benches so slow ones (ScienceWorld, BoxingGym) do not all land at the end
    jobs.sort(key=lambda j: hash((j[3], j[2])) % 1000)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_episode_subprocess, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            bench, sub, score, err = f.result()
            if err or k % 50 == 0:
                print(f"[{k}/{len(jobs)} {time.time() - t0:.0f}s] {bench}/{sub} score={score} {err or ''}"[:220], flush=True)
    print(f"run done in {time.time() - t0:.0f}s", flush=True)


def merge(a):
    rows = []
    for bench in a.benches.split(","):
        b = get(bench)
        for sub in b.subtasks:
            eps = []
            for inst in b.instances(sub, "search", limit=a.n):
                p = _path(a.out, bench, sub, inst)
                if os.path.exists(p):
                    eps.append(json.load(open(p)))
            ok = [e for e in eps if not e.get("error")]          # errored episodes are not scores
            sc = [e["score"] for e in ok if isinstance(e.get("score"), (int, float)) and not math.isnan(e["score"])]
            wins = [e["success"] for e in ok if e.get("success") is not None]
            rows.append({
                "bench": bench, "subtask": sub, "episodes": len(eps),
                "errors": sum(bool(e.get("error")) for e in eps),
                "mean_score": round(sum(sc) / len(sc), 4) if sc else None,
                "success_rate": round(sum(map(bool, wins)) / len(wins), 4) if wins else None,
                "higher_is_better": eps[0].get("higher_is_better", True) if eps else None,
                "mean_steps": round(sum(e.get("steps", 0) for e in eps) / len(eps), 1) if eps else None,
                "policy_out_tok": sum(e.get("policy_tokens", {}).get("out_tok", 0) for e in eps),
                "seconds": round(sum(e.get("seconds", 0) for e in eps), 1),
            })
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    keys = list(rows[0])
    with open(os.path.join(a.out, "summary.tsv"), "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"{len(rows)} subtasks merged -> {a.out}/summary.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)   # internal: one episode, as JSON
    a = ap.parse_args()
    if a.one:
        _episode(tuple(json.loads(a.one)))
    elif not a.out:
        ap.error("--out is required")
    else:
        merge(a) if a.merge else run(a)


if __name__ == "__main__":
    main()
