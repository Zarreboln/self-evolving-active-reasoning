"""Cluster side of the git mailbox (see mb/evolve/remote.py): runs as a CPU job, no login needed.

Every loop: pull the mailbox; for each request without a final result, run eval_cand.sbatch (at most
--max-jobs at once); when a job ends, publish its summary (and, for search, the export) to results/,
or resubmit it if episodes are missing (lowprio / preemption); write a heartbeat; push.

The driver keeps its own state outside the mailbox, so a restarted driver picks up the jobs that are
still queued. driver.sbatch resubmits the driver before its time limit runs out.

    python -m mb.evolve.driver --mailbox ~/mailbox --cluster orcd --runs ~/results/mb/evo_b
    touch ~/mailbox_STOP      # stops the driver chain after the current loop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from mb.evolve.remote import PLACES, SBATCH_ARGS, Mailbox, env_for

HERE = Path(__file__).resolve().parent
MAX_TRIES = 6


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=True)


def running(jid):
    """jid may be a comma-separated group of copies (race.sh): running while any copy is queued or running."""
    return any(sh(f"squeue -h -j {j}", check=False).stdout.strip() for j in str(jid).split(",") if j)


def step(box: Mailbox, runs: Path, cluster: str, max_jobs: int, state: dict, log, drain=False):
    box.pull()
    def asked(d):                                   # first come, first served (not by name: "ours"
        try:                                        # would wait behind every "abl_*"/"mh_*" request)
            return json.loads((d / "request.json").read_text()).get("submitted", 0)
        except Exception:
            return 0
    reqs = sorted((p.parent for p in (box.repo / "requests").rglob("request.json")), key=asked)
    places = PLACES[cluster]
    used = [sum(1 for s in state.values() if s.get("job") and s.get("place") == i and running(s["job"]))
            for i in range(len(places))]
    in_flight = sum(used)
    changed = []
    for req in reqs:
        key = str(req.relative_to(box.repo / "requests"))
        params = json.loads((req / "request.json").read_text())
        res = box.repo / "results" / key
        if any((res / f).exists() for f in ("summary.json", "rejected.json", "failed.json")):
            continue
        s = state.get(key)
        if s and s["submitted"] != params["submitted"]:        # the Mac asked again: start over
            s = None
        work = runs / key
        out = work / "out"
        if s and s.get("job") and running(s["job"]):
            continue
        if s and s.get("job"):                                  # the job has ended
            rej, summ = out / "rejected.json", out / "summary.json"
            if rej.exists() or (summ.exists() and json.loads(summ.read_text())["infra_errors"] == 0):
                res.mkdir(parents=True, exist_ok=True)
                shutil.copy(rej if rej.exists() else summ, res / (rej if rej.exists() else summ).name)
                if params.get("export") and (work / "export").exists():
                    shutil.copytree(work / "export", res / "export", dirs_exist_ok=True)
                changed.append(key)
                log(f"{key}: done (job {s['job']})")
                state.pop(key)
                continue
            if s["tries"] >= MAX_TRIES:
                res.mkdir(parents=True, exist_ok=True)
                (res / "failed.json").write_text(json.dumps({"tries": s["tries"], "last_job": s["job"]}))
                changed.append(key)
                log(f"{key}: gave up after {s['tries']} tries")
                state.pop(key)
                continue
        if drain:                                     # long-lived workers serve new requests
            state.pop(key, None)
            continue
        free = [i for i, (_, cap) in enumerate(places) if used[i] < cap]
        if in_flight >= max_jobs or not free:
            continue
        place = free[0]
        if not s:
            work.mkdir(parents=True, exist_ok=True)
            (work / "harness").mkdir(exist_ok=True)
            if params.get("per_cell"):
                for h in req.glob("harness_*.py"):
                    d = work / "harness" / h.stem[len("harness_"):]
                    d.mkdir(parents=True, exist_ok=True)
                    shutil.copy(h, d / "harness.py")
            else:
                shutil.copy(req / "harness.py", work / "harness" / "harness.py")
            for d in (out, work / "export"):                   # a fresh request starts clean
                shutil.rmtree(d, ignore_errors=True)
            s = {"submitted": params["submitted"], "tries": 0}
        p = dict(params, smoke=params.get("smoke") and s["tries"] == 0)
        env = " ".join(f"{k}={v}" for k, v in env_for(p, work / "harness", out, work / "export").items())
        shutil.rmtree(work / "claim", ignore_errors=True)      # a new group of copies races afresh
        if cluster == "orcd":                                    # one copy per GPU type (race.sh)
            name = "mbe_" + hashlib.sha1(key.encode()).hexdigest()[:8]
            jid = sh(f"cd ~/logs && export {env} && source {HERE}/race.sh && "
                     f"race_submit {name} '{places[place][0]} --export=ALL' {HERE}/eval_cand.sbatch",
                     check=False).stdout.strip()
        else:
            jid = sh(f"cd ~/logs && {env} sbatch --parsable {places[place][0]} "
                     f"{HERE}/eval_cand.sbatch").stdout.strip()
        if not jid:
            log(f"{key}: submission failed")
            continue
        s.update(job=jid, tries=s["tries"] + 1, place=place)
        state[key] = s
        in_flight += 1
        used[place] += 1
        log(f"{key}: submitted job {jid} (try {s['tries']})")
    hb = box.repo / "heartbeat" / f"{cluster}.json"
    hb.parent.mkdir(exist_ok=True)
    hb.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "in_flight": in_flight,
                              "jobs": {k: v.get("job") for k, v in state.items()}}, indent=1))
    box.push(f"{cluster}: " + (", ".join(changed) if changed else "heartbeat"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailbox", required=True)
    ap.add_argument("--cluster", required=True, choices=sorted(SBATCH_ARGS))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--max-jobs", type=int, default=6)
    ap.add_argument("--drain", action="store_true", help="publish the jobs already submitted, submit no new ones")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--until", type=float, default=0, help="unix time to exit at (0 = never)")
    a = ap.parse_args()
    box, runs = Mailbox(Path(a.mailbox)), Path(a.runs).expanduser()
    runs.mkdir(parents=True, exist_ok=True)
    state_file = runs / f"driver_state_{a.cluster}.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    stop = Path.home() / "mailbox_STOP"

    def log(msg):
        print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)

    while not stop.exists() and not (a.until and time.time() > a.until):
        try:
            step(box, runs, a.cluster, a.max_jobs, state, log, a.drain)
        except Exception as e:                                  # git or slurm hiccup: try again next loop
            import traceback
            log(f"loop error: {e!r}\n{traceback.format_exc(limit=4)}")
        state_file.write_text(json.dumps(state, indent=1))
        time.sleep(a.poll)
    log("driver exiting")


if __name__ == "__main__":
    main()
