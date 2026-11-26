"""Where candidates are scored: over ssh (bath), or through a git mailbox (any cluster, no login needed).

Both backends do the same three things for the orchestrator on the Mac:
    submit(key, harness, params)  ship one harness and ask for one evaluation
    result(key)                   None while running; else {"summary": ...} or {"rejected": [...]}
    fetch_export(key, dest)       copy back what the proposer may read

SshRemote  -- the Mac logs in and runs sbatch itself (needs a live ssh connection).
Mailbox    -- the Mac only pushes to and pulls from a private git repository. A driver job on the
              cluster (mb/evolve/driver.py) pulls requests, runs eval_cand.sbatch, and pushes results.
              Only outbound connections from the cluster are used, so nobody has to stay logged in.

Mailbox layout (one directory per evaluation, key = "<run>/<cid>/<split>"):
    requests/<key>/harness.py, request.json      written by the Mac
    results/<key>/summary.json | rejected.json   written by the driver
    results/<key>/export/...                     what the proposer may read (search only)
    heartbeat/<cluster>.json                      the driver's last loop (time, jobs in flight)
Selection and test results carry no export, so nothing from those splits can reach a workspace.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

# sbatch options per cluster; eval_cand.sbatch itself holds only what every cluster shares.
SBATCH_ARGS = {
    "bath": "--partition=h100_1 --qos=lowprio --gres=gpu:1",
    "orcd": "--partition=mit_preemptable --gres=gpu:l40s:1 --cpus-per-task=8 --mem=48G",
}
# Where the driver may place jobs, in order of preference, with the per-user cap of each partition.
# ORCD: mit_normal_gpu is not preempted (2 GPUs, 6 h); mit_preemptable adds 4 more but can be preempted
# (eval_cand.sbatch is --requeue and resumes).
PLACES = {
    "bath": [("--partition=h100_1 --qos=lowprio --gres=gpu:1", 8)],
    # whichever GPU (type and partition) starts first: the driver submits one copy per type through
    # race.sh (the QOS caps: 2 + 4 GPUs per user)
    "orcd": [("--requeue --cpus-per-task=8 --mem=48G --time=05:59:00", 6)],
}


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=True)


class SshRemote:
    def __init__(self, host="bath", code="~/code/multibench", runs="~/results/mb/evo_b"):
        self.host, self.code, self.runs = host, code, runs
        self.jobs = {}

    def sync_code(self, here):
        sh(f"rsync -a --delete --exclude __pycache__ {here}/ {self.host}:{self.code}/mb/evolve/")

    def _base(self, key):
        return f"{self.runs}/{key}"

    def submit(self, key, harness, params):
        b = self._base(key)
        sh(f"ssh {self.host} 'mkdir -p {b}/harness && rm -f {b}/out/summary.json'")
        sh(f"scp -q {harness} {self.host}:{b}/harness/harness.py")
        env = " ".join(f"{k}={v}" for k, v in env_for(params, f"{b}/harness", f"{b}/out",
                                                     f"{b}/export").items())
        self.jobs[key] = sh(f"ssh {self.host} 'cd ~/logs && {env} sbatch --parsable "
                            f"{SBATCH_ARGS[self.host]} {self.code}/mb/evolve/eval_cand.sbatch'").stdout.strip()

    def result(self, key):
        if sh(f"ssh {self.host} 'squeue -h -j {self.jobs[key]}'", check=False).stdout.strip():
            return None
        return read_result(lambda name: sh(f"ssh {self.host} 'cat {self._base(key)}/out/{name} 2>/dev/null'",
                                           check=False).stdout)

    def fetch_export(self, key, dest):
        sh(f"rsync -a {self.host}:{self._base(key)}/export/ {dest}/")


class Mailbox:
    def __init__(self, repo: Path):
        self.repo = Path(repo).expanduser()

    def _git(self, *args, check=True):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=check, text=True,
                              capture_output=True)

    @contextlib.contextmanager
    def lock(self):
        """Several orchestrators (and nothing else) share one clone: one git operation at a time."""
        with open(self.repo / ".git" / "mailbox.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def pull(self):
        """A failed pull (network, GitHub) is left for the next poll: callers poll in a loop."""
        with self.lock():
            r = self._git("pull", "-q", "--rebase", "--autostash", check=False)
            if r.returncode != 0:
                self._git("rebase", "--abort", check=False)
                print(f"mailbox pull failed (will retry): {(r.stderr or '').strip()[-200:]}", flush=True)

    def push(self, msg):
        with self.lock():
            self._push(msg)

    def _push(self, msg):
        self._git("add", "-A")
        if self._git("diff", "--cached", "--quiet", check=False).returncode == 0:
            return
        self._git("commit", "-qm", msg)
        for _ in range(5):                      # the driver may have pushed in between
            if self._git("push", "-q", check=False).returncode == 0:
                return
            self._git("pull", "-q", "--rebase", "--autostash")
        raise RuntimeError("mailbox push failed 5 times")

    def sync_code(self, here):
        pass                                    # the cluster runs the code its own checkout holds

    def submit(self, key, harness, params):
        """Idempotent: the same harness(es) and parameters under an existing key keep the request (and
        whatever the driver has already played or published for it), so a restarted Mac loop resumes."""
        with self.lock():
            self._git("pull", "-q", "--rebase", "--autostash", check=False)
            req = self.repo / "requests" / key
            if (req / "request.json").exists():
                old = json.loads((req / "request.json").read_text())
                want = {**params, **({"per_cell": True} if isinstance(harness, dict) else {})}
                files = ({f"harness_{c.replace(':', '_')}.py": h for c, h in harness.items()}
                         if isinstance(harness, dict) else {"harness.py": harness})
                same_files = (sorted(f.name for f in req.glob("harness*.py")) == sorted(files) and
                              all((req / n).read_bytes() == Path(h).read_bytes() for n, h in files.items()))
                if same_files and {k: old.get(k) for k in want} == want:
                    return
            shutil.rmtree(self.repo / "results" / key, ignore_errors=True)
            req.mkdir(parents=True, exist_ok=True)
            if isinstance(harness, dict):        # one harness file per task: {cell: path}
                for old in req.glob("harness*.py"):
                    old.unlink()
                for c, h in harness.items():
                    shutil.copy(h, req / f"harness_{c.replace(':', '_')}.py")
                params = {**params, "per_cell": True}
            else:
                shutil.copy(harness, req / "harness.py")
            (req / "request.json").write_text(json.dumps({**params, "submitted": time.time()}, indent=1))
            self._push(f"request {key}")

    def result(self, key):
        """None until the driver has a final answer (it resubmits incomplete runs itself)."""
        self.pull()
        with self.lock():
            return self._result(key)

    def _result(self, key):
        d = self.repo / "results" / key
        if (d / "failed.json").exists():
            raise RuntimeError(f"{key}: driver gave up: {(d / 'failed.json').read_text()[:300]}")
        if not ((d / "summary.json").exists() or (d / "rejected.json").exists()):
            return None
        return read_result(lambda name: (d / name).read_text() if (d / name).exists() else "")

    def fetch_export(self, key, dest):
        with self.lock():
            shutil.copytree(self.repo / "results" / key / "export", dest, dirs_exist_ok=True)


def play(box: "Mailbox", key, harness, cell, split, instances, visible_only=True, poll=10, extra=None):
    """Play these instances with this harness on the cluster's long-lived worker (mb/evolve/worker.py);
    returns the local folder of what the method may read. Scores stay in play_results/<key>/private.json,
    which the caller's method code must not read (our reports do). `extra` adds fields to the request
    (e.g. {"raw": True}: the worker also returns the raw episode records under play_results/<key>/raw/).
    A key already answered for the same harness and problems is not played again (resume)."""
    params = {"cell": cell, "split": split, "instances": list(instances), "visible_only": visible_only,
              **(extra or {})}
    with box.lock():
        box._git("pull", "-q", "--rebase", "--autostash")
        req = box.repo / "play_requests" / key
        res = box.repo / "play_results" / key
        if (res / "private.json").exists() and (req / "harness.py").exists() \
                and (req / "harness.py").read_bytes() == Path(harness).read_bytes():
            old = json.loads((req / "request.json").read_text())
            if {k: old.get(k) for k in params} == params:
                return res / "visible"
        shutil.rmtree(box.repo / "play_results" / key, ignore_errors=True)
        req.mkdir(parents=True, exist_ok=True)
        shutil.copy(harness, req / "harness.py")
        (req / "request.json").write_text(json.dumps({**params, "submitted": time.time()}))
        box._push(f"play request {key}")
    res = box.repo / "play_results" / key
    while True:
        box.pull()
        if (res / "private.json").exists():
            return res / "visible"
        time.sleep(poll)


def env_for(params, cand, out, dest):
    """Environment for eval_cand.sbatch, from the request's parameters."""
    env = {"CAND": cand, "OUT": out, "CELL": params["cell"], "SPLIT": params["split"], "N": params["n"]}
    if params.get("per_cell"):          # CAND is a folder of per-task folders: CAND/<cell_slug>/harness.py
        env["PER_CELL"] = 1
    if params.get("export"):
        env["DEST"] = dest
    if params.get("smoke"):
        env["SMOKE"] = 1
    return env


def read_result(read):
    """None if the evaluation has not produced a final answer; resubmission is the caller's choice."""
    rej = read("rejected.json")
    if rej:
        return {"rejected": json.loads(rej)["rejected"]}
    summ = read("summary.json")
    if summ:
        return {"summary": json.loads(summ)}
    return {"missing": True}


def make(spec):
    """'ssh:bath' or 'mailbox:~/path/to/clone'."""
    kind, _, arg = spec.partition(":")
    if kind == "ssh":
        return SshRemote(host=arg or "bath")
    if kind == "mailbox":
        return Mailbox(Path(arg))
    raise ValueError(spec)
