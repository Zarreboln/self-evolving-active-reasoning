"""The two seams of the integration: model sessions (`agent_session`) and episode play (`play`).

Both run on the Mac, where the control loop lives.

agent_session(role, label, context_dir, mission, harvest) -> Reply
    Every LLM role of Harness² (INV/DOM editor, composer, blind selector, experience reflector) is
    one agentic session, as in the official code (core/proposer.py runs OpenCode or the Claude CLI
    in a fresh directory holding `context/`). Here: Claude Code, headless, inside our Docker
    sandbox (ar_harness/sandbox/run_proposer.py) on the user's subscription, model claude-opus-5-5.
    The credential stays in its 600 file and is read only by run_proposer, never here.

play(harness_path, cell, split, instance_ids, key=...) -> [episode dict]
    Plays a harness on problems and returns the broker's episode records (mb/evolve/broker.py:
    `calls`, `steps_log`, `end`, `error`, `score`, ...). Candidate code is only ever executed by the
    broker's child sandbox (`python -I child.py` after mb/evolve/check.py), never imported here.
    The Harness² loop reads only the label-free part of a record (evidence.py); scores are kept in
    `private/` for our reports.

Both are wrapped in on-disk caches (CachedSession, CachedPlay) keyed by content and label, so a run
killed at any point resumes without repeating a finished session or episode.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

TRANSPORT_ATTEMPTS = 2           # official config.TRANSPORT_ATTEMPTS
SESSION_CONCURRENCY = 2          # official config.PROPOSER_CONCURRENCY
SESSION_TIMEOUT = 3600           # official ChannelSpec.timeout
MODEL = "claude-opus-5-5"


class SessionError(RuntimeError):
    """A session produced no usable reply after its attempts (official ProposerError)."""


@dataclass
class Reply:
    label: str
    text: str
    ok: bool = True
    workspace: Path | None = None
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    cached: bool = False
    error: str = ""


class AgentSession(Protocol):
    def __call__(self, role: str, label: str, context_dir: Path | None, mission: str,
                 harvest: str | None = None) -> Reply: ...


def fingerprint(label: str, mission: str, context_dir: Path | None) -> str:
    """Official proposer._call_fingerprint, minus the absolute path (our step dirs are stable)."""
    h = hashlib.sha256()
    for v in (label, mission):
        h.update(v.encode("utf-8", "replace") + b"\0")
    if context_dir and Path(context_dir).is_dir():
        root = Path(context_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


class CachedSession:
    """Audit trail + cache + retries around a raw session runner (official Proposer.call)."""

    def __init__(self, runner: Callable, audit_dir: Path, attempts: int = TRANSPORT_ATTEMPTS):
        self.runner, self.audit, self.attempts = runner, Path(audit_dir), attempts
        self.audit.mkdir(parents=True, exist_ok=True)
        self.sem = threading.BoundedSemaphore(SESSION_CONCURRENCY)
        self.lock = threading.Lock()

    def _paths(self, label, fp):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label) + "." + fp
        return {k: self.audit / f"{safe}.{k}" for k in ("mission.md", "response.md", "meta.json", "workspace")}

    def __call__(self, role, label, context_dir, mission, harvest=None) -> Reply:
        fp = fingerprint(label, mission, context_dir)
        p = self._paths(label, fp)
        if p["response.md"].is_file() and (not harvest or p["workspace"].is_dir()):
            meta = json.loads(p["meta.json"].read_text()) if p["meta.json"].is_file() else {}
            return Reply(label, p["response.md"].read_text(), workspace=p["workspace"] if harvest else None,
                         usage=meta.get("usage", {}), cached=True)
        last, usage_all, t0 = "", [], time.time()
        for attempt in range(1, self.attempts + 1):
            if attempt > 1:
                time.sleep(min(10 * (attempt - 1), 30))
            tmp = Path(tempfile.mkdtemp(prefix="h2sess_", dir=self.audit))
            try:
                if context_dir and Path(context_dir).is_dir():
                    shutil.copytree(context_dir, tmp / "context")
                with self.sem:
                    text, usage, err = self.runner(role, label, tmp, mission)
                usage_all.append(usage)
                if not (text or "").strip():
                    last = err or "empty reply"
                    continue
                if harvest:
                    shutil.rmtree(p["workspace"], ignore_errors=True)
                    if (tmp / harvest).is_dir():
                        shutil.copytree(tmp / harvest, p["workspace"])
                p["mission.md"].write_text(mission)
                p["meta.json"].write_text(json.dumps({"role": role, "label": label, "attempts": attempt,
                                                      "seconds": round(time.time() - t0, 1),
                                                      "usage": merge_usage(usage_all)}, indent=1))
                p["response.md"].write_text(text)           # written last: its presence marks done
                return Reply(label, text, workspace=p["workspace"] if harvest and p["workspace"].is_dir() else None,
                             usage=merge_usage(usage_all), seconds=round(time.time() - t0, 1))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        with self.lock, open(self.audit / "failures.jsonl", "a") as f:
            f.write(json.dumps({"label": label, "fp": fp, "error": last, "usage": merge_usage(usage_all)}) + "\n")
        raise SessionError(f"[{label}] {last}")


def merge_usage(us):
    out = {}
    for u in us:
        for k, v in (u or {}).items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
    return out


class SandboxClaudeCode:
    """Raw runner: one headless Claude Code session in the ar-proposer container (subscription).

    The workspace mounted at /work holds `context/` (the official evidence tree) and TASK.md (the
    official mission). run_proposer.run reads TASK.md and passes it as the prompt. The final message
    of the stream-json log is the reply, as the official claude_cli channel reads it.
    """

    def __init__(self, logs: Path, model: str = MODEL, timeout: int = SESSION_TIMEOUT):
        self.logs, self.model, self.timeout = Path(logs), model, timeout
        sandbox = Path(__file__).resolve().parents[4] / "sandbox"
        sys.path.insert(0, str(sandbox))
        import run_proposer          # noqa: E402  (reads the credential file itself)
        self.rp = run_proposer

    def __call__(self, role, label, ws: Path, mission: str):
        (ws / "TASK.md").write_text(mission)
        log = self.logs / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}.{int(time.time())}.jsonl"
        rc = self.rp.run(ws, log, model=self.model, timeout=self.timeout)
        text, usage, err = parse_stream(log)
        if rc not in (0,) and not text:
            err = err or f"claude exited rc={rc}"
        return text, usage, err


def parse_stream(log: Path):
    """(final text, usage, error) from a `claude -p --output-format stream-json` log."""
    text, usage, err = "", {}, ""
    if not log.exists():
        return text, usage, "no log"
    for line in log.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            if e.get("is_error") or e.get("subtype") not in (None, "success"):
                err, text = f"CLI error: {e.get('subtype')} {str(e.get('result'))[:200]}", ""
            else:
                text = str(e.get("result") or "")
            u = e.get("usage") or {}
            usage = {"sessions": 1, "turns": e.get("num_turns", 0),
                     "input_tokens": u.get("input_tokens", 0),
                     "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
                     "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
                     "output_tokens": u.get("output_tokens", 0),
                     "cost_usd_equiv": e.get("total_cost_usd", 0) or 0}
    return text, usage, err


# ----------------------------------------------------------------------------------------------- play

class Play(Protocol):
    def __call__(self, harness_path: Path, cell: str, split: str, instance_ids: list[str], *,
                 key: str) -> list[dict]: ...


class CachedPlay:
    """One file per (key, instance) under private/episodes/. An `infra` record is retried."""

    def __init__(self, play: Play, store: Path, retries: int = 3):
        self.play, self.store, self.retries = play, Path(store), retries
        self.calls = 0

    def path(self, key, inst):
        return self.store / (re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + "__" +
                             re.sub(r"[^A-Za-z0-9_.-]+", "_", str(inst)) + ".json")

    def __call__(self, harness_path, cell, split, instance_ids, *, key):
        out = []
        for inst in instance_ids:
            p = self.path(key, inst)
            if p.is_file():
                out.append(json.loads(p.read_text()))
                continue
            for _ in range(self.retries):
                self.calls += 1
                [e] = self.play(harness_path, cell, split, [inst], key=key)
                if "infra" not in e:
                    break
            if "infra" in e:
                raise RuntimeError(f"{key}/{inst}: infrastructure failure after {self.retries} tries: {e['infra']}")
            e = {**e, "cell": cell, "split": split, "play_key": key}
            p.parent.mkdir(parents=True, exist_ok=True)
            p.with_suffix(".tmp").write_text(json.dumps(e))
            os.replace(p.with_suffix(".tmp"), p)
            out.append(e)
        return out


class LocalBrokerPlay:
    """Plays through mb/evolve/broker.py on a machine that has the environments and the vLLM
    server (POLICY_BASE_URLS, POLICY_MODEL set, as in eval_cand.sbatch). This is what an ORCD
    worker runs per request; on the Mac it is unusable (no envs, no GPU)."""

    def __init__(self, work: Path):
        self.work = Path(work)

    def __call__(self, harness_path, cell, split, instance_ids, *, key):
        from mb.evolve.check import problems
        from mb.screen import _path
        src = Path(harness_path).read_text()
        bad = problems(src)
        bench, sub = cell.split(":", 1)
        cand = self.work / "cand" / re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        cand.mkdir(parents=True, exist_ok=True)
        (cand / "harness.py").write_text(src)
        url = os.environ["POLICY_BASE_URLS"].split(",")[0].strip().rstrip("/")
        out = []
        for inst in instance_ids:
            if bad:     # never reached in practice: every tree is validated before it is played
                out.append({"instance_id": inst, "score": 0.0, "end": "rejected", "error": "; ".join(bad),
                            "calls": [], "steps_log": []})
                continue
            od = self.work / "out" / re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            job = [str(cand), bench, sub, inst, url, str(od)]
            cp = subprocess.run([sys.executable, "-m", "mb.evolve.broker", "one", json.dumps(job)],
                                capture_output=True, text=True, timeout=2400)
            p = Path(_path(str(od), bench, sub, inst))
            out.append(json.loads(p.read_text()) if cp.returncode == 0 and p.exists()
                       else {"instance_id": inst, "score": math.nan, "infra": cp.stderr[-400:]})
        return out


class MailboxPlay:
    """The ORCD long-lived worker (mb/evolve/worker.py) reached through the git mailbox
    (mb/evolve/remote.play). One request per (key, instance), keyed `<prefix>/<key>/<instance>`, so a
    resubmitted key after a restart returns the stored records. The worker runs the broker, which runs
    mb.evolve.check on the harness first, and returns the raw episode records (calls, steps_log) that
    LocalBrokerPlay would give; CachedPlay keeps them under private/."""

    def __init__(self, clone: str, prefix: str = "harness2"):
        from mb.evolve.remote import Mailbox
        self.box, self.prefix = Mailbox(Path(clone).expanduser()), prefix.strip("/")

    def __call__(self, harness_path, cell, split, instance_ids, *, key):
        from mb.evolve import remote
        out = []
        for inst in instance_ids:
            k = f"{self.prefix}/{key}/{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(inst))}"
            vis = remote.play(self.box, k, harness_path, cell, split, [inst], visible_only=False,
                              extra={"raw": True})
            res = vis.parent
            if (res / "visible" / "rejected.json").exists():
                why = json.loads((res / "visible" / "rejected.json").read_text())
                out.append({"instance_id": inst, "score": 0.0, "end": "rejected",
                            "error": "; ".join(why.get("rejected", [])) if isinstance(why, dict) else str(why),
                            "calls": [], "steps_log": []})
                continue
            raw = sorted((res / "raw").glob("*.json")) if (res / "raw").is_dir() else []
            rec = json.loads(raw[0].read_text()) if raw else None
            if rec is None or "infra" in rec:          # drop the stored answer so a retry plays again
                with self.box.lock():
                    shutil.rmtree(res, ignore_errors=True)
                    self.box._push(f"retry {k}")
                rec = {"instance_id": inst, "score": math.nan,
                       "infra": (rec or {}).get("infra") or f"{k}: no episode record returned"}
            out.append(rec)
        return out
