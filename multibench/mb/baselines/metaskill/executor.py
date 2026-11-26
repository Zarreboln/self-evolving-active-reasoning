"""Where episodes are played. The method's control loop runs on the Mac; episodes run on ORCD.

Two calls, both with what the environment shows and nothing else:

    public_card(cell, instance) -> {"instructions", "observation", "max_steps", "reads_full_reply",
                                    "valid_actions"}             what a harness sees before its first action
    play(harness_path, cell, split, instances, tag) -> [episode]  one episode per problem, in the given order

An episode is {"instance", "score", "success", "steps", "end", "error", "model_calls", "trace"}, where
`trace` is broker._render's trace.md (every message the Target saw, its replies, actions and
observations). `instances` is a list of problem ids, or an int n for "the first n problems of the
split" (the broker's own order; that is how the search split is defined in experiment B).

Implementations:
  LocalExecutor    plays through the real mb/evolve/child.py (subprocess, `python -I`, empty cwd and
                   environment) against given env/chat factories. With fake factories it is the CPU test;
                   it is also the mock run after every build.
  MailboxExecutor  batches through the existing git mailbox (mb/evolve/remote.py + driver.py +
                   eval_cand.sbatch): one request per (harness, cell, split). Fine wherever nothing adapts
                   between problems, which is every play() of this method (the bank is fixed during a
                   batch and frozen at test time). Cards are read from files made on the cluster by
                   prep_cluster.py.
The parent's long-lived ORCD worker replaces MailboxExecutor by implementing the same two calls.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mb.evolve.broker import CHAT_CAP, CHILD, LIB, _render, _source

EPISODE_TIMEOUT = 600
FIRST_SEARCH = "first:search"       # public_card of the first search problem (a dev task's build card)


class Executor:
    def public_card(self, cell: str, instance) -> dict:
        raise NotImplementedError

    def play(self, harness_path: Path, cell: str, split: str, instances, tag: str) -> list[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------- local (child.py) ----------

def run_child(harness_path, env, chat, timeout=EPISODE_TIMEOUT) -> dict:
    """One episode of a harness file through child.py, answered like broker.episode does."""
    work = tempfile.mkdtemp(prefix="ms_cand_")
    errf = open(os.path.join(work, "stderr.txt"), "w+")
    child = subprocess.Popen([sys.executable, "-I", CHILD, str(harness_path), LIB], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=errf, text=True, cwd=work,
                             env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
    t0 = time.time()

    def send(m):
        child.stdin.write(json.dumps(m) + "\n")
        child.stdin.flush()

    calls, steps, replies, prompts = [], [], [], []
    env_text, end, error = env.instructions, None, None
    send({"instructions": env.instructions, "observation": env.observation, "max_steps": env.max_steps,
          "reads_full_reply": env.reads_full_reply, "admissible": env.admissible()})
    try:
        while True:
            if time.time() - t0 > timeout:
                end, error = "crash", f"timeout after {timeout}s"
                break
            line = child.stdout.readline()
            if not line:
                end, error = "crash", f"harness process exited (code {child.wait()})"
                break
            m = json.loads(line)
            if m["op"] == "chat":
                if len(calls) >= CHAT_CAP * max(env.max_steps, 1):
                    send({"error": f"model-call budget of {CHAT_CAP} per step used up"})
                    continue
                reply = chat(m["messages"], temperature=m.get("temperature"), max_tokens=m.get("max_tokens"))
                calls.append({"step": env.steps, "messages": m["messages"], "reply": reply})
                replies.append(reply)
                prompts.append("\n".join(str(x.get("content", "")) for x in m["messages"] if x.get("role") != "assistant"))
                send({"reply": reply})
            elif m["op"] == "act":
                if env.done:
                    send({"error": "the episode is over"})
                    continue
                src = _source(m["action"], replies, prompts, env_text)
                obs = env.act(m["action"])
                env_text += "\n" + obs
                steps.append({"action": m["action"], "observation": obs, "source": src})
                replies, prompts = [], []
                send({"obs": obs, "done": env.done, "steps": env.steps, "admissible": env.admissible()})
            else:
                end, error = m["op"], m.get("error")
                break
    finally:
        child.kill()
        errf.seek(0)
        stderr = errf.read()[-2000:]
        errf.close()
    if end == "end" and not env.done:
        error = "run_episode returned before the episode ended"
    r = {**env.result(), "steps_log": steps, "calls": calls, "end": end, "error": error, "harness_stderr": stderr}
    return r


def to_episode(r: dict, instance) -> dict:
    return {"instance": instance, "score": r.get("score"), "success": r.get("success"), "steps": r.get("steps"),
            "end": r.get("end"), "error": r.get("error"), "model_calls": len(r.get("calls", [])),
            "trace": _render(r)}


class LocalExecutor(Executor):
    """make_env(cell, instance) -> env with the TextEnv attributes + result(); make_chat(cell) -> chat."""

    def __init__(self, make_env, make_chat, instances_of):
        self.make_env, self.make_chat, self.instances_of = make_env, make_chat, instances_of
        self.log = []                         # (tag, cell, split, instance), for the tests

    def public_card(self, cell, instance):
        if instance == FIRST_SEARCH:
            instance = self.instances_of(cell, "search", 1)[0]
        env = self.make_env(cell, instance)
        return {"instructions": env.instructions, "observation": env.observation, "max_steps": env.max_steps,
                "reads_full_reply": env.reads_full_reply, "valid_actions": env.admissible()}

    def play(self, harness_path, cell, split, instances, tag):
        ids = self.instances_of(cell, split, instances) if isinstance(instances, int) else list(instances)
        out = []
        for inst in ids:
            self.log.append((tag, cell, split, inst))
            r = run_child(harness_path, self.make_env(cell, inst), self.make_chat(cell))
            out.append(to_episode(r, inst))
        return out


# ---------------------------------------------------------------- git mailbox (batch) --------

class MailboxExecutor(Executor):
    """Batches through mb/evolve/remote.py. Requests are keyed <run>/<tag>/<split>, so a resubmitted
    request after a Mac restart is the same request (the driver resumes episodes on disk)."""

    def __init__(self, spec: str, run: str, cards_dir: Path, poll: int = 60, via_worker: bool = True):
        from mb.evolve import remote
        self.remote, self.run, self.cards, self.poll = remote.make(spec), run, Path(cards_dir), poll
        self.via_worker = via_worker          # search batches on the long-lived worker, not a GPU job each

    def public_card(self, cell, instance):
        from mb.baselines.metaskill.prep_cluster import opaque
        name = "first_search" if instance == FIRST_SEARCH else opaque(cell, str(instance))
        p = self.cards / cell.replace(":", "_") / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: run prep_cluster.py on the cluster and copy cards/ here")
        return json.loads(p.read_text())

    def play(self, harness_path, cell, split, instances, tag):
        key = f"metaskill/{self.run}/{tag}/{split}"
        params = {"cell": cell, "split": split, "export": split == "search", "smoke": False}
        if isinstance(instances, int) and getattr(self, "via_worker", False):
            return self._play_first_n(harness_path, cell, split, instances, key)
        if isinstance(instances, int):
            params["n"] = instances
        else:
            # an explicit list (test: the shared fixed order) goes to the long-lived worker, which plays
            # exactly these problems; the batch driver would play the first n of the split instead
            return self._play_listed(harness_path, cell, split, list(instances), key)
        while True:
            self.remote.submit(key, str(harness_path), params)
            while (r := self.remote.result(key)) is None:
                time.sleep(self.poll)
            if "rejected" in r:
                return [{"instance": None, "score": 0.0, "success": False, "steps": 0, "end": "rejected",
                         "error": "; ".join(r["rejected"]), "model_calls": 0, "trace": ""}]
            if "summary" in r and r["summary"].get("infra_errors", 0) == 0:
                break
        if not params["export"]:
            return [{"instance": None, "summary": r["summary"]}]       # test: per-episode results stay on the cluster
        dest = Path(tempfile.mkdtemp(prefix="ms_export_"))
        self.remote.fetch_export(key, dest)
        eps = []
        for d in sorted((dest / "episodes").iterdir()):
            res = json.loads((d / "result.json").read_text())
            eps.append({"instance": res.get("instance_id"), "score": res.get("score"), "success": res.get("success"),
                        "steps": res.get("steps"), "end": res.get("end"), "error": res.get("error"),
                        "model_calls": res.get("model_calls"), "trace": (d / "trace.md").read_text()})
        return eps


    def _play_first_n(self, harness_path, cell, split, n, key):
        """Search episodes (the first n of the split) on the long-lived worker instead of one GPU job
        each; the worker returns the broker's export, the same files fetch_export gives."""
        from mb.evolve import remote
        for _ in range(3):
            vis = remote.play(self.remote, key, str(harness_path), cell, split, [], visible_only=False,
                              extra={"n": n})
            if (vis / "rejected.json").exists():
                r = json.loads((vis / "rejected.json").read_text())
                return [{"instance": None, "score": 0.0, "success": False, "steps": 0, "end": "rejected",
                         "error": "; ".join(r.get("rejected", [])), "model_calls": 0, "trace": ""}]
            priv = json.loads((vis.parent / "private.json").read_text())
            if priv.get("infra_errors", 0) == 0 and (vis / "episodes").is_dir():
                eps = []
                for d in sorted((vis / "episodes").iterdir()):
                    res = json.loads((d / "result.json").read_text())
                    eps.append({"instance": res.get("instance_id"), "score": res.get("score"),
                                "success": res.get("success"), "steps": res.get("steps"), "end": res.get("end"),
                                "error": res.get("error"), "model_calls": res.get("model_calls"),
                                "trace": (d / "trace.md").read_text()})
                return eps
            with self.remote.lock():
                import shutil
                shutil.rmtree(vis.parent, ignore_errors=True)
                self.remote._push(f"retry {key}")
        raise RuntimeError(f"{key}: worker could not play the search problems after 3 tries")

    def _play_listed(self, harness_path, cell, split, ids, key):
        """Test episodes on the worker. Per-episode scores come from private.json, which only our
        report code reads: at test time the bank is frozen and nothing goes back to the Builder."""
        from mb.evolve import remote
        for _ in range(3):
            vis = remote.play(self.remote, key, str(harness_path), cell, split, ids, visible_only=True)
            res = vis.parent
            if (vis / "rejected.json").exists():
                r = json.loads((vis / "rejected.json").read_text())
                return [{"instance": i, "score": 0.0, "success": False, "steps": 0, "end": "rejected",
                         "error": "; ".join(r.get("rejected", [])), "model_calls": 0} for i in ids]
            priv = json.loads((res / "private.json").read_text())
            per = priv.get("per_episode") or {}
            if priv.get("infra_errors", 0) == 0 and all(i in per for i in ids):
                return [{"instance": i, **{k: per[i].get(k) for k in ("score", "success", "steps", "end", "error",
                                                                      "model_calls")}} for i in ids]
            with self.remote.lock():                       # incomplete: drop it so the next try replays
                import shutil
                shutil.rmtree(res, ignore_errors=True)
                self.remote._push(f"retry {key}")
        raise RuntimeError(f"{key}: worker could not play all problems after 3 tries")


def mean_score(eps):
    s = [e["score"] for e in eps if isinstance(e.get("score"), (int, float)) and not math.isnan(e["score"])]
    return sum(s) / len(s) if s else 0.0
