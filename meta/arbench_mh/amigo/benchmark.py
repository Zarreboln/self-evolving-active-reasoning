"""Run one AMIGO harness over a split and write the episodes.

Three things this does that the AR-Bench benchmarks do not need.

**It resumes.** Episodes are appended to `--out` as they finish and a rerun skips the ids already
there, so a VLM server that reaches its time limit mid-split costs the episodes in flight, not the
split. The serve job comes back on a new port, and the caller re-reads the endpoint.

**It audits cross-episode state.** On Guess Number a candidate discovered it could merge sixteen
concurrently running episodes into one model call and divide the per-episode cost by sixteen --
a property of this runner, not of the task. Before the run, every module-level name in the
harness's module is recorded; after it, they are compared. A candidate whose module globals
changed is reported as a violation and scored zero.

**It counts per episode.** A shared counter read under sixteen threads attributes whatever else
was in flight to whichever episode happened to finish, which inflated the AR-Bench numbers 2.2x
before it was found. Each episode gets its own counter, chained to a run total.
"""
from __future__ import annotations

import argparse
import base64
import copy
import importlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# run as a file path with -I, the way the loop runs the other benchmarks, so the package root has
# to go on sys.path here: -I drops both PYTHONPATH and the working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from arbench_mh.amigo.env import AMIGOEnv, BudgetExhausted   # noqa: E402
from arbench_mh.amigo.jev import Jev                          # noqa: E402

MODEL = os.environ.get("POLICY_MODEL", "Qwen/Qwen3.5-9B")
POLICY = os.environ["POLICY_BASE_URL"].rstrip("/")
SPLIT_ROOT = os.path.expanduser("~/amigo/splits")
SEALED_ROOT = os.path.expanduser("~/sealed/amigo")


class Counter:
    def __init__(self, parent=None):
        self.calls = self.in_tok = self.out_tok = self.jev_in = 0
        self.parent = parent

    def add(self, i, o, jev=0):
        self.calls += 1
        self.in_tok += i
        self.out_tok += o
        self.jev_in += jev
        if self.parent is not None:
            self.parent.add(i, o, jev)


def policy_call(counter):
    def call(messages, temperature=0.7, max_tokens=None):
        body = {"model": MODEL, "messages": messages, "temperature": temperature,
                "top_p": 0.8, "chat_template_kwargs": {"enable_thinking": False}}
        if max_tokens:
            body["max_tokens"] = max_tokens
        req = urllib.request.Request(POLICY + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer x"})
        for k in range(4):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    o = json.loads(r.read().decode())
                u = o.get("usage") or {}
                counter.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                return o["choices"][0]["message"]["content"]
            except Exception:
                time.sleep(2 * (k + 1))
        return ""
    return call


def run_one(ex, cls, total):
    counter = Counter(parent=total)
    jev = Jev()
    env = AMIGOEnv(ex, jev)
    harness = cls(env, policy_call(counter))
    try:
        pred = harness.solve()
    except BudgetExhausted:
        pred = 0
    except Exception as e:                      # a candidate that throws scores zero, not crash
        pred = -1
        err = "%s: %s" % (type(e).__name__, e)
    else:
        err = None
    counter.jev_in += jev.in_tok
    skips = sum(1 for _, a in env.history if a == "Skip")
    return {"id": ex["id"], "pred": pred, "correct": bool(env.score(pred)),
            "gallery_size": env.n, "turns": len(env.history), "skips": skips,
            "skip_rate": skips / max(len(env.history), 1),
            "llm_calls": counter.calls, "input_token": counter.in_tok,
            "output_token": counter.out_tok, "jev_input_token": counter.jev_in,
            "total_token": counter.in_tok + counter.out_tok,
            "error": err,
            "transcript": [{"question": q, "answer": a} for q, a in env.history]}


def module_globals(mod):
    """A shallow fingerprint of the module's own names, for the cross-episode state audit."""
    out = {}
    for k, v in vars(mod).items():
        if k.startswith("__") or callable(v) or isinstance(v, type):
            continue
        try:
            out[k] = copy.deepcopy(v)
        except Exception:
            out[k] = repr(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tau06")
    ap.add_argument("--harness", required=True)
    ap.add_argument("--split", default="search")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-out", default=None,
                    help="where the loop reads the score from")
    a = ap.parse_args()

    # the held-out split lives outside the directory the proposer works in
    root = SEALED_ROOT if a.split == "test" else SPLIT_ROOT
    path = os.path.join(root, a.tier, a.split + ".json")
    examples = json.load(open(path))["examples"]
    if a.limit:
        examples = examples[:a.limit]

    out = os.path.expanduser(a.out)
    done = {}
    if os.path.exists(out):                     # resume
        for r in json.load(open(out)):
            done[r["id"]] = r
    todo = [e for e in examples if e["id"] not in done]
    print("%s / %s %s  n=%d  (%d already done)"
          % (a.tier, a.split, a.harness, len(examples), len(done)), flush=True)

    mod = importlib.import_module("arbench_mh.amigo.agents." + a.harness)
    before = module_globals(mod)
    cls = mod.Harness

    total = Counter()
    t0 = time.time()
    rows = list(done.values())
    if todo:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(lambda e: run_one(e, cls, total), todo):
                rows.append(r)
                json.dump(rows, open(out, "w"), indent=1)     # checkpoint every episode

    after = module_globals(mod)
    changed = sorted(k for k in set(before) | set(after)
                     if repr(before.get(k)) != repr(after.get(k)))
    order = {e["id"]: i for i, e in enumerate(examples)}
    rows.sort(key=lambda r: order.get(r["id"], 1 << 30))
    json.dump(rows, open(out, "w"), indent=1)

    n = len(rows)
    acc = sum(1 for r in rows if r["correct"]) / n
    chance = sum(1 / r["gallery_size"] for r in rows) / n
    skip = sum(r["skip_rate"] for r in rows) / n
    print("%-24s acc=%.1f%%  chance=%.1f%%  skip=%.1f%%  turns=%.1f  tok/ep=%.0f  %.0fs"
          % (a.harness, 100 * acc, 100 * chance, 100 * skip,
             sum(r["turns"] for r in rows) / n,
             sum(r["total_token"] for r in rows) / n, time.time() - t0), flush=True)
    if a.val_out:
        json.dump({"n": n, "accuracy": acc, "chance": chance, "skip_rate": skip,
                   "turns": sum(r["turns"] for r in rows) / n,
                   "total_tokens": sum(r["total_token"] for r in rows) / n,
                   "jev_input_tokens": sum(r["jev_input_token"] for r in rows) / n,
                   "errors": sum(1 for r in rows if r["error"])},
                  open(os.path.expanduser(a.val_out), "w"), indent=1)

    if changed:
        print("CROSS-EPISODE STATE VIOLATION: module globals changed during the run: %s"
              % ", ".join(changed), flush=True)
        sys.exit(3)
    print("  -> %s" % out)


if __name__ == "__main__":
    main()
