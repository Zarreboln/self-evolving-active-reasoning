"""Evaluate one candidate harness on one split of one cell, and export what the proposer may read.

The broker is the only side that holds the environment and the model endpoint. For every episode
it starts mb/evolve/child.py with the candidate in a fresh process (empty cwd, empty environment,
`python -I`), sends it the instructions, and answers its `chat` and `act` requests. So the
candidate gets exactly what a harness gets in mb/harness/: the instructions, one observation per
action, the admissible actions, and model calls.

Rules, as in MAS.S62 HW2 (README, "What the loop saves"), adapted to this suite:
  * a harness may do anything a harness does, including deciding actions in code;
  * every model call and action is recorded, and each action is labelled by where it came from
    (`source`: model / model-picked / code; see `_source`). The labels are ours: `export` leaves
    them out of what the proposer reads;
  * the proposer reads, per candidate, its search scores and, per episode, everything the model
    saw and did (trace.md) plus the episode's result (result.json). That includes what the
    environment itself says, e.g. Wordle naming the secret after a lost game. It never reads the
    instance key, the environment's internals, the selection or test splits, or our labels.

Errors: a harness that raises, returns early, or calls the model more than CHAT_CAP times per step
keeps the score the environment gives at that point; that is the harness's result. An unreachable
model server is infrastructure, not the harness: the episode is marked `infra` and reruns on resume.

    python -m mb.evolve.broker run    --cand DIR --cell wordle:basic5 --split search --n 30 --out OUT
    python -m mb.evolve.broker export --out OUT --dest DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

from mb.benches import get
from mb.screen import _path, _slug

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "child.py")
LIB = os.path.join(HERE, "ws_template")          # harness_lib.py lives here
CHAT_CAP = 8                  # model calls per environment step, on average over the episode
EPISODE_TIMEOUT = 2400
_TOK = re.compile(r"[a-z0-9]+")
_FORMAT_TOKENS = {"action"}


def _tokens(text):
    return set(_TOK.findall((text or "").lower()))


def _source(action, replies, prompts, env_text):
    """Where an action came from: 'model' if its words are in this step's model replies and were not
    put in front of the model by the harness; 'model-picked' if they are in the replies but the
    harness had shown them (a candidate the harness listed, not something the environment said);
    'code' if no reply of this step contains them."""
    words = _tokens(action) - _FORMAT_TOKENS
    if not words:
        return "model" if replies else "code"
    if not any(words <= _tokens(r) for r in replies):
        return "code"
    shown = set().union(*(_tokens(p) for p in prompts)) if prompts else set()
    return "model-picked" if words <= shown and not words <= _tokens(env_text) else "model"


class Infra(Exception):
    pass


def episode(cand, bench, sub, inst, url, model):
    from mb.llm import Counter, make_chat
    b = get(bench)
    pc, rc = Counter(), Counter()
    responder = None
    if b.responder:
        if os.environ.get("MB_USE_JEV", "1") == "1":
            from mb.jev import Jev, responder_for
            responder = responder_for(bench, Jev())
        responder = responder or make_chat(url, model, rc)
    env = b.make(sub, inst, responder)
    chat = make_chat(url, model, pc)
    work = tempfile.mkdtemp(prefix="cand_")
    errf = open(os.path.join(work, "stderr.txt"), "w+")
    child = subprocess.Popen([sys.executable, "-I", CHILD, os.path.join(cand, "harness.py"), LIB],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                             cwd=work, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})

    def send(m):
        child.stdin.write(json.dumps(m) + "\n")
        child.stdin.flush()

    calls, steps = [], []
    replies, prompts = [], []                 # since the last action
    env_text = env.instructions
    end, error = None, None
    send({"instructions": env.instructions, "max_steps": env.max_steps,
          "reads_full_reply": env.reads_full_reply, "admissible": env.admissible()})
    try:
        while True:
            line = child.stdout.readline()
            if not line:
                end, error = "crash", f"harness process exited (code {child.wait()})"
                break
            m = json.loads(line)
            if m["op"] == "chat":
                if len(calls) >= CHAT_CAP * max(env.max_steps, 1):
                    send({"error": f"model-call budget of {CHAT_CAP} per step used up"})
                    continue
                msgs = m["messages"]
                try:
                    reply = chat(msgs, temperature=m.get("temperature", 0.7), max_tokens=m.get("max_tokens", 8192))
                except urllib.error.HTTPError as e:      # e.g. 400: the prompt outgrew the context
                    calls.append({"step": env.steps, "messages": msgs, "error": f"HTTP {e.code}"})
                    send({"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"})
                    continue
                except Exception as e:                   # server unreachable: not the harness's doing
                    raise Infra(repr(e))
                calls.append({"step": env.steps, "messages": msgs, "reply": reply})
                replies.append(reply)
                prompts.append("\n".join(str(x.get("content", "")) for x in msgs if x.get("role") != "assistant"))
                send({"reply": reply})
            elif m["op"] == "act":
                if env.done:
                    send({"error": "the episode is over"})
                    continue
                action = m["action"]
                src = _source(action, replies, prompts, env_text)
                obs = env.act(action)
                env_text += "\n" + obs
                steps.append({"action": action, "observation": obs, "source": src,
                              "calls_this_step": len(replies)})
                replies, prompts = [], []
                send({"obs": obs, "done": env.done, "steps": env.steps, "admissible": env.admissible()})
            elif m["op"] in ("end", "crash"):
                end = m["op"]
                error = m.get("error")
                break
    finally:
        try:
            child.kill()
        except Exception:
            pass
        errf.seek(0)
        stderr = errf.read()[-2000:]
        errf.close()
    if end == "end" and not env.done:
        error = "run_episode returned before the episode ended"
    r = {**env.result(), "steps_log": steps, "calls": calls, "end": end, "error": error,
         "harness_stderr": stderr, "policy_tokens": pc.as_dict(), "responder_tokens": rc.as_dict()}
    r["source_counts"] = {k: sum(s["source"] == k for s in steps) for k in ("model", "model-picked", "code")}
    return r


def _write(p, r):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(r, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)


def _one(job):
    cand, bench, sub, inst, url, out = job
    t0 = time.time()
    try:
        r = episode(cand, bench, sub, inst, url, os.environ["POLICY_MODEL"])
    except Infra as e:
        r = {"bench": bench, "subtask": sub, "instance": inst, "score": float("nan"), "infra": repr(e)}
    r["seconds"] = round(time.time() - t0, 1)
    _write(_path(out, bench, sub, inst), r)


def _one_subprocess(job):
    cand, bench, sub, inst, url, out = job
    try:
        cp = subprocess.run([sys.executable, "-m", "mb.evolve.broker", "one", json.dumps(job)],
                            capture_output=True, text=True, timeout=EPISODE_TIMEOUT)
        if cp.returncode == 0:
            return inst, json.load(open(_path(out, bench, sub, inst))).get("score")
        err = f"exit {cp.returncode}: {cp.stderr.strip()[-400:]}"
    except subprocess.TimeoutExpired:
        err = f"timeout after {EPISODE_TIMEOUT}s"
    # a broker failure (not a harness error, which is recorded inside the episode) reruns on resume
    _write(_path(out, bench, sub, inst), {"bench": bench, "subtask": sub, "instance": inst,
                                          "score": float("nan"), "infra": err})
    return inst, None


def instances(cell, split, n):
    bench, sub = cell.split(":", 1)
    return get(bench).instances(sub, split, limit=n)


def run(a):
    from mb.evolve.check import leaks, problems
    if os.path.exists(os.path.join(a.out, "rejected.json")):
        return
    src = open(os.path.join(a.cand, "harness.py")).read()
    bad = problems(src)
    if a.seen_split:
        lk = leaks(src, instances(a.cell, a.seen_split, a.seen_n), instances(a.cell, "selection", None))
        bad += [lk] if lk else []
    os.makedirs(a.out, exist_ok=True)
    if bad:
        json.dump({"rejected": bad}, open(os.path.join(a.out, "rejected.json"), "w"), indent=1)
        print("rejected:", bad, flush=True)
        return
    urls = [u.strip().rstrip("/") for u in os.environ["POLICY_BASE_URLS"].split(",") if u.strip()]
    bench, sub = a.cell.split(":", 1)
    jobs, done = [], 0
    for inst in instances(a.cell, a.split, a.n):
        p = _path(a.out, bench, sub, inst)
        if os.path.exists(p) and "infra" not in json.load(open(p)):
            done += 1
            continue
        jobs.append((os.path.abspath(a.cand), bench, sub, inst, urls[len(jobs) % len(urls)], a.out))
    print(f"{done} on disk, {len(jobs)} to run", flush=True)
    if a.smoke and jobs and not done:
        # HW2's smoke test (DGM): one episode first; a child that cannot run at all is rejected
        # before the rest. The reward is ignored, one episode says nothing about quality.
        _one_subprocess(jobs[0])
        e = json.load(open(_path(a.out, bench, sub, jobs[0][3])))
        why = ("crashes: " + (e.get("error") or "")[-300:] if e.get("end") == "crash"
               else "made no model call" if "infra" not in e and not e.get("calls") else "")
        if why:
            json.dump({"rejected": [why]}, open(os.path.join(a.out, "rejected.json"), "w"), indent=1)
            print("rejected:", why, flush=True)
            return
        jobs = jobs[1:]
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(_one_subprocess, j) for j in jobs]):
            f.result()
    summarize(a.out)


def _episodes(out):
    for root, _, files in os.walk(out):
        for f in sorted(files):
            if f.endswith(".json") and root != out:
                yield json.load(open(os.path.join(root, f)))


def summarize(out):
    eps = list(_episodes(out))
    ok = [e for e in eps if "infra" not in e]
    sc = [e["score"] for e in ok if isinstance(e.get("score"), (int, float)) and not math.isnan(e["score"])]
    src = {k: sum(e.get("source_counts", {}).get(k, 0) for e in ok) for k in ("model", "model-picked", "code")}
    s = {"episodes": len(eps), "infra_errors": len(eps) - len(ok),
         "mean_score": round(sum(sc) / len(sc), 4) if sc else None,
         "success_rate": round(sum(bool(e.get("success")) for e in ok) / len(ok), 4) if ok else None,
         "harness_errors": sum(bool(e.get("error")) for e in ok),
         "policy_out_tok": sum(e.get("policy_tokens", {}).get("out_tok", 0) for e in ok),
         "policy_calls": sum(e.get("policy_tokens", {}).get("calls", 0) for e in ok),
         "action_source": src}          # ours; not exported
    json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(s), flush=True)
    return s


def _clean(text):
    """Tracebacks name files by their absolute paths on bath; keep only the file name."""
    return re.sub(r'File "[^"]*/([^/"]+)"', r'File "\1"', text or "")


def _render(e):
    """trace.md: every message the model saw and every action, as in HW2's trace.md."""
    lines = [f"# episode {e['instance_id']}", "",
             f"score {e.get('score')} · success {e.get('success')} · steps {e.get('steps')} · "
             f"end {e.get('end')} · error {_clean(e.get('error')) or '-'}", ""]
    prev = []
    by_step = {}
    for c in e.get("calls", []):
        by_step.setdefault(c["step"], []).append(c)
    for k in range(len(e.get("steps_log", [])) + 1):
        for c in by_step.get(k, []):
            msgs = c["messages"]
            same = 0
            while same < min(len(prev), len(msgs)) and prev[same] == msgs[same]:
                same += 1
            lines.append(f"## model call (before action {k + 1})")
            if same:
                lines.append(f"_[first {same} messages identical to the previous call]_")
            for m in msgs[same:]:
                lines += [f"**{m.get('role')}:**", "```", str(m.get("content", "")), "```"]
            lines += ["**reply:**" if "reply" in c else f"**error:** {c.get('error')}", "```",
                      str(c.get("reply", "")), "```", ""]
            prev = msgs + ([{"role": "assistant", "content": c["reply"]}] if "reply" in c else [])
        if k < len(e.get("steps_log", [])):
            s = e["steps_log"][k]
            lines += [f"## action {k + 1}", "```", s["action"], "```", "**observation:**", "```",
                      s["observation"], "```", ""]
    if e.get("harness_stderr"):
        lines += ["## harness stderr (last 2000 chars)", "```", _clean(e["harness_stderr"]), "```"]
    return "\n".join(lines)


def export(a):
    """What the proposer may read: scores.json and, per episode, trace.md + result.json."""
    os.makedirs(a.dest, exist_ok=True)
    rej = os.path.join(a.out, "rejected.json")
    if os.path.exists(rej):
        json.dump(json.load(open(rej)), open(os.path.join(a.dest, "scores.json"), "w"), indent=1)
        return
    s = json.load(open(os.path.join(a.out, "summary.json")))
    per = {}
    for e in _episodes(a.out):
        if "infra" in e:
            continue
        iid = e["instance_id"]
        res = {k: e.get(k) for k in ("instance_id", "score", "success", "steps", "end", "error", "seconds")}
        res["error"] = _clean(res["error"])
        res["model_calls"] = e.get("policy_tokens", {}).get("calls")
        res["model_out_tokens"] = e.get("policy_tokens", {}).get("out_tok")
        per[iid] = res
        d = os.path.join(a.dest, "episodes", _slug(iid))
        os.makedirs(d, exist_ok=True)
        json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
        open(os.path.join(d, "trace.md"), "w").write(_render(e))
    pub = {k: s[k] for k in ("episodes", "mean_score", "success_rate", "harness_errors", "policy_out_tok", "policy_calls")}
    json.dump({**pub, "per_episode": per}, open(os.path.join(a.dest, "scores.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "export", "summarize", "one"])
    ap.add_argument("job", nargs="?")
    ap.add_argument("--cand")
    ap.add_argument("--cell", default="wordle:basic5")
    ap.add_argument("--split", default="search")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seen-split", default="search", help="the split whose traces the proposer reads")
    ap.add_argument("--seen-n", type=int, default=30)
    ap.add_argument("--out")
    ap.add_argument("--dest")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="run one episode first, reject a child that cannot run")
    a = ap.parse_args()
    if a.cmd == "one":
        _one(tuple(json.loads(a.job)))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "summarize":
        summarize(a.out)
    else:
        export(a)


if __name__ == "__main__":
    main()
