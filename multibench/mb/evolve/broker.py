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
Runaway output is the agent's failure, judged by tokens: a reply that fills MAX_TOKENS ends the episode
with score 0 (end "runaway"). Running past EPISODE_TIMEOUT (wall clock, load-dependent) is infrastructure.

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
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

from mb.benches import get
from mb.screen import _path, _slug

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "child.py")
LIB = os.path.join(HERE, "ws_template")          # harness_lib.py lives here
CHAT_CAP = 8                  # model calls per environment step, on average over the episode
MAX_TOKENS = 8192             # cap on every reply (2026-11-26); harness_lib's chat() default is the same
EPISODE_TIMEOUT = 14400         # wall clock (4 h); a timeout is infrastructure and is rerun. BabyAI Open episodes run up to 576 steps (8-21 min on an idle GPU)
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


class Diverged(Exception):
    """Replaying an episode's journal did not reproduce it (the harness or the environment is not
    deterministic): the journal is dropped and the episode restarts from its first step."""


def _h(msgs):
    import hashlib
    return hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def episode(cand, bench, sub, inst, url, model, journal=None):
    """journal: a file recording every model reply and every environment step as they happen. If an earlier
    run of this episode was cut off (preempted, killed, moved to another GPU), its journal is replayed first:
    the harness runs again, its model calls are answered from the journal instead of the server, and every
    action is applied to a fresh copy of the environment, whose observations must match the journal. Then the
    episode continues live from where it stopped (user, 2026-11-27)."""
    from collections import deque
    from mb.llm import Counter, make_chat
    b = get(bench)
    if b.responder:
        journal = None        # the environment itself calls a model: a replay could not reproduce it
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
    import hashlib
    head = {"v": 1, "harness": hashlib.sha1(open(os.path.join(cand, "harness.py"), "rb").read()).hexdigest(),
            "instance": str(inst)}
    replay, jf = deque(), None
    if journal:
        try:
            lines = [json.loads(x) for x in open(journal) if x.strip()]
            if lines and lines[0] == head:
                replay = deque(lines[1:])
        except (OSError, ValueError):
            replay = deque()
        if not replay:
            jf = open(journal, "w")
            jf.write(json.dumps(head) + "\n")
            jf.flush()
    replayed = len(replay)

    def jot(e):
        nonlocal jf
        if journal is None:
            return
        if jf is None:
            jf = open(journal, "a")
        jf.write(json.dumps(e) + "\n")
        jf.flush()

    send({"instructions": env.instructions, "observation": env.observation, "max_steps": env.max_steps,
          "reads_full_reply": env.reads_full_reply, "admissible": env.admissible(), "seed": str(inst)})
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
                # one cap for every harness and method: a reply longer than MAX_TOKENS is runaway output
                cap = min(m.get("max_tokens") or MAX_TOKENS, MAX_TOKENS)
                if replay:
                    e = replay.popleft()
                    if e.get("t") not in ("chat", "chat_err") or e.get("h") != _h(msgs):
                        raise Diverged(f"model call {len(calls)}: the harness asked something else")
                    if e["t"] == "chat_err":
                        calls.append({"step": env.steps, "messages": msgs, "error": e["err"]})
                        send({"error": e["msg"]})
                        continue
                    reply, last = e["reply"], e.get("last", {})
                else:
                    try:
                        reply = chat(msgs, temperature=m.get("temperature", 0.7), max_tokens=cap,
                                     top_p=m.get("top_p"), stop=m.get("stop"))
                    except urllib.error.HTTPError as e:      # e.g. 400: the prompt outgrew the context
                        msg = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                        jot({"t": "chat_err", "h": _h(msgs), "err": f"HTTP {e.code}", "msg": msg})
                        calls.append({"step": env.steps, "messages": msgs, "error": f"HTTP {e.code}"})
                        send({"error": msg})
                        continue
                    except Exception as e:                   # server unreachable: not the harness's doing
                        raise Infra(repr(e))
                    last = dict(getattr(pc, "last", {}) or {})
                    jot({"t": "chat", "h": _h(msgs), "reply": reply, "last": last})
                calls.append({"step": env.steps, "messages": msgs, "reply": reply, **last})
                if cap == MAX_TOKENS and last.get("finish") == "length":
                    # runaway output: the reply filled the whole cap. The agent's failure (as the
                    # official baselines count output that runs past the context), not infrastructure.
                    end, error = "runaway", f"a reply filled the {MAX_TOKENS}-token cap"
                    break
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
                if replay:
                    e = replay.popleft()
                    if e.get("t") != "act" or e.get("a") != action or e.get("o") != obs:
                        raise Diverged(f"step {env.steps}: a different action or observation than recorded")
                else:
                    jot({"t": "act", "a": action, "o": obs})
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
        if jf:
            jf.close()
        errf.seek(0)
        stderr = errf.read()[-2000:]
        errf.close()
    if end == "end" and not env.done:
        error = "run_episode returned before the episode ended"
    r = {**env.result(), "steps_log": steps, "calls": calls, "end": end, "error": error,
         "harness_stderr": stderr, "policy_tokens": pc.as_dict(), "responder_tokens": rc.as_dict()}
    r["source_counts"] = {k: sum(s["source"] == k for s in steps) for k in ("model", "model-picked", "code")}
    if replayed:
        r["resumed"] = {"journal_events": replayed}
    if end == "runaway":
        r.update(score=0.0, success=False)
    return r


def _write(p, r):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(r, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)


def _one(job):
    cand, bench, sub, inst, url, out = job
    t0 = time.time()
    p = _path(out, bench, sub, inst)
    journal = p + ".journal"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        try:
            r = episode(cand, bench, sub, inst, url, os.environ["POLICY_MODEL"], journal)
        except Diverged as d:
            print(f"journal of {inst} not reproducible ({d}); restarting it", file=sys.stderr, flush=True)
            os.remove(journal)
            r = episode(cand, bench, sub, inst, url, os.environ["POLICY_MODEL"], journal)
            r["resumed"] = {"diverged": str(d)}
    except Infra as e:          # the journal stays: the rerun continues from it
        r = {"bench": bench, "subtask": sub, "instance": inst, "score": float("nan"), "infra": repr(e)}
    r["seconds"] = round(time.time() - t0, 1)
    _write(p, r)
    if "infra" not in r and os.path.exists(journal):
        os.remove(journal)


def timeout_record(bench, sub, inst):
    """An episode killed at EPISODE_TIMEOUT: every field export and summarize read, scored 0."""
    return {"bench": bench, "subtask": sub, "instance": inst, "instance_id": inst, "score": 0.0,
            "success": False, "steps": 0, "end": "timeout", "steps_log": [], "calls": [],
            "harness_stderr": "", "policy_tokens": {}, "responder_tokens": {}, "seconds": EPISODE_TIMEOUT,
            "error": f"episode exceeded {EPISODE_TIMEOUT}s (killed; runaway output counts as failure)"}


# Episode claims, so that idle workers can help with a long evaluation (user, 2026-11-27: "有空卡的时候就把
#长的分散到各个上面"). Before an episode starts it takes <record>.claim (an atomic mkdir on the shared
# filesystem); a running broker touches its claims every minute, a claim untouched for CLAIM_STALE seconds
# (its worker died or was preempted) is taken over. Who plays an episode never changes its record.
CLAIM_STALE = 900


LOAD_DIR = os.path.expanduser("~/results/mb/load")   # <slurm job>/<pid>.json: episodes each broker is running
REBALANCE_EVERY = 120          # seconds between load checks
RELEASE_HOLD = 300             # a handed-back episode is not taken back by the same broker for this long


def priority(out):
    """Scheduling order when GPUs are short (user, 2026-11-27): 0 our method, 1 its ablations, 2 the rest.
    It decides only who gets the GPUs first, never how anything is scored."""
    o = str(out).replace(os.sep, "/")
    if "/evo_b/ours3/" in o:
        return 0
    if "/evo_b/ours3_" in o:
        return 1
    return 2


LOW_PRIORITY_CAP = 2          # episodes the whole of a priority-2 evaluation may run while ours are running


def _write_load(n, out=""):
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return
    d = os.path.join(LOAD_DIR, job)
    os.makedirs(d, exist_ok=True)
    _write(os.path.join(d, f"{os.getpid()}.json"), {"n": n, "t": time.time(), "out": str(out),
                                                    "prio": priority(out)})


def _all_loads():
    rows = []
    for f in __import__("glob").glob(os.path.join(LOAD_DIR, "*", "*.json")):
        try:
            x = json.load(open(f))
            if time.time() - x["t"] < 90:
                rows.append((f, x))
        except (OSError, ValueError, KeyError):
            pass
    return rows


def low_priority_allowance(out, mine):
    """How many episodes this broker may run: no limit unless it is priority 2 and an evaluation of ours is
    running, then LOW_PRIORITY_CAP for the whole evaluation (all its brokers together)."""
    if priority(out) < 2:
        return None
    rows = _all_loads()
    if not any(x.get("prio", 2) < 2 and x["n"] > 0 for _, x in rows):
        return None
    me = f"{os.getpid()}.json"
    others = sum(x["n"] for f, x in rows if x.get("out") == str(out) and not f.endswith(os.sep + me))
    return max(0, LOW_PRIORITY_CAP - others)


def _drop_load():
    try:
        os.remove(os.path.join(LOAD_DIR, os.environ.get("SLURM_JOB_ID", "-"), f"{os.getpid()}.json"))
    except OSError:
        pass


def gpu_loads():
    """{slurm job of every running worker: episodes its brokers are running} (fresh load files only)."""
    r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "RUNNING", "-o", "%i %j"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return {}
    jobs = [l.split()[0] for l in r.stdout.splitlines() if len(l.split()) > 1 and l.split()[1].startswith("mb_worker")]
    gpu_loads.names = {l.split()[0]: l.split()[1][3:] for l in r.stdout.splitlines() if len(l.split()) > 1}
    loads = {}
    for j in jobs:
        n = 0
        for f in (os.listdir(os.path.join(LOAD_DIR, j)) if os.path.isdir(os.path.join(LOAD_DIR, j)) else []):
            try:
                x = json.load(open(os.path.join(LOAD_DIR, j, f)))
                if time.time() - x["t"] < 90:
                    n += x["n"]
            except (OSError, ValueError, KeyError):
                pass
        loads[j] = n
    return loads


def rebalance(running):
    """Hand back episodes this broker runs on a GPU busier than the others while some GPU sits idle: the
    episode is stopped, its journal stays, and an idle worker resumes it (user, 2026-11-27: "自动再平衡")."""
    me = os.environ.get("SLURM_JOB_ID")
    if not me or len(running) < 2:
        return
    try:
        loads = gpu_loads()
    except Exception:
        return
    if me not in loads or len(loads) < 2:
        return
    helping = set(idle_workers(fresh=180, max_busy=99))      # workers that say they take others' episodes
    idle = [j for j, n in loads.items() if n == 0 and j != me and gpu_loads.names.get(j) in helping]
    fair = math.ceil(sum(loads.values()) / len(loads))
    extra = min(loads[me] - fair, len(running) - 1, 2 * len(idle))
    if not idle or extra <= 0:
        return
    with _HELD_LOCK:
        cands = [i for i in running if i in _PROCS]
    for inst in cands[-extra:]:                 # the latest started have the least to replay
        _RELEASED[inst] = time.time()
        with _HELD_LOCK:
            proc = _PROCS.get(inst)
        if proc:
            proc.terminate()
    print(f"rebalance: handed back {min(extra, len(cands))} episodes (this GPU {loads[me]}, fair {fair}, "
          f"idle GPUs {len(idle)})", flush=True)


OWNER_GRACE = 150        # seconds an episode waits for a helper before the owner plays it


def idle_workers(me=None, fresh=180, max_busy=6):
    """Other workers that are alive and have free slots, from the heartbeat each worker writes into its own
    mailbox clone every poll (~/mailbox*/heartbeat/<name>.json, the freshest copy per name)."""
    import glob
    best = {}
    for f in glob.glob(os.path.expanduser("~/mailbox*/heartbeat/worker*.json")):
        try:
            hb = json.load(open(f))
            t = time.mktime(time.strptime(hb["time"], "%Y-%m-%dT%H:%M:%S"))
        except (OSError, ValueError, KeyError):
            continue
        name = os.path.basename(f)[:-5]
        if name not in best or t > best[name][0]:
            best[name] = (t, hb.get("busy", []), hb.get("helps"))   # only workers that say they help
    return [n for n, (t, busy, helps) in best.items()
            if n != me and helps is True and time.time() - t < fresh and len(busy) < max_busy]


def owner_share(n_jobs, me=None):
    """The owner plays its even share of the episodes and leaves the rest unclaimed for the idle workers,
    which check every minute; with no idle worker it plays them all."""
    try:
        k = 1 + len(idle_workers(me))
    except Exception:
        k = 1
    return max(1, math.ceil(n_jobs / k))
_HELD = set()
_HELD_LOCK = threading.Lock()


def _claim_ep(p):
    d = p + ".claim"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for _ in range(2):
        try:
            os.mkdir(d)
            with _HELD_LOCK:
                _HELD.add(d)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(d) < CLAIM_STALE:
                    return False
                tmp = f"{d}.stale.{os.getpid()}.{threading.get_ident()}"
                os.rename(d, tmp)              # only one of several takers succeeds
                os.rmdir(tmp)
            except OSError:
                return False
    return False


def _release_ep(p):
    d = p + ".claim"
    with _HELD_LOCK:
        _HELD.discard(d)
    try:
        os.rmdir(d)
    except OSError:
        pass


def _claimed_by_other(p):
    d = p + ".claim"
    with _HELD_LOCK:
        if d in _HELD:
            return False
    try:
        return time.time() - os.path.getmtime(d) < CLAIM_STALE
    except OSError:
        return False


def _heartbeat(stop):
    while not stop.wait(60):
        with _HELD_LOCK:
            held = list(_HELD)
        for d in held:
            try:
                os.utime(d)
            except OSError:
                pass


def _finished(p):
    """A record that is not an infrastructure failure (those rerun)."""
    try:
        return "infra" not in json.load(open(p))
    except (OSError, ValueError):
        return False


def _one_subprocess(job):
    cand, bench, sub, inst, url, out = job
    p = _path(out, bench, sub, inst)
    if _finished(p) or not _claim_ep(p):    # done, or another worker is playing it
        return inst, None
    try:
        return _one_subprocess_claimed(job)
    finally:
        _release_ep(p)


_PROCS = {}                    # instance -> its running "broker one" process (for rebalancing)
_RELEASED = {}                 # instance -> when this broker handed it back for an idle GPU


def _one_subprocess_claimed(job):
    cand, bench, sub, inst, url, out = job
    proc = subprocess.Popen([sys.executable, "-m", "mb.evolve.broker", "one", json.dumps(job)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with _HELD_LOCK:
        _PROCS[inst] = proc
    try:
        _, stderr = proc.communicate(timeout=EPISODE_TIMEOUT)
        if proc.returncode == 0:
            return inst, json.load(open(_path(out, bench, sub, inst))).get("score")
        err = f"exit {proc.returncode}: {(stderr or '').strip()[-400:]}"
        if inst in _RELEASED:
            err = "handed to an idle GPU (resumes from its journal)"
    except subprocess.TimeoutExpired:
        proc.kill()
        # wall-clock time depends on the server's load, not only on the agent: infrastructure, rerun.
        # Runaway output is judged by tokens instead (a reply that fills MAX_TOKENS ends the episode).
        err = f"timeout after {EPISODE_TIMEOUT}s (rerun)"
    finally:
        with _HELD_LOCK:
            _PROCS.pop(inst, None)
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
        base = ""
        for prefix in ("leak_baseline_", "leakbase_"):    # the second: runs before 2026-11-27
            bp = os.path.join(os.path.dirname(os.path.abspath(a.cand)), prefix + os.path.basename(
                os.path.abspath(a.cand)), "harness.py")
            if os.path.exists(bp):
                base = open(bp).read()
                break
        lk = leaks(src, instances(a.cell, a.seen_split, a.seen_n), instances(a.cell, "selection", None), base)
        bad += [lk] if lk else []
    os.makedirs(a.out, exist_ok=True)
    if bad:
        json.dump({"rejected": bad}, open(os.path.join(a.out, "rejected.json"), "w"), indent=1)
        print("rejected:", bad, flush=True)
        return
    urls = [u.strip().rstrip("/") for u in os.environ["POLICY_BASE_URLS"].split(",") if u.strip()]
    bench, sub = a.cell.split(":", 1)
    jobs, done = [], 0
    chosen = a.instances.split(",") if a.instances else instances(a.cell, a.split, a.n)
    if a.instances:                       # only ids that really are in this split
        valid = set(instances(a.cell, a.split, None))
        assert all(i in valid for i in chosen), "instance not in split"
    for inst in chosen:
        p = _path(a.out, bench, sub, inst)
        if os.path.exists(p) and "infra" not in json.load(open(p)):
            done += 1
            continue
        jobs.append((os.path.abspath(a.cand), bench, sub, inst, urls[len(jobs) % len(urls)], a.out))
    print(f"{done} on disk, {len(jobs)} to run", flush=True)
    marker = os.path.join(a.out, ".sharded")
    if a.assist:
        if not os.path.exists(marker) or not any(os.path.exists(_path(a.out, bench, sub, i)) for i in chosen):
            return                            # owner not sharding, or its smoke episode not done yet
    else:
        open(marker, "w").close()
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(stop,), daemon=True).start()
    try:
        _run_jobs(a, jobs, done, bench, sub)
    finally:
        stop.set()


def _run_jobs(a, jobs, done, bench, sub):
    if a.smoke and jobs and not done:
        # HW2's smoke test (DGM): one episode first; a child that cannot run at all is rejected
        # before the rest. The reward is ignored, one episode says nothing about quality.
        _one_subprocess(jobs[0])
        p0 = _path(a.out, bench, sub, jobs[0][3])
        e = json.load(open(p0)) if _finished(p0) else {}   # {}: another worker holds it (stale claim taken)
        # a harness may decide actions in code, even without calling the model (user, 2026-11-26);
        # only a harness that cannot run at all is rejected here
        why = "crashes: " + (e.get("error") or "")[-300:] if e.get("end") == "crash" else ""
        if why:
            json.dump({"rejected": [why]}, open(os.path.join(a.out, "rejected.json"), "w"), indent=1)
            print("rejected:", why, flush=True)
            return
        jobs = jobs[1:]
    # The owner, and each helper, starts only its even share of the unclaimed episodes (an episode is claimed when
    # it starts; with 48 threads the owner once claimed all 20 at once and nobody could help). After OWNER_GRACE
    # seconds anyone still running takes whatever is unclaimed (no helper came, or a helper died); claims keep two
    # workers from playing the same episode.
    def rec(j):
        return _path(a.out, bench, sub, j[3])
    left_n = sum(1 for j in jobs if not _finished(rec(j)) and not _claimed_by_other(rec(j)))
    if a.assist:
        try:
            k = max(1, len(idle_workers()))
        except Exception:
            k = 1
        share = max(1, min(a.workers, math.ceil(left_n / k)))
    else:
        share = min(a.workers, owner_share(max(1, left_n), os.environ.get("MB_WORKER_NAME")))
    print(f"{'helper' if a.assist else 'owner'} starts {share} of {left_n} unclaimed episodes", flush=True)
    t0, running, tries, last_rb, last_pr, allow = time.time(), {}, {}, time.time(), 0.0, None
    with ThreadPoolExecutor(a.workers) as ex:
        while True:
            running = {i: f for i, f in running.items() if not f.done()}
            todo = [j for j in jobs if not _finished(rec(j))]
            free = [j for j in todo if j[3] not in running and not _claimed_by_other(rec(j))   # a stale claim is free
                    and tries.get(j[3], 0) < 4 and time.time() - _RELEASED.get(j[3], 0) > RELEASE_HOLD]
            others = [j for j in todo if _claimed_by_other(rec(j))
                      or time.time() - _RELEASED.get(j[3], 0) <= RELEASE_HOLD]   # handed back, not yet taken
            if not running and not free and (a.assist or not others):
                break                             # done (a helper does not wait for the others), or only
                                                  # infrastructure failures left (rerun on resume)
            limit = share if time.time() - t0 < OWNER_GRACE else a.workers
            if time.time() - last_pr > 60:
                last_pr = time.time()
                try:
                    allow = low_priority_allowance(a.out, len(running))
                except Exception:
                    allow = None
                if allow is not None and len(running) > allow:      # make room for ours: hand back (journal kept)
                    with _HELD_LOCK:
                        extra = [i for i in running if i in _PROCS][allow:]
                    for inst in extra:
                        _RELEASED[inst] = time.time()
                        with _HELD_LOCK:
                            proc = _PROCS.get(inst)
                        if proc:
                            proc.terminate()
                    print(f"priority: paused {len(extra)} episodes while ours run", flush=True)
            if allow is not None:
                limit = min(limit, allow)
                if a.assist and not running and limit == 0:
                    break                         # a helper of a paused evaluation steps aside
            for j in free[:max(0, limit - len(running))]:
                if j[3] not in _RELEASED:         # a handed-back episode did not fail: no try used
                    tries[j[3]] = tries.get(j[3], 0) + 1
                running[j[3]] = ex.submit(_one_subprocess, j)
            _write_load(len(running), a.out)
            if time.time() - last_rb > REBALANCE_EVERY:
                last_rb = time.time()
                rebalance(list(running))
            time.sleep(15)
    _drop_load()
    if a.assist:                              # a helper: the owner summarizes
        return
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
    ap.add_argument("--instances", default="", help="comma-separated instance ids to play (default: first --n)")
    ap.add_argument("--smoke", action="store_true", help="run one episode first, reject a child that cannot run")
    ap.add_argument("--assist", action="store_true", help="help another worker's run: play unclaimed episodes "
                    "of it, never summarize")
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
