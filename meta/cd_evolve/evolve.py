"""Harness-evolution loop for CircuitDecoding, mirroring
meta_harness_official/reference_examples/text_classification/meta_harness.py.

One iteration = propose (fresh `claude -p` session with the meta-harness-cd
skill, filesystem access to this workspace) -> validate (import + mock smoke
test + regime static check) -> benchmark each candidate on the search split
with the fixed 7B model served on Engaging -> score -> update frontier and
evolution_summary.jsonl. Held-out numbers only exist after `--test`, which
freezes the run.

    python evolve.py --run-name open1 --regime open --iterations 1
    python evolve.py --run-name open1 --regime open --iterations 7      # resumes
    python evolve.py --run-name open1 --test                             # one-shot held-out, then frozen

The task model runs on Engaging: submit engaging/serve_7b.sbatch; this loop
reads ~/results/ar_harness/serve_7b.endpoint over ssh and runs the runner
inside that job's allocation (srun --overlap), so nothing heavy runs on the
login node. Candidate files are synced before each benchmark.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import claude_wrapper

WS = Path(__file__).resolve().parent            # this workspace (proposer cwd)
ROOT = WS.parent.parent                          # ar_harness
AGENTS = WS / "agents"
BASELINES = ["bare", "enum15", "symbolic", "facts", "splits", "oracle_hint", "t3_gate"]
SEED_FILES = {f"{b}.py" for b in BASELINES} | {"__init__.py"}
REMOTE = "engaging"
REMOTE_CODE = "~/code/ar_harness"
REMOTE_RESULTS = "~/results/ar_harness/cd_evolve"
SEARCH_SEEDS, HELDOUT_SEEDS, CONCURRENCY = 2, 3, 48
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]
STATE_ONLY_FORBIDDEN = ["available_bits", "expected_bits", "best_query", "split_table_text",
                        "log2", "entropy"]
# Critic denylist (RRSI-style precheck): a harness has no business touching files,
# the network, the interpreter, or anything that smells like the truth table.
CRITIC_DENY = [
    # "truth table" is the task's own vocabulary and appears in legitimate harness text;
    # only code-level access patterns are denied (the View carries no truth anyway).
    (r"truth_bits|\[['\"]truth['\"]\]|\.truth\b|\binst\[|\binst\.", "accesses truth / instance fields"),
    (r"cd_test|circuit_joint_lookup|orig_index|\[['\"]circuits['\"]\]", "references the instance file / truth fields"),
    (r"\bopen\(|\bPath\(|json\.load|\bglob\b|pickle|shelve", "file IO"),
    (r"subprocess|\bos\.|\bsys\.(?!path)|urllib|requests|socket|httpx", "process / network access"),
    (r"\beval\(|\bexec\(|__import__|importlib", "dynamic code execution"),
]
IMPORT_ALLOW = {"numpy", "np", "re", "math", "itertools", "collections", "functools", "typing",
                "dataclasses", "random", "string", "harness_lib", "cd_oracle", "statistics", "heapq"}
SPLIT_FILE = ROOT / "splits/cd_split.json"
ENDPOINT_FILE = "serve_7b.endpoint"          # under ~/results/ar_harness on the cluster
REMOTE_RUNNER = "runner/cd_harness_run.py"   # relative to REMOTE_CODE
DELTA_OVERRIDE = None
BETA0, BETA1 = 0.1, 2.0        # cost rule: dC <= beta0 + beta1 * dS  (dS in uniq@end fraction)
WITHIN_BAND_SAVING = 0.05      # inside the noise band only a >=5% token saving is admissible
CRITIC_MODE = "llm"

# per-run paths, set in main()
RUN = None
LOCAL_MOCK = None      # --local-mock POLICY: benchmark with the scripted task model on this machine
STUB_PROPOSER = False  # --stub-proposer: copy a seed harness under a new name instead of calling claude


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def log(msg):
    print(f"{ts()} {msg}", flush=True)
    with (RUN / "evolve.log").open("a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")


def sh(cmd, timeout=3600, check=False, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                       timeout=timeout, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stderr[-2000:]}")
    return r


def ssh(cmd, timeout=3600):
    return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", REMOTE, cmd], timeout=timeout)


# ----------------------------------------------------------------- remote GPU

def remote_endpoint():
    """-> (job_id, node, port) of a running task-model server, or None.
    The endpoint file is written by engaging/serve_7b.sbatch as node:port:jobid once
    vLLM answers /health, and removed when the job ends."""
    r = ssh(f"cat ~/results/ar_harness/{ENDPOINT_FILE} 2>/dev/null; echo ---; "
            "squeue -u $USER -n serve7b -h -o '%i %T'")
    ep, _, jobs = r.stdout.partition("---")
    ep = ep.strip()
    running = [l.split()[0] for l in jobs.strip().splitlines() if "RUNNING" in l]
    if not running:
        return None
    if ":" not in ep:
        return (running[0], None, None)
    parts = ep.split(":")
    node, port = parts[0], parts[1]
    job = parts[2] if len(parts) > 2 and parts[2] in running else running[0]
    return (job, node, port)


def wait_endpoint(max_wait=6 * 3600):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        ep = remote_endpoint()
        if ep and ep[1]:
            return ep
        state = "no serve7b job" if not ep else "serve7b starting (no endpoint yet)"
        log(f"waiting for the task-model server: {state}")
        time.sleep(60)
    raise RuntimeError("no task-model server after max_wait")


def sync_agent(name):
    sh(["scp", "-q", str(AGENTS / f"{name}.py"), str(WS / "harness_lib.py"),
        f"{REMOTE}:{REMOTE_CODE}/meta/cd_evolve/agents/"], check=True)
    sh(["scp", "-q", str(WS / "harness_lib.py"), f"{REMOTE}:{REMOTE_CODE}/meta/cd_evolve/"], check=True)
    sh(["scp", "-q", str(SPLIT_FILE), f"{REMOTE}:{REMOTE_CODE}/splits/"], check=True)


def benchmark(name, split, seeds, out_local, timeout=5400):
    """Run the fixed task model on `split` with harness `name`; sync traces back."""
    if LOCAL_MOCK:
        out_local.parent.mkdir(parents=True, exist_ok=True)
        r = sh([sys.executable, str(ROOT / "runner/cd_harness_run.py"), "--harness",
                str(AGENTS / f"{name}.py"), "--mock", LOCAL_MOCK, "--split", split, "--seeds",
                str(seeds), "--split-file", str(SPLIT_FILE), "--concurrency", "64",
                "--out", str(out_local)], timeout=timeout, check=True)
        log(f"  mock-benchmark {name}/{split}: {r.stdout.strip().splitlines()[-1]}")
        return r
    job, node, port = wait_endpoint()
    sync_agent(name)
    remote_out = f"{REMOTE_RESULTS}/{RUN.name}/{name}/{split}.jsonl"
    runner = (f"cd {REMOTE_CODE} && mkdir -p $(dirname {remote_out}) && "
              f"conda_env/bin/python {REMOTE_RUNNER} "
              f"--harness meta/cd_evolve/agents/{name}.py --split {split} --seeds {seeds} "
              f"--split-file splits/{SPLIT_FILE.name} "
              f"--concurrency {CONCURRENCY} --base-url http://127.0.0.1:{port}/v1 "
              f"--out {remote_out}")
    cmd = (f"srun --jobid={job} --overlap --ntasks=1 --cpus-per-task=2 --mem=4G "
           f"--quiet bash -lc {json.dumps(runner)}")
    t0 = time.time()
    r = ssh(cmd, timeout=timeout)
    if "done in" not in r.stdout and "to run 0" not in r.stdout:
        # fall back to the login node talking to the compute node directly
        log(f"  srun step failed ({r.returncode}); falling back to login-node runner")
        runner_ln = runner.replace("http://127.0.0.1", f"http://{node}")
        r = ssh(runner_ln, timeout=timeout)
    tail = "\n".join(r.stdout.splitlines()[-3:])
    out_local.parent.mkdir(parents=True, exist_ok=True)
    sh(["rsync", "-az", f"{REMOTE}:{remote_out}", str(out_local)], check=True)
    log(f"  benchmark {name}/{split}: {time.time()-t0:.0f}s  {tail.strip().splitlines()[-1] if tail.strip() else ''}")
    return r


def score(out_local, split, vs=None):
    cmd = [sys.executable, str(ROOT / "runner/cd_score.py"), str(out_local), "--split", split,
           "--split-file", str(SPLIT_FILE)]
    if vs:
        cmd += ["--vs", str(vs)]
    sj = RUN / "scores" / f"{out_local.parent.name}.{split}.json"
    sj.parent.mkdir(parents=True, exist_ok=True)
    r = sh(cmd + ["--out", str(sj)], check=True)
    data = json.loads(sj.read_text())
    return data, r.stdout


# ------------------------------------------------------- RRSI-style regularizers

def noise_delta():
    """delta = z * sqrt(2) * se(uniq@end) of the bare baseline on the search split
    (RRSI calibrate.py, z = 2): the band inside which a score difference is noise."""
    if DELTA_OVERRIDE is not None:
        return float(DELTA_OVERRIDE)
    cache = RUN / "noise.json"
    if cache.exists():
        return json.loads(cache.read_text())["delta"]
    import random as _r
    vals = []
    f = RUN / "traces/bare/search.jsonl"
    if not f.exists():
        return 0.0
    seen = {}
    for line in f.open():
        if line.strip():
            r = json.loads(line)
            seen[(r["key"], r["seed"])] = int(bool((r.get("result") or {}).get("unique_at_end")))
    vals = list(seen.values())
    rng = _r.Random(7)
    n = len(vals)
    boots = [sum(rng.choice(vals) for _ in range(n)) / n for _ in range(2000)]
    import statistics as _st
    se = _st.pstdev(boots)
    cal = {"delta": round(2 * (2 ** 0.5) * se, 4), "se": round(se, 4), "n": n, "S_bare": sum(vals) / n,
           "method": "bootstrap over bare search episodes, z=2, sqrt2 for a difference"}
    cache.write_text(json.dumps(cal, indent=1))
    log(f"  noise band delta={cal['delta']} (se {cal['se']}, n={n})")
    return cal["delta"]


def critic_regex(src):
    hits = []
    for pat, why in CRITIC_DENY:
        if re.search(pat, src):
            hits.append(why)
    for m in re.finditer(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", src, re.M):
        mod = (m.group(1) or m.group(2)).split(".")[0]
        if mod not in IMPORT_ALLOW:
            hits.append(f"import of {mod} not allowed")
    return hits


CRITIC_PROMPT = """You are the critic of a harness-evolution loop (RRSI-style leakage screen).
A proposer model wrote a new HARNESS for a fixed 7B task model playing CircuitDecoding:
each turn the harness may add text after the environment's answer, re-ask the model
(filter_action), rewrite the context the model sees (compose_messages) and add text before
the final guess. It sees only public state (hypothesis space H implied by the gate
inventory, feasible indices, the query/answer history, the model's transcript). The
candidate is evaluated on a fixed set of instances it will also be selected on, so
anti-overfitting review is critical. Regime for this run: {regime}.
  open: the harness may compute anything from the public state, including recommending queries.
  state_only: the harness may present/reorganize state and history, veto repeats or determined
  entries, but must NOT compute information gain, split counts, entropy or rank/recommend queries.

REJECT if ANY of:
1. INSTANCE LEAKAGE: hard-coded truth tables, instance ids/keys, inventory-specific lookup
   tables that only make sense for one instance, or branching on instance identity.
2. TRUTH / IO ACCESS: any attempt to read files, the instance data, the truth, the network,
   or to run code outside the harness functions.
3. DEGENERATE: the mechanism claimed in the docstring/hypothesis is not actually implemented
   (dead code, unused flag, comment-only), or the file is a near-copy of its parent with only
   constants or wording changed while claiming a new mechanism.
4. UNBOUNDED WORK: loops without an exit, re-ask logic that can never accept a reply, or
   context growth that will obviously exceed a 16k-token budget by turn 18.
5. REGIME VIOLATION under state_only: computing or presenting per-query information
   values, split counts, or a recommended query.
Otherwise ACCEPT. Review intent and content, not style; runtime errors are checked
separately. Do not use any tools. Reply with STRICT JSON only:
{{"verdict": "accept"|"reject", "reasons": ["..."], "risk_notes": ["..."]}}

CANDIDATE agents/{name}.py (hypothesis: {hypothesis}):
```python
{src}
```
PARENT agents/{parent}.py:
```python
{parent_src}
```"""


def critic_llm(name, cand, regime):
    src = (AGENTS / f"{name}.py").read_text()
    parent = cand.get("parent") or "bare"
    ppath = AGENTS / f"{parent}.py"
    parent_src = ppath.read_text() if ppath.exists() else "(not found)"
    prompt = CRITIC_PROMPT.format(regime=regime, name=name, hypothesis=cand.get("hypothesis", ""),
                                  src=src[:12000], parent=parent, parent_src=parent_src[:6000])
    os.environ.pop("CLAUDECODE", None)
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    _isolate_memory(f"critic_{name}")
    res = claude_wrapper.run(prompt=prompt, model="opus", allowed_tools=None, cwd=str(WS),
                             log_dir=str(RUN / "critic_sessions"), name=f"critic_{name}",
                             timeout_seconds=600, effort="medium")
    if saved:
        os.environ["ANTHROPIC_API_KEY"] = saved
    _archive_memory(f"critic_{name}")
    text = (getattr(res, "final_text", None) or getattr(res, "result", None)
            or getattr(res, "text", None) or "")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(m.group(0)) if m else {"verdict": "accept", "reasons": ["unparsable critic reply"]}
    except Exception:
        verdict = {"verdict": "accept", "reasons": ["unparsable critic reply"], "raw": text[:500]}
    (RUN / "critic").mkdir(exist_ok=True)
    (RUN / "critic" / f"{name}.json").write_text(json.dumps({"verdict": verdict, "raw": text[:2000]}, indent=1))
    return verdict


def critic(name, cand, regime):
    """-> None if accepted, else the rejection reason string."""
    if CRITIC_MODE == "off":
        return None
    hits = critic_regex((AGENTS / f"{name}.py").read_text())
    if hits:
        return "critic precheck: " + "; ".join(hits)
    if CRITIC_MODE == "llm":
        v = critic_llm(name, cand, regime)
        if str(v.get("verdict", "accept")).lower().startswith("reject"):
            return "critic: " + "; ".join(v.get("reasons", []))[:400]
    return None


def admissible(s, incumbent, s_star, delta):
    """RRSI selection side on our metrics: S = uniq@end, C = tokens per episode."""
    S, C = s["uniq"], (s["tok_in"] or 0) + (s["tok_out"] or 0)
    if incumbent is None:
        return True, "first entry", 0.0, 0.0
    S0, C0 = incumbent["uniq"], incumbent.get("cost") or 1.0
    dS, dC = S - S0, (C - C0) / C0 if C0 else 0.0
    if S < s_star - delta:
        return False, f"below floor: S {S:.3f} < S* {s_star:.3f} - delta {delta:.3f}", dS, dC
    if dS > delta:
        budget = BETA0 + BETA1 * dS
        ok = dC <= budget
        return ok, (f"gain {dS:+.3f} > delta {delta:.3f}; cost {dC:+.3f} "
                    f"{'<=' if ok else '>'} budget {budget:.3f}"), dS, dC
    ok = dC <= -WITHIN_BAND_SAVING
    return ok, (f"gain {dS:+.3f} within delta {delta:.3f}; "
                f"{'token saving ' + format(dC, '+.3f') if ok else 'no token saving (' + format(dC, '+.3f') + ')'}"), dS, dC


# ------------------------------------------------------------------ validation

def mock_check(name):
    for pol in ("sequential", "greedy"):
        out = RUN / "mock" / f"{name}__{pol}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        r = sh([sys.executable, str(ROOT / "runner/cd_harness_run.py"), "--harness",
                str(AGENTS / f"{name}.py"), "--mock", pol, "--split", "search", "--limit", "6",
                "--split-file", str(SPLIT_FILE), "--seeds", "1", "--out", str(out)], timeout=300)
        if r.returncode != 0:
            return f"mock {pol}: {r.stderr[-800:]}"
        errs = [json.loads(l).get("error") for l in out.open() if l.strip()]
        if any(errs):
            return f"mock {pol}: episode error {next(e for e in errs if e)}"
    return None


def validate(cand, regime):
    name = cand["name"]
    path = AGENTS / f"{name}.py"
    if not path.exists():
        return f"{path} missing"
    if f"{name}.py" in SEED_FILES:
        return "name collides with a seed harness"
    src = path.read_text()
    if not re.search(r"^NAME\s*=\s*['\"]%s['\"]" % re.escape(name), src, re.M):
        return "NAME constant must equal the file stem"
    if re.search(r"^\s*from\s+agents\.|^\s*import\s+agents\.", src, re.M):
        return "imports another candidate"
    if regime == "state_only":
        hit = [w for w in STATE_ONLY_FORBIDDEN if w in src]
        if hit:
            return f"state_only regime forbids {hit}"
    return mock_check(name)


# --------------------------------------------------------------- bookkeeping

def read_summary():
    f = RUN / "evolution_summary.jsonl"
    return [json.loads(l) for l in f.open() if l.strip()] if f.exists() else []


def frontier():
    f = RUN / "frontier_val.json"
    return json.loads(f.read_text()) if f.exists() else {"_best": None, "candidates": {}}


def update_frontier(name, s, iteration):
    fr = frontier()
    entry = {"uniq": s["uniq"], "uniq_ci": s["uniq_ci"], "exact": s["exact"], "eff": s["eff"],
             "repeat": s["repeat"], "illegal": s["illegal"], "oracle": s["oracle"],
             "cost": (s["tok_in"] or 0) + (s["tok_out"] or 0),
             "n": s["n"], "iteration": iteration}
    fr["candidates"][name] = entry
    best = fr.get("_best")
    key = lambda e: (round(e["uniq"], 6), round(e["exact"], 6))
    if best is None or key(entry) > key(fr["candidates"][best["agent"]]):
        fr["_best"] = {"agent": name, **entry}
    (RUN / "frontier_val.json").write_text(json.dumps(fr, indent=1))
    return fr


def append_summary(row):
    with (RUN / "evolution_summary.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def summary_row(iteration, cand, s, best_before, regime, status="ok", timing=None):
    row = {"iteration": iteration, "system": cand["name"], "regime": regime, "status": status,
           "parent": cand.get("parent"), "axis": cand.get("axis"),
           "hypothesis": cand.get("hypothesis", ""), "mechanism": cand.get("mechanism", ""),
           "prediction": cand.get("prediction", "")}
    if s:
        row.update({k: (round(s[k], 4) if isinstance(s[k], float) else s[k]) for k in
                    ("n", "uniq", "exact", "eff", "zero", "oracle", "repeat", "illegal", "parsed",
                     "alive_end", "tok_out")})
        row["uniq_ci"] = [round(x, 4) for x in s["uniq_ci"]]
        d = s["uniq"] - (best_before or 0.0)
        row["delta_uniq"] = round(d, 4)
        row["outcome"] = f"uniq {100*s['uniq']:.1f}% ({100*d:+.1f}) exact {100*s['exact']:.1f}%"
    else:
        row["outcome"] = status
    if timing:
        row["timing_s"] = timing
    return row


def render_task_prompt(iteration, regime):
    fr = frontier()
    best = fr.get("_best")
    best_txt = (f"{best['agent']} @ uniq@end {100*best['uniq']:.1f}% / exact {100*best['exact']:.1f}%"
                if best else "none yet")
    return (f"Run iteration {iteration} of CircuitDecoding harness evolution. REGIME: {regime}.\n\n"
            f"Current frontier: {best_txt}.\n\n"
            f"## Run directories (all paths relative to this workspace)\n"
            f"- `logs/{RUN.name}/evolution_summary.jsonl` — every candidate evaluated so far\n"
            f"- `logs/{RUN.name}/frontier_val.json` — frontier\n"
            f"- `logs/{RUN.name}/scores/<name>.search.json` — per-candidate metrics with CIs\n"
            f"- `logs/{RUN.name}/traces/<name>/search.jsonl` — raw traces (use tools/show_trace.py)\n"
            f"- `logs/{RUN.name}/reports/` — post-eval reports (write missing ones first)\n"
            f"- Write pending_eval.json to: `logs/{RUN.name}/pending_eval.json`\n")


def _claude_memory_dir():
    """Claude Code keeps an auto-memory directory per working directory. A proposer session
    must see nothing but the workspace, so that directory is emptied before each session and
    whatever the session wrote is archived with the session log afterwards."""
    slug = "-" + str(WS).strip("/").replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / "memory"


def _isolate_memory(tag):
    mem = _claude_memory_dir()
    if mem.exists() and any(mem.iterdir()):
        dst = RUN / "claude_sessions" / f"{tag}_memory_before"
        dst.mkdir(parents=True, exist_ok=True)
        for f in mem.iterdir():
            shutil.move(str(f), str(dst / f.name))
        log(f"  moved pre-existing proposer memory ({len(list(dst.iterdir()))} files) to {dst.name}")


def _archive_memory(tag):
    mem = _claude_memory_dir()
    if mem.exists() and any(mem.iterdir()):
        dst = RUN / "claude_sessions" / f"{tag}_memory_written"
        dst.mkdir(parents=True, exist_ok=True)
        for f in mem.iterdir():
            shutil.move(str(f), str(dst / f.name))
        log(f"  session {tag} wrote {len(list(dst.iterdir()))} memory file(s); archived to {dst.name}")


def propose(iteration, regime, timeout):
    if STUB_PROPOSER:   # loop plumbing test: three renamed copies of seed harnesses
        cands = []
        for src in ("enum15", "facts", "oracle_hint"):
            name = f"stub{iteration}_{src}"
            body = (AGENTS / f"{src}.py").read_text().replace(f'NAME = "{src}"', f'NAME = "{name}"')
            (AGENTS / f"{name}.py").write_text(body)
            cands.append({"name": name, "file": f"agents/{name}.py", "parent": src,
                          "hypothesis": "stub", "axis": "stub", "mechanism": "copy of " + src})
        (RUN / "pending_eval.json").write_text(json.dumps({"iteration": iteration, "candidates": cands}))
        return cands
    os.environ.pop("CLAUDECODE", None)
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    pending = RUN / "pending_eval.json"
    if pending.exists():
        pending.unlink()
    _isolate_memory(f"iter{iteration}")
    res = claude_wrapper.run(
        prompt=render_task_prompt(iteration, regime), model="opus",
        allowed_tools=PROPOSER_TOOLS, skills=[str(WS / ".claude/skills/meta-harness-cd")],
        cwd=str(WS), log_dir=str(RUN / "claude_sessions"), name=f"iter{iteration}",
        timeout_seconds=timeout, effort="max")
    if saved:
        os.environ["ANTHROPIC_API_KEY"] = saved
    _archive_memory(f"iter{iteration}")
    if res.exit_code != 0:
        log(f"  proposer exit={res.exit_code} {res.stderr[:300] if res.stderr else ''}")
    try:
        res.show()
    except Exception:
        pass
    if not pending.exists():
        return None
    return json.loads(pending.read_text()).get("candidates", [])


# ------------------------------------------------------------------- phases

def phase_baselines(names):
    fr = frontier()
    for b in names:
        if b in fr["candidates"]:
            continue
        log(f"baseline {b}")
        out = RUN / "traces" / b / "search.jsonl"
        benchmark(b, "search", SEARCH_SEEDS, out)
        s, txt = score(out, "search")
        s = s[next(iter(s))]
        update_frontier(b, s, 0)
        append_summary(summary_row(0, {"name": b, "parent": None, "axis": "seed"}, s, None,
                                   "seed", status="seed"))
        log("  " + txt.strip().splitlines()[-1])


def iterate(iteration, regime, args):
    t_iter = time.time()
    best_before = (frontier().get("_best") or {}).get("uniq")
    log(f"=== iteration {iteration} regime={regime} frontier={best_before}")
    t0 = time.time()
    cands = propose(iteration, regime, args.propose_timeout)
    t_prop = time.time() - t0
    if not cands:
        log("  proposer produced no candidates")
        append_summary({"iteration": iteration, "system": None, "regime": regime,
                        "status": "proposer_failed", "timing_s": {"propose": round(t_prop)}})
        return False
    log(f"  proposed {len(cands)} in {t_prop:.0f}s: " + ", ".join(c["name"] for c in cands))
    delta = noise_delta()
    fr0 = frontier()
    inc0 = fr0["candidates"].get((fr0.get("_best") or {}).get("agent")) if fr0.get("_best") else None
    s_star0 = (fr0.get("_best") or {}).get("uniq", 0.0)   # S*: best ACCEPTED score so far (RRSI)
    scored = []                                            # (cand, s, row) judged against inc0
    for c in cands:
        err = validate(c, regime)
        if err:
            log(f"  INVALID {c['name']}: {err[:300]}")
            append_summary(summary_row(iteration, c, None, best_before, regime,
                                       status=f"invalid: {err[:200]}"))
            continue
        rej = critic(c["name"], c, regime)
        if rej:
            log(f"  REJECTED {c['name']}: {rej[:300]}")
            append_summary(summary_row(iteration, c, None, best_before, regime,
                                       status=f"rejected: {rej[:300]}"))
            continue
        t1 = time.time()
        out = RUN / "traces" / c["name"] / "search.jsonl"
        try:
            benchmark(c["name"], "search", SEARCH_SEEDS, out)
            s, txt = score(out, "search", vs=RUN / "traces/bare/search.jsonl")
            s = s[next(iter(s))]
        except Exception as e:
            log(f"  benchmark failed for {c['name']}: {e}")
            append_summary(summary_row(iteration, c, None, best_before, regime,
                                       status=f"benchmark_failed: {str(e)[:200]}"))
            continue
        if s["err"] > 0.05:
            log(f"  {c['name']}: {100*s['err']:.0f}% errored episodes -> invalid")
            append_summary(summary_row(iteration, c, s, best_before, regime, status="too_many_errors"))
            continue
        ok, why, dS, dC = admissible(s, inc0, s_star0, delta)
        row = summary_row(iteration, c, s, best_before, regime,
                          timing={"propose": round(t_prop), "bench": round(time.time() - t1)})
        row.update({"admissible": ok, "selection": why, "delta_S": round(dS, 4), "delta_C": round(dC, 4),
                    "noise_delta": delta})
        scored.append((c, s, row, ok))
        log("  " + "\n  ".join(txt.strip().splitlines()[-2:]))
        log(f"  selection[{c['name']}]: {'ADMISSIBLE' if ok else 'not admissible'} -- {why}")
    # round winner: the admissible candidate with the highest uniq@end (ties: exact); siblings
    # never price each other, and only the winner becomes the incumbent
    adm = [(c, s, row) for c, s, row, ok in scored if ok]
    winner = max(adm, key=lambda t: (t[1]["uniq"], t[1]["exact"]), default=None)
    fr = frontier()
    for c, s, row, ok in scored:
        is_winner = winner is not None and c["name"] == winner[0]["name"]
        row["round_winner"] = is_winner
        if is_winner:
            update_frontier(c["name"], s, iteration)
        else:
            fr = frontier()
            fr["candidates"][c["name"]] = {"uniq": s["uniq"], "exact": s["exact"], "eff": s["eff"],
                                           "cost": (s["tok_in"] or 0) + (s["tok_out"] or 0),
                                           "iteration": iteration, "admissible": ok}
            (RUN / "frontier_val.json").write_text(json.dumps(fr, indent=1))
        append_summary(row)
    best_after = (frontier().get("_best") or {})
    log(f"  iteration {iteration} done in {time.time()-t_iter:.0f}s; frontier now "
        f"{best_after.get('agent')} @ {100*best_after.get('uniq', 0):.1f}%")
    return True


def finalize(args):
    frozen = RUN / "finalized.json"
    fr = frontier()
    if not fr.get("_best"):
        sys.exit("no frontier yet")
    if frozen.exists() and json.loads(frozen.read_text()).get("status") == "complete":
        log("already finalized"); return
    ranked = sorted(fr["candidates"].items(), key=lambda kv: (kv[1]["uniq"], kv[1]["exact"]), reverse=True)
    chosen = [n for n, _ in ranked if n not in BASELINES][:3]
    systems = sorted(set(["bare", "enum15"]) | set(chosen) | {fr["_best"]["agent"]})
    frozen.write_text(json.dumps({"status": "in_progress", "systems": systems,
                                  "selected_before_test": fr["_best"]["agent"],
                                  "started": datetime.now().isoformat()}, indent=1))
    log(f"FINALIZE (held-out x{HELDOUT_SEEDS} seeds): {systems}")
    hold_dir = ROOT / "results/cd_heldout" / RUN.name
    hold_dir.mkdir(parents=True, exist_ok=True)
    for name in systems:
        out = hold_dir / f"{name}.jsonl"
        benchmark(name, "heldout", HELDOUT_SEEDS, out)
    r = sh([sys.executable, str(ROOT / "runner/cd_score.py")] + [str(hold_dir / f"{n}.jsonl") for n in systems]
           + ["--split", "heldout", "--split-file", str(SPLIT_FILE), "--vs", str(hold_dir / "enum15.jsonl"),
              "--out", str(hold_dir / "scores.json")])
    print(r.stdout)
    (hold_dir / "table.txt").write_text(r.stdout)
    frozen.write_text(json.dumps({**json.loads(frozen.read_text()), "status": "complete",
                                  "completed": datetime.now().isoformat()}, indent=1))
    log("finalized; this run name refuses further evolution")


def main():
    global RUN
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--regime", choices=["open", "state_only"], default=None)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--propose-timeout", type=int, default=2400)
    p.add_argument("--baselines", default="bare,enum15",
                   help="seed harnesses to evaluate on the search split first")
    p.add_argument("--test", action="store_true", help="freeze and evaluate on held-out once")
    p.add_argument("--split-file", default=str(ROOT / "splits/cd_split.json"))
    p.add_argument("--endpoint-file", default="serve_7b.endpoint",
                   help="endpoint file under ~/results/ar_harness written by serve_7b.sbatch")
    p.add_argument("--remote-runner", default="runner/cd_harness_run.py")
    p.add_argument("--delta", type=float, default=None, help="noise band override (uniq@end fraction)")
    p.add_argument("--beta0", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=2.0)
    p.add_argument("--critic", choices=["llm", "regex", "off"], default="llm")
    p.add_argument("--local-mock", default=None, help="CPU test: benchmark with this scripted policy")
    p.add_argument("--stub-proposer", action="store_true", help="CPU test: no claude call")
    args = p.parse_args()
    global LOCAL_MOCK, STUB_PROPOSER, SPLIT_FILE, DELTA_OVERRIDE, BETA0, BETA1, CRITIC_MODE
    global ENDPOINT_FILE, REMOTE_RUNNER
    LOCAL_MOCK, STUB_PROPOSER = args.local_mock, args.stub_proposer
    ENDPOINT_FILE, REMOTE_RUNNER = args.endpoint_file, args.remote_runner
    SPLIT_FILE, DELTA_OVERRIDE, BETA0, BETA1, CRITIC_MODE = (Path(args.split_file).resolve(), args.delta,
                                                            args.beta0, args.beta1, args.critic)
    RUN = WS / "logs" / args.run_name
    for d in ("reports", "traces", "scores", "claude_sessions"):
        (RUN / d).mkdir(parents=True, exist_ok=True)
    if args.test:
        finalize(args); return
    frozen = RUN / "finalized.json"
    if frozen.exists() and json.loads(frozen.read_text()).get("status") == "complete":
        sys.exit(f"run {args.run_name} is finalized; use a new --run-name")
    regime = args.regime or (read_summary() or [{}])[-1].get("regime") or "open"
    (RUN / "REGIME").write_text(regime)
    phase_baselines([b for b in args.baselines.split(",") if b])
    start = max([r.get("iteration", 0) for r in read_summary()] + [0]) + 1
    fails = 0
    for i in range(start, start + args.iterations):
        ok = iterate(i, regime, args)
        fails = 0 if ok else fails + 1
        if fails >= 2:
            log("two consecutive proposer failures; stopping"); break
    log(f"done. finalize with: python evolve.py --run-name {args.run_name} --test")


if __name__ == "__main__":
    main()
