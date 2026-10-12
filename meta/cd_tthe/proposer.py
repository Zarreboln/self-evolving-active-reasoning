"""Proposers and judge for the CircuitDecoding TTHE loop.

Backends:
  claude   a fresh `claude -p` session per role (subscription), via meta/cd_evolve/claude_wrapper.py
  solver   the same, but Claude Code's model calls are routed to the FROZEN task model
           (ANTHROPIC_BASE_URL -> vLLM's Anthropic-compatible endpoint): TTHE's own setting,
           where solver, proposers and judge are one frozen backbone
  stub     no model: copies the parent with a cosmetic edit; judge picks by signals (CPU tests)

Everything a proposer or judge sees is label-free: traces (prompts, replies, answer bits),
the signals of signals.py, the candidate source files. Gold never enters.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cd_evolve"))
import claude_wrapper  # noqa: E402  (copied verbatim from Meta-Harness, MIT)

AGENTS = HERE / "agents"
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]

ROLES = [
    ("conservative_repair",
     "CONSERVATIVE REPAIR ROLE: keep every parent behavior the traces show working. Make the smallest "
     "general change that removes a concrete, trace-visible failure (a crashed episode, a wasted query, "
     "an unparsable final guess). Do not add machinery the traces do not call for."),
    ("independent_exploration",
     "INDEPENDENT EXPLORATION ROLE: test a genuinely different causal hypothesis about why the frozen "
     "model wastes queries or mis-assembles the final table. Change what the model is shown, when it is "
     "re-asked, or how the final guess is assembled -- one mechanism, implemented simply. Do not copy the "
     "other branch's mechanism because it is the current default."),
    ("adversarial_audit",
     "ADVERSARIAL AUDIT AND SYNTHESIS ROLE: start from disagreements between the branches' traces. Find "
     "where one branch's harness and another's diverge on the same instance, decide from the evidence "
     "which behavior is actually supported, and synthesize only the independently supported strengths. "
     "Never equate 'ran without error' with 'pinned the circuits down'."),
]


# ----------------------------------------------------------------------------- traces

def signal_line(r):
    s = r.get("signals") or {}
    f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    return (f"status={r.get('status')} health={f(s.get('health'))} queries={f(s.get('n_queries'))} "
            f"uniq_end={f(s.get('uniq_end'))} alive_end={f(s.get('n_alive_end'))} zero_info={f(s.get('zero_info'))} "
            f"repeat={f(s.get('repeat'))} eff={f(s.get('eff'))} calls={r.get('solver_calls')} "
            f"tokens_out={(r.get('tokens') or {}).get('out')}")


def render_trace(r):
    """One episode as markdown: what the harness did, what the model said, what the environment
    answered. The truth table and exact-match are deliberately absent."""
    L = [f"# Trace -- harness `{r['harness']}` -- instance {r['index']}",
         f"status: {r.get('status')}   {signal_line(r)}"]
    if r.get("error"):
        L.append(f"## ERROR\n```\n{r['error'][-1200:]}\n```")
    L.append("## Queries and answers (in order)")
    L += [f"- {c}({a}, {b}, {x}) = {bit}" for (c, a, b, x), bit in r.get("history", [])]
    L.append("## Model calls (what the harness sent, what the model replied)")
    for i, st in enumerate(r.get("trace", []), 1):
        if st.get("step") != "llm":
            continue
        msgs = st["messages"]
        shown = msgs[-1]["content"] if msgs else ""
        if len(shown) > 1500:
            shown = shown[:700] + f"\n... [{len(shown) - 1400} chars elided] ...\n" + shown[-700:]
        L.append(f"\n### call {i} -- last user message sent:\n{shown}\n\nMODEL REPLY:\n{str(st.get('response'))[:1200]}")
    L.append(f"\n## Final text returned by solve()\n{r.get('guess_text', '')[-400:]}")
    return "\n".join(L)


def write_traces(results, trace_dir):
    trace_dir = Path(trace_dir); trace_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        (trace_dir / f"{r['harness']}__i{r['index']}.md").write_text(render_trace(r))
        (trace_dir / f"{r['harness']}__i{r['index']}.json").write_text(json.dumps(r))


def summary_table(results_by_name):
    rows = []
    for name, rs in results_by_name.items():
        sig = [r.get("signals") or {} for r in rs]
        m = lambda k: (sum(1 for s in sig if s.get(k)) / len(sig)) if sig else 0.0
        mean = lambda k: (sum(s[k] for s in sig if s.get(k) is not None) / max(1, sum(1 for s in sig if s.get(k) is not None)))
        toks = sum((r.get("tokens") or {}).get("out", 0) for r in rs) / max(1, len(rs))
        rows.append(f"{name:<34} n={len(rs):<3} health={100*m('health'):5.1f}%  uniq_end={100*m('uniq_end'):5.1f}%  "
                    f"alive_end={mean('n_alive_end'):6.2f}  zero_info={100*mean('zero_info'):5.1f}%  repeat={100*mean('repeat'):5.1f}%  "
                    f"eff={mean('eff'):.3f}  tokens_out/ep={toks:7.0f}  crashes={sum(1 for r in rs if r.get('status') != 'ok')}")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- prompts

def interface_doc():
    return (HERE / "harness_base.py").read_text()


def check_script(run_dir, tag, name, indices, solver_cfg, arm, timeout):
    """A script the proposer can run: its candidate on the batch instances, label-free output."""
    p = Path(run_dir) / f"check_{tag}.py"
    p.write_text(f'''import json, subprocess, sys, os
from pathlib import Path
HERE = Path({str(HERE)!r})
name = {name!r}; idx = {list(indices)!r}; solver = {solver_cfg!r}; arm = {arm!r}
out_dir = Path({str(run_dir)!r}) / "checks" / {tag!r}; out_dir.mkdir(parents=True, exist_ok=True)
for i in idx:
    task = out_dir / f"task_{{i}}.json"; res = out_dir / f"res_{{i}}.json"
    task.write_text(json.dumps({{"harness": name, "index": i, "output": str(res), "solver": solver, "arm": arm}}))
    try:
        subprocess.run([sys.executable, str(HERE / "solve_worker.py"), "--task", str(task)], timeout={int(timeout)}, check=False)
    except subprocess.TimeoutExpired:
        print(f"instance {{i}}: TIMEOUT after {int(timeout)}s"); continue
    if not res.exists():
        print(f"instance {{i}}: no result (worker died)"); continue
    r = json.loads(res.read_text()); s = r.get("signals") or {{}}
    print(f"instance {{i}}: status={{r['status']}} health={{s.get('health')}} queries={{s.get('n_queries')}} "
          f"uniq_end={{s.get('uniq_end')}} alive_end={{s.get('n_alive_end')}} zero_info={{s.get('zero_info')}} "
          f"repeat={{s.get('repeat')}} eff={{s.get('eff')}} calls={{r.get('solver_calls')}}")
    if r.get("error"): print("   error:", r["error"][-400:].replace("\\n", " | "))
''')
    return p


def propose_prompt(parent, child, role_text, trace_dir, active, summary, check_path, arm, card_path, batch_indices):
    lib_note = ("You MAY `from cd_oracle import CDGame` (the public hypothesis space implied by the inventory: "
                "CDGame(inventory).H is an (N,24) 0/1 array, .feasible(history) / .expected_bits(q, alive) / "
                ".available_bits(alive)); it contains no truth."
                if arm == "lib" else
                "There is NO helper library: if you want the harness to track what the answers imply, you must "
                "write that logic yourself from the task's rules (gate inventory -> possible circuits).")
    return f"""A FROZEN 7B chat model plays CircuitDecoding inside a HARNESS you control. You CANNOT see the
truth tables. GOAL: edit the harness so the model RELIABLY pins the three circuits down within the 18
queries and returns the right 24-bit table -- judged only by label-free evidence.

READ FIRST:
- `{HERE / 'harness_base.py'}` -- the interface and the two invariants (frozen solver, label-free).
- `{AGENTS / (parent + '.py')}` -- your branch's CURRENT harness: copy it, then improve it.
- `{trace_dir}` -- full traces of every active branch on this batch ({', '.join(active)}): one `.md` per
  harness x instance (`<harness>__i<index>.md`), plus `.json` with the raw messages.
- Signals of every active branch on this batch (label-free, computed from the inventory and the
  answers; uniq_end = the answers pin the circuits down to one configuration):
{summary}

Harness API: self.env.query(c, a, b, x) -> bit (c in "ABC"; 18 per episode, then BudgetExhausted),
self.env.remaining, self.env.history, self.env.inventory, self.env.system_prompt,
self.env.last_turn_prompt, self.env.format_answer(...), self.llm(messages, temperature=0.7,
max_tokens=1200) -> str. The model must still be the one writing the queries you send; the harness
may re-ask it, change what it sees, veto a reply, assemble the final guess from what it said -- but
may not choose a query itself. {lib_note}

The frozen model's known habits (from earlier runs): it scans entries in a fixed order, keeps asking
outputs its own answers already determine, ignores 'do not ask these again' lists, and at the end
cannot assemble the 24-bit table from its own notes. Which of these the traces of THIS batch show
is for you to find out.

VERIFY BY RE-RUNNING -- do not guess, do not finish on a broken harness:
  1. Edit `{AGENTS / (child + '.py')}` (start from the parent). Keep `from harness_base import CDHarness`
     and `class Harness(CDHarness)`.
  2. Run it on this batch's instances:  python {check_path}
     It prints per-instance status / health / queries / uniq_end / alive_end / zero_info / repeat / eff.
  3. If an instance crashes or health is False, fix and re-run. Iterate until it runs cleanly.
  4. Do not stop on a crash. General mechanisms only: never branch on an instance's inventory or
     index, never hard-code tables.

Then write a proposal card to `{card_path}` as JSON:
{{"candidate": "{child}", "parent": "{parent}", "role": "<your role name>",
  "hypothesis": "<falsifiable claim>", "behavior_changes": ["..."], "preserved_behaviors": ["..."],
  "verification": ["what you ran and saw"], "risks": ["..."]}}

{role_text}"""


def judge_prompt(cands, incumbent, trace_dir, summary, choice_path, check_paths):
    checks = "\n".join(f"  {n}: python {p}" for n, p in check_paths.items())
    return f"""You are the JUDGE of a test-time harness-evolution loop for CircuitDecoding. {len(cands)} harnesses
were run on the same batch of instances. The incumbent is `{incumbent}`; the others are this batch's
candidates. Pick the ONE harness most likely to pin the circuits down and answer correctly on NEW
instances. You cannot see the truth tables.

Evidence (label-free):
{summary}

Full traces: `{trace_dir}` (`<harness>__i<index>.md`). Source files: `{AGENTS}`.
Rules of evidence: uniq_end and alive_end come from the answers and cannot be faked by text, but a
harness that chooses queries itself violates the interface -- read the source and REJECT such
candidates. Prefer general mechanisms over anything keyed to this batch. Prefer fewer wasted
queries (zero_info, repeat) and a parsable final guess (health). If no candidate is clearly better
than the incumbent on this evidence, keep the incumbent. You may re-run any harness on the batch:
{checks}

Decide, then write ONLY the chosen harness name to `{choice_path}` (e.g. run: echo <name> > {choice_path}).
Then STOP."""


# ----------------------------------------------------------------------------- backends

def _memory_dir():
    """Claude Code's per-directory auto-memory for this workspace; emptied before and archived after
    every session so a proposer or judge sees nothing but the run's files (see meta/cd_evolve/evolve.py)."""
    slug = "-" + str(HERE).strip("/").replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / "memory"


def _move_memory(run_dir, tag):
    mem = _memory_dir()
    if mem.exists() and any(mem.iterdir()):
        dst = Path(run_dir) / "claude_sessions" / f"{tag}_memory"
        dst.mkdir(parents=True, exist_ok=True)
        for f in mem.iterdir():
            shutil.move(str(f), str(dst / f.name))


def _session(prompt, name, run_dir, backend, model, timeout, effort):
    os.environ.pop("CLAUDECODE", None)
    _move_memory(run_dir, f"{name}_before")
    saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY",)}
    if backend == "solver":
        # route Claude Code's model calls to the frozen task model (vLLM Anthropic-compatible endpoint)
        os.environ["ANTHROPIC_BASE_URL"] = os.environ.get("TTHE_SOLVER_ANTHROPIC_URL", "http://127.0.0.1:18000")
        os.environ["ANTHROPIC_AUTH_TOKEN"] = "EMPTY"
    try:
        res = claude_wrapper.run(prompt=prompt, model=model, allowed_tools=PROPOSER_TOOLS, cwd=str(HERE),
                                 log_dir=str(Path(run_dir) / "claude_sessions"), name=name,
                                 timeout_seconds=timeout, effort=effort, progress=False)
        return getattr(res, "exit_code", 1)
    finally:
        _move_memory(run_dir, f"{name}_written")
        if backend == "solver":
            os.environ.pop("ANTHROPIC_BASE_URL", None); os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def stub_propose(parent, child, role_idx):
    src = (AGENTS / f"{parent}.py").read_text()
    (AGENTS / f"{child}.py").write_text(src + f"\n# stub edit by role {role_idx}\n")
    return True


def propose(parent, child, role_idx, trace_dir, active, results_by_name, run_dir, tag, batch_indices,
            solver_cfg, arm, backend, model, timeout, solve_timeout, effort="high"):
    """Returns True if agents/<child>.py exists afterwards (the loop re-audits it)."""
    (AGENTS / f"{child}.py").unlink(missing_ok=True)
    if backend == "stub":
        return stub_propose(parent, child, role_idx)
    role_name, role_text = ROLES[role_idx % len(ROLES)]
    summary = summary_table(results_by_name)
    check = check_script(run_dir, f"{tag}_{child}", child, batch_indices, solver_cfg, arm, solve_timeout)
    card = Path(run_dir) / "proposal_cards" / f"{child}.json"; card.parent.mkdir(parents=True, exist_ok=True)
    prompt = propose_prompt(parent, child, role_text, trace_dir, active, summary, check, arm, card, batch_indices)
    _session(prompt, f"{tag}_{child}_{role_name}", run_dir, backend, model, timeout, effort)
    return (AGENTS / f"{child}.py").exists()


def judge(cands, incumbent, trace_dir, results_by_name, run_dir, tag, batch_indices, solver_cfg, arm,
          backend, model, timeout, solve_timeout, effort="high"):
    """Returns the chosen harness name (falls back to the incumbent)."""
    if backend == "stub":
        def key(n):
            rs = results_by_name.get(n, [])
            if not rs:
                return (-1, -1)
            u = sum(1 for r in rs if (r.get("signals") or {}).get("uniq_end")) / len(rs)
            h = sum(1 for r in rs if (r.get("signals") or {}).get("health")) / len(rs)
            return (u, h)
        best = max(cands, key=key)
        return best if key(best) > key(incumbent) else incumbent
    summary = summary_table({n: results_by_name[n] for n in cands if n in results_by_name})
    checks = {n: check_script(run_dir, f"{tag}_judge_{n}", n, batch_indices, solver_cfg, arm, solve_timeout)
              for n in cands}
    choice = Path(run_dir) / f"choice_{tag}.txt"; choice.unlink(missing_ok=True)
    prompt = judge_prompt(cands, incumbent, trace_dir, summary, choice, checks)
    _session(prompt, f"{tag}_judge", run_dir, backend, model, timeout, effort)
    if choice.exists():
        picked = choice.read_text().strip().split()[-1].strip("`'\"") if choice.read_text().strip() else ""
        if picked in cands:
            return picked
    return incumbent
