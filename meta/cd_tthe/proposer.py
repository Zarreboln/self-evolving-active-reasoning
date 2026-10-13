"""Proposers and judge for the Circuit Decoding TTHE loop.

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
import time
import subprocess
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cd_evolve"))
import claude_wrapper  # noqa: E402  (copied verbatim from Meta-Harness, MIT)

AGENTS = HERE / "agents"
BUDGET = 18          # set by optimize.py from --budget
STRICT = False       # --signals strict: proposer/judge see only raw traces + run status; no oracle-derived
                     # signals (uniq_end, alive_end, zero_info, eff) and no prior about the model's habits
LABELED = False      # --signals labeled: strict + each episode's gold outcome (joint truth table correct / circuit-wise accuracy)
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
    if STRICT:
        line = (f"status={r.get('status')} parsed_final_guess={f(s.get('parsed_guess'))} queries_used={f(s.get('n_queries'))} "
                f"calls={r.get('solver_calls')} tokens_out={(r.get('tokens') or {}).get('out')}")
        if LABELED:
            g = r.get("gold") or {}
            line += f" | OUTCOME: joint truth table correct={g.get('joint_truth_table_correct')} circuit-wise accuracy={round(3 * float(g.get('circuit_wise_accuracy') or 0))}/3"
        return line
    return (f"status={r.get('status')} health={f(s.get('health'))} queries={f(s.get('n_queries'))} "
            f"uniq_end={f(s.get('uniq_end'))} alive_end={f(s.get('n_alive_end'))} zero_info={f(s.get('zero_info'))} "
            f"repeat={f(s.get('repeat'))} eff={f(s.get('eff'))} calls={r.get('solver_calls')} "
            f"tokens_out={(r.get('tokens') or {}).get('out')}")


def render_trace(r):
    """One episode as markdown: what the harness did, what the model said, what the environment
    answered. The truth table and the gold outcome are deliberately absent."""
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


PUBLIC_SIGNALS = ("health", "crashed", "parsed_guess", "n_queries", "repeat")
OUTCOME_KEYS = ("parsed_guess", "joint_truth_table_correct", "circuit_wise_accuracy")


def public_record(r, labeled=False):
    """What a proposer or judge may see of one episode: never the gold record or the oracle-derived
    signals; in the labeled arm the episode's outcome is added (labels are shown there by design)."""
    pub = {k: v for k, v in r.items() if k not in ("gold", "signals")}
    s = r.get("signals") or {}
    pub["signals"] = {k: s.get(k) for k in PUBLIC_SIGNALS}
    if labeled and r.get("gold"):
        pub["gold"] = {k: (r["gold"] or {}).get(k) for k in OUTCOME_KEYS}
    return pub


def write_traces(results, trace_dir, labeled=False):
    trace_dir = Path(trace_dir); trace_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        pub = public_record(r, labeled)
        (trace_dir / f"{r['harness']}__i{r['index']}.md").write_text(render_trace(pub))
        (trace_dir / f"{r['harness']}__i{r['index']}.json").write_text(json.dumps(pub))


OFF_LIMITS = ("\n\nOFF-LIMITS (your session transcript is audited automatically; a violation discards your work): "
              "do not read, list, grep or copy `~/.ar_harness_controller/`, `mt_puzzles/`, {stage0}other "
              "runs under `logs/` and their candidates `agents/cand_<otherrun>_*.py`, sessions under `claude_sessions/`, any `.claude/` directory (memory included), anything "
              "outside `meta/cd_tthe/` (the rest of the repository, its README and experiments), or any file that contains "
              "truth tables; do not run git. Work from this run's trace files named above, the harness sources in `agents/`, "
              "the interface doc, your check script and your scratch directory.")


def off_limits(arm):
    return OFF_LIMITS.format(stage0="" if arm == "lib" else "`stage0/` (the oracle code), ")


# ----------------------------------------------------------------------------- session audit
# What a proposer/judge session may touch: this folder, its scratch directory and system binaries.
# Requests are judged on PATHS (resolved against the command's `cd`), never on prose the session writes.
ALLOWED_ROOTS = [str(HERE)] + ["/private/tmp/claude-501", "/tmp", "/private/tmp", "/dev/", "/usr/", "/bin/", "/sbin/",
                               "/opt/", "/Library/", "/System/", "/etc/", "/var/folders", "/private/var/folders"]
NAME_RULES = [  # applied to path-bearing fields only
    ("controller store", r"ar_harness_controller"),
    ("memory / Claude Code state", r"/\.claude/(?!projects/[^/\s'\"]+/[0-9a-f-]+/tool-results/)"),
    ("session transcripts", r"claude_sessions"),
    ("task data file", r"cd_test\.jsonl|mt_puzzles/[A-Za-z0-9_]+\.(?:jsonl|txt)"),
    ("git", r"(?:^|[;&|(\s])git\s+(?:log|status|diff|show|blame|grep|ls-files|stash|checkout)\b"),
]
ORACLE_RULES = [("oracle code", r"stage0/|cd_oracle\.py|(?:from|import)\s+cd_oracle\b")]
# instance truths / gold / oracle-derived values as they appear in DATA (lower-case JSON), not in our source code
VAL_TRUTH = r"\"truth_bits\"\s*:|['\"]truth(?:_table|_tables|s)?['\"]\s*:\s*['\"\[(]?\s*[01]|\"circuit_joint_lookup\"\s*:|\"avg_reveals\"\s*:\s*[0-9]"
VAL_GOLD = r"\"gold\":\s*\{\s*\"parsed_guess\":\s*(?:true|false)"
VAL_ORACLE = r"\"(uniq_end|n_alive_end|zero_info|oracle_agree|eff)\":\s*(?:true|false|[0-9.]+|null)"
# real path candidates only: absolute paths under a real top-level directory, `../` and `~/`
_PATH_TOKEN = re.compile(r"(?<![A-Za-z0-9_.:])((?:~/|\.\./|/(?:Users|private|tmp|opt|usr|etc|var|Library|System|dev|bin|sbin|home|Volumes|Applications)(?:/|\b))[A-Za-z0-9_./~+-]*)")
_CD = re.compile(r"(?:^|[;&|]\s*|\n\s*)cd\s+([^\s;&|]+)")


def _path_fields(name, inp):
    if name == "Bash":
        return [inp.get("command") or ""]
    if name in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit"):
        return [inp.get("file_path") or inp.get("notebook_path") or ""]
    if name == "Glob":
        return [inp.get("pattern") or "", inp.get("path") or ""]
    if name == "Grep":
        return [inp.get("path") or "", inp.get("glob") or ""]
    return []


def _resolve(tok, cwd):
    tok = os.path.expanduser(tok)
    return os.path.normpath(tok if tok.startswith("/") else os.path.join(cwd, tok))


_SPILL = re.compile(r"/\.claude/projects/[^/]+/[0-9a-f-]+/tool-results/")   # Claude Code's own large-output spill files


def _path_ok(resolved):
    if _SPILL.search(resolved):
        return True
    return any(resolved == r.rstrip("/") or resolved.startswith(r if r.endswith("/") else r + "/") for r in ALLOWED_ROOTS)


_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\1\s*(?=\n|$)", re.S)
_ACCESS_CTX = re.compile(r"(?:open|Path|glob|iglob|listdir|scandir|exists|isfile|isdir|read_text|read_bytes|walk|"
                         r"cat|ls|head|tail|sed|grep|cp|mv|rm|python3?|source)\s*\(?\s*['\"]?$")


def _split_heredocs(text):
    """-> (shell text with heredoc bodies removed, [heredoc bodies])"""
    bodies = [m.group(2) for m in _HEREDOC.finditer(text)]
    return _HEREDOC.sub(lambda m: f"<<{m.group(1)}\n{m.group(1)}", text), bodies


def _escapes(text, cwd0, whole=False):
    """Paths in a command/field that resolve outside the allowed roots. Relative ones are resolved
    against the command's `cd` target; a `cd` that itself lands outside is reported too. Heredoc
    bodies (scripts, docstrings, cards written through the shell) are text: a path there counts only
    when it is absolute or sits in an access-like context such as open('../x')."""
    if whole:
        r = _resolve(text.strip(), cwd0)
        return [] if _path_ok(r) else [f"{text.strip()} -> {r}"]
    text, bodies = _split_heredocs(text)
    bad = []
    _escapes.cwd_after = cwd0
    for body in bodies:
        for m in _PATH_TOKEN.finditer(body):
            t = m.group(1)
            if t.startswith("../") and not _ACCESS_CTX.search(body[max(0, m.start() - 40):m.start()]):
                continue
            r = _resolve(t.rstrip("/"), cwd0)
            if not _path_ok(r):
                bad.append(f"{t} -> {r} (heredoc)")
    cwd = cwd0
    pos = 0
    segments = []
    for m in _CD.finditer(text):           # cwd changes along the command
        segments.append((pos, m.start(), cwd))
        cwd = _resolve(m.group(1).strip("'\""), cwd); pos = m.end()
        if not _path_ok(cwd):
            bad.append(f"cd {m.group(1)} -> {cwd}")
    segments.append((pos, len(text), cwd))
    _escapes.cwd_after = cwd
    for a, b, c in segments:
        for t in _PATH_TOKEN.findall(text[a:b]):
            r = _resolve(t.rstrip("/"), c)
            if not _path_ok(r):
                bad.append(f"{t} -> {r}")
    return bad


def audit_session(run_dir, name, arm):
    """Scan the newest session named `name`: which paths it touched and what came back. Returns the
    audit dict (clean / hits) and records it in the session dir and in session_audits.jsonl."""
    sessions = sorted((Path(run_dir) / "claude_sessions").glob(f"*_{name}"))
    if not sessions:
        return {"name": name, "clean": True, "hits": [], "note": "no session dir"}
    sess = sessions[-1]; ev = sess / "events.jsonl"
    run_name = re.escape(Path(run_dir).name)
    rules = [(k, re.compile(rx)) for k, rx in NAME_RULES] + [("other run's logs", re.compile(rf"logs/(?!{run_name}\b)[A-Za-z0-9_]+\b")),
                                                             ("other run's candidates", re.compile(rf"cand_(?!{run_name}_)[A-Za-z0-9]+_[A-Za-z0-9_]*\.py"))]
    if arm != "lib":
        rules += [(k, re.compile(rx)) for k, rx in ORACLE_RULES]
    val_rx = [("instance truth", re.compile(VAL_TRUTH)), ("oracle signal value", re.compile(VAL_ORACLE))]
    if not LABELED:
        val_rx.append(("gold record", re.compile(VAL_GOLD)))
    uses, hits = {}, []
    cwd = str(HERE)                     # Claude Code's Bash keeps its working directory between calls
    for line in (ev.read_text().splitlines() if ev.exists() else []):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") == "assistant":
            for b in (d.get("message") or {}).get("content") or []:
                if b.get("type") != "tool_use":
                    continue
                uses[b["id"]] = (b["name"], b.get("input") or {})
                for field in _path_fields(b["name"], b.get("input") or {}):
                    if not field:
                        continue
                    for kind, rx in rules:
                        m = rx.search(field)
                        if m:
                            hits.append({"kind": kind, "match": m.group(0)[:80], "tool": b["name"], "input": field[:300]})
                    esc = _escapes(field, cwd, whole=(b["name"] != "Bash"))
                    if b["name"] == "Bash":
                        cwd = _escapes.cwd_after
                    if esc:
                        hits.append({"kind": "path outside the folder", "match": esc[0][:120], "tool": b["name"], "input": field[:300]})
        elif d.get("type") == "user":
            for b in (d.get("message") or {}).get("content") or []:
                if b.get("type") != "tool_result":
                    continue
                c = b.get("content"); txt = c if isinstance(c, str) else json.dumps(c)
                for kind, rx in val_rx:
                    m = rx.search(txt)
                    if m:
                        tool, inp = uses.get(b.get("tool_use_id"), ("?", {}))
                        hits.append({"kind": kind, "match": m.group(0)[:80], "tool": tool, "input": json.dumps(inp)[:300]})
    res = {"name": name, "session": sess.name, "clean": not hits, "hits": hits[:20]}
    (sess / "audit.json").write_text(json.dumps(res, indent=1))
    with (Path(run_dir) / "session_audits.jsonl").open("a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def summary_table(results_by_name):
    rows = []
    for name, rs in results_by_name.items():
        sig = [r.get("signals") or {} for r in rs]
        if STRICT:
            parsed = sum(1 for x in sig if x.get("parsed_guess")) / max(1, len(sig))
            q = sum(x.get("n_queries") or 0 for x in sig) / max(1, len(sig))
            toks = sum((r.get("tokens") or {}).get("out", 0) for r in rs) / max(1, len(rs))
            line = (f"{name:<34} n={len(rs):<3} parsed_final_guess={100*parsed:5.1f}%  queries_used/ep={q:5.1f}  "
                    f"tokens_out/ep={toks:7.0f}  crashes={sum(1 for r in rs if r.get('status') != 'ok')}")
            if LABELED:
                ex = sum(1 for r in rs if (r.get("gold") or {}).get("joint_truth_table_correct")) / max(1, len(rs))
                ck = sum(float((r.get("gold") or {}).get("circuit_wise_accuracy") or 0) for r in rs) / max(1, len(rs))
                line += f"  | OUTCOME: joint truth table correct={100*ex:5.1f}%  circuit-wise accuracy={100*ck:5.1f}%"
            rows.append(line)
            continue
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


def check_script(run_dir, tag, name, indices, solver_cfg, arm, timeout, budget=18):
    """A script the proposer can run: its candidate on the batch instances, label-free output."""
    p = Path(run_dir) / f"check_{tag}.py"
    reveal = "outcome" if LABELED else "none"
    p.write_text(f'''import json, subprocess, sys, os
from pathlib import Path
HERE = Path({str(HERE)!r})
name = {name!r}; idx = {list(indices)!r}; solver = {solver_cfg!r}; arm = {arm!r}
out_dir = Path({str(run_dir)!r}) / "checks" / {tag!r}; out_dir.mkdir(parents=True, exist_ok=True)
for i in idx:
    task = out_dir / f"task_{{i}}.json"; res = out_dir / f"res_{{i}}.json"
    task.write_text(json.dumps({{"harness": name, "index": i, "output": str(res), "solver": solver, "arm": arm, "budget": {budget}, "reveal": {reveal!r}}}))
    try:
        subprocess.run([sys.executable, str(HERE / "solve_worker.py"), "--task", str(task)], timeout={int(timeout)}, check=False)
    except subprocess.TimeoutExpired:
        print(f"instance {{i}}: TIMEOUT after {int(timeout)}s"); continue
    if not res.exists():
        print(f"instance {{i}}: no result (worker died)"); continue
    r = json.loads(res.read_text()); s = r.get("signals") or {{}}
    if {STRICT!r}:
        line = f"instance {{i}}: status={{r['status']}} parsed_final_guess={{s.get('parsed_guess')}} queries_used={{s.get('n_queries')}} calls={{r.get('solver_calls')}}"
        if {LABELED!r}:
            g = r.get("gold") or {{}}
            line += f" | OUTCOME: joint truth table correct={{g.get('joint_truth_table_correct')}} circuit-wise accuracy={{round(3*float(g.get('circuit_wise_accuracy') or 0))}}/3"
        print(line)
    else:
        print(f"instance {{i}}: status={{r['status']}} health={{s.get('health')}} queries={{s.get('n_queries')}} "
              f"uniq_end={{s.get('uniq_end')}} alive_end={{s.get('n_alive_end')}} zero_info={{s.get('zero_info')}} "
              f"repeat={{s.get('repeat')}} eff={{s.get('eff')}} calls={{r.get('solver_calls')}}")
    if r.get("error"): print("   error:", r["error"][-400:].replace("\\n", " | "))
''')
    return p


def propose_prompt(parent, child, role_text, trace_dir, active, summary, check_path, arm, card_path, batch_indices):
    if STRICT:
        return propose_prompt_strict(parent, child, role_text, trace_dir, active, summary, check_path, card_path)
    lib_note = ("You MAY `from cd_oracle import CDGame` (the public hypothesis space implied by the inventory: "
                "CDGame(inventory).H is an (N,24) 0/1 array, .feasible(history) / .expected_bits(q, alive) / "
                ".available_bits(alive)); it contains no truth."
                if arm == "lib" else
                "There is NO helper library: if you want the harness to track what the answers imply, you must "
                "write that logic yourself from the task's rules (gate inventory -> possible circuits).")
    return f"""A FROZEN 7B chat model plays Circuit Decoding inside a HARNESS you control. You CANNOT see the
truth tables. GOAL: edit the harness so the model RELIABLY pins the three circuits down within the {BUDGET}
queries and returns the right 24-bit table -- judged only by label-free evidence.

READ FIRST:
- `{HERE / 'harness_base.py'}` -- the interface and the two invariants (frozen solver, label-free).
- `{AGENTS / (parent + '.py')}` -- your branch's CURRENT harness: copy it, then improve it.
- `{trace_dir}` -- full traces of every active branch on this batch ({', '.join(active)}): one `.md` per
  harness x instance (`<harness>__i<index>.md`), plus `.json` with the raw messages.
- Signals of every active branch on this batch (label-free, computed from the inventory and the
  answers; uniq_end = the answers pin the circuits down to one configuration):
{summary}

Harness API: self.env.query(c, a, b, x) -> bit (c in "ABC"; self.env.budget per episode -- {BUDGET} in this run -- then BudgetExhausted),
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


def propose_prompt_strict(parent, child, role_text, trace_dir, active, summary, check_path, card_path):
    outcome_note = ("Each episode's OUTCOME (joint truth table correct?, and the circuit-wise accuracy: how many of the "
                    "three circuits' tables were right) is shown in the traces and the run summary; nothing else is evaluated for you."
                    if LABELED else
                    "You are given no evaluation of the episodes beyond whether they ran. Decide what \"going well\" "
                    "means from the task rules and the traces yourself.")
    return f"""A FROZEN 7B chat model plays Circuit Decoding inside a HARNESS you control. You CANNOT see the
truth tables. {outcome_note} GOAL: edit the harness so the model RELIABLY deduces the three circuits
within the {BUDGET} queries and returns the right 24-bit table.

READ FIRST:
- `{HERE / 'harness_base.py'}` -- the interface and the two invariants (frozen solver, label-free).
- `{AGENTS / (parent + '.py')}` -- your branch's CURRENT harness: copy it, then improve it.
- `{trace_dir}` -- full traces of every active branch on this batch ({', '.join(active)}): one `.md` per
  harness x instance (`<harness>__i<index>.md`): the queries the model asked, the answers the
  environment gave, every message sent to the model and every reply. Nothing else is provided.
- Run summary (what ran, what crashed):
{summary}

Harness API: self.env.query(c, a, b, x) -> bit (c in "ABC"; self.env.budget per episode -- {BUDGET} in this
run -- then BudgetExhausted), self.env.remaining, self.env.history, self.env.inventory,
self.env.system_prompt, self.env.last_turn_prompt, self.env.format_answer(...), self.llm(messages,
temperature=0.7, max_tokens=1200) -> str. The model must still be the one writing the queries you send;
the harness may re-ask it, change what it sees, veto a reply, assemble the final guess from what it
said -- but may not choose a query itself. There is no helper library: anything you want the harness
to know about the circuits must come from the task rules in the system prompt and the answers.

VERIFY BY RE-RUNNING -- do not guess, do not finish on a broken harness:
  1. Edit `{AGENTS / (child + '.py')}` (start from the parent). Keep `from harness_base import CDHarness`
     and `class Harness(CDHarness)`.
  2. Run it on this batch's instances:  python {check_path}
     It prints per-instance run status, whether a final guess was produced, and queries used.
     Read the traces it writes to judge the rest yourself.
  3. If an instance crashes, fix and re-run. General mechanisms only: never branch on an instance's
     inventory or index, never hard-code tables.

Then write a proposal card to `{card_path}` as JSON:
{{"candidate": "{child}", "parent": "{parent}", "role": "<your role name>",
  "hypothesis": "<falsifiable claim>", "behavior_changes": ["..."], "preserved_behaviors": ["..."],
  "verification": ["what you ran and saw"], "risks": ["..."]}}

{role_text}"""


def judge_prompt(cands, incumbent, trace_dir, summary, choice_path, check_paths):
    if STRICT:
        checks = "\n".join(f"  {n}: python {p}" for n, p in check_paths.items())
        return f"""You are the JUDGE of a test-time harness-evolution loop for Circuit Decoding. {len(cands)} harnesses
were run on the same batch of instances. The incumbent is `{incumbent}`; the others are this batch's
candidates. Pick the ONE harness most likely to deduce the circuits correctly on NEW instances. You
cannot see the truth tables. {"Each episode's OUTCOME (joint truth table correct? circuit-wise accuracy?) is in the summary and traces." if LABELED else "No evaluation beyond run status is provided."} Judge from the traces
(`{trace_dir}`, `<harness>__i<index>.md`) and the source files (`{AGENTS}`) using the task rules.

Run summary:
{summary}

A harness that chooses queries itself violates the interface -- read the source and REJECT such
candidates. Prefer general mechanisms over anything keyed to this batch. If no candidate is clearly
better than the incumbent on this evidence, keep the incumbent. You may re-run any harness on the batch:
{checks}

Decide, then write ONLY the chosen harness name to `{choice_path}` (e.g. run: echo <name> > {choice_path}).
Then STOP."""
    checks = "\n".join(f"  {n}: python {p}" for n, p in check_paths.items())
    return f"""You are the JUDGE of a test-time harness-evolution loop for Circuit Decoding. {len(cands)} harnesses
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

def _memory_dirs():
    """Claude Code keys its auto-memory by project directory, and for a session started inside this
    folder that key turned out to be the repository root, not HERE: the pilot sessions wrote notes to
    ~/.claude/projects/<repo-root-slug>/memory and later sessions read them (a cross-session channel,
    experiments/08). Every memory dir whose project slug starts with the repo root's slug is emptied
    before and archived after each session."""
    root_slug = re.sub(r"[^A-Za-z0-9]", "-", str(HERE.parent.parent))   # Claude Code's slug: every non-alphanumeric -> '-'
    base = Path.home() / ".claude" / "projects"
    return [d / "memory" for d in base.glob(root_slug + "*") if (d / "memory").exists()]


def _move_memory(run_dir, tag):
    for mem in _memory_dirs():
        files = list(mem.iterdir())
        if files:
            dst = Path(run_dir) / "claude_sessions" / f"{tag}_memory" / mem.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.move(str(f), str(dst / f.name))


LIMIT_RX = re.compile(r"hit your (?:session|usage|weekly|daily) limit|usage limit reached|resets? (?:at )?\d{1,2}(?::\d{2})?\s*(?:am|pm)", re.I)


def _newest_session_dir(run_dir, name):
    ds = sorted((Path(run_dir) / "claude_sessions").glob(f"*_{name}"))
    return ds[-1] if ds else None


def _limit_hit(run_dir, name):
    d = _newest_session_dir(run_dir, name)
    if d is None:
        return False
    txt = (d / "response.md").read_text() if (d / "response.md").exists() else ""
    return bool(LIMIT_RX.search(txt[:600]))


def wait_for_subscription(model, poll_s=600, max_s=4 * 3600):
    """The Max subscription answered with its limit message: probe with a one-word call every poll_s
    seconds until it answers normally (or give up after max_s)."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        print(f"   SUBSCRIPTION LIMIT: waiting {poll_s // 60} min before probing again "
              f"({(time.time() - t0) / 60:.0f} min so far)", flush=True)
        time.sleep(poll_s)
        env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "ANTHROPIC_API_KEY")}
        try:
            out = subprocess.run(["claude", "-p", "Reply with the single word ok.", "--model", model, "--output-format", "json"],
                                 capture_output=True, text=True, timeout=180, env=env).stdout
            res = json.loads(out).get("result", "") if out.strip().startswith("{") else out
        except Exception as e:                       # noqa: BLE001
            res = f"probe failed: {e}"
        if not LIMIT_RX.search(res or ""):
            print("   SUBSCRIPTION LIMIT lifted, resuming", flush=True)
            return True
    return False


def _session(prompt, name, run_dir, backend, model, timeout, effort):
    for attempt in range(4):
        code = _session_once(prompt, name, run_dir, backend, model, timeout, effort)
        if backend != "claude" or not _limit_hit(run_dir, name):
            return code
        d = _newest_session_dir(run_dir, name)
        if d is not None:
            d.rename(d.with_name(d.name + "_LIMIT"))   # keep it out of the audit's "newest session" lookup
        print(f"   SUBSCRIPTION LIMIT hit in session {name} (attempt {attempt + 1}); pausing", flush=True)
        if not wait_for_subscription(model):
            break
    return 1


def _session_once(prompt, name, run_dir, backend, model, timeout, effort):
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


STRICT_ROLES = [   # the same three search roles, with no examples of failure types (strict mode gives no priors)
    ("conservative_repair",
     "CONSERVATIVE REPAIR ROLE: keep every parent behavior the traces show working. Make the smallest "
     "general change that removes a concrete, trace-visible failure. Do not add machinery the traces do not call for."),
    ("independent_exploration",
     "INDEPENDENT EXPLORATION ROLE: test a genuinely different causal hypothesis about where the episodes go "
     "wrong, implemented as one simple mechanism. Do not copy the other branch's mechanism because it is the current default."),
    ("adversarial_audit",
     "ADVERSARIAL AUDIT AND SYNTHESIS ROLE: start from disagreements between the branches' traces on the same "
     "instance, decide from the evidence which behavior is supported, and synthesize only the independently "
     "supported strengths. Never equate 'ran without error' with 'solved'."),
]


def propose(parent, child, role_idx, trace_dir, active, results_by_name, run_dir, tag, batch_indices,
            solver_cfg, arm, backend, model, timeout, solve_timeout, effort="high"):
    """Returns True if agents/<child>.py exists afterwards (the loop re-audits it)."""
    (AGENTS / f"{child}.py").unlink(missing_ok=True)
    if backend == "stub":
        return stub_propose(parent, child, role_idx)
    role_name, role_text = (STRICT_ROLES if STRICT else ROLES)[role_idx % len(ROLES)]
    summary = summary_table(results_by_name)
    check = check_script(run_dir, f"{tag}_{child}", child, batch_indices, solver_cfg, arm, solve_timeout, BUDGET)
    card = Path(run_dir) / "proposal_cards" / f"{child}.json"; card.parent.mkdir(parents=True, exist_ok=True)
    prompt = propose_prompt(parent, child, role_text, trace_dir, active, summary, check, arm, card, batch_indices) + off_limits(arm)
    _session(prompt, f"{tag}_{child}_{role_name}", run_dir, backend, model, timeout, effort)
    aud = audit_session(run_dir, f"{tag}_{child}_{role_name}", arm)
    if not aud["clean"]:
        print(f"   AUDIT: proposer session {tag}_{child} touched off-limits information -> candidate discarded "
              f"({aud['hits'][0]['kind']}: {aud['hits'][0]['match']})", flush=True)
        (AGENTS / f"{child}.py").unlink(missing_ok=True)
        return False
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
    checks = {n: check_script(run_dir, f"{tag}_judge_{n}", n, batch_indices, solver_cfg, arm, solve_timeout, BUDGET)
              for n in cands}
    choice = Path(run_dir) / f"choice_{tag}.txt"; choice.unlink(missing_ok=True)
    prompt = judge_prompt(cands, incumbent, trace_dir, summary, choice, checks) + off_limits(arm)
    _session(prompt, f"{tag}_judge", run_dir, backend, model, timeout, effort)
    aud = audit_session(run_dir, f"{tag}_judge", arm)
    if not aud["clean"]:
        print(f"   AUDIT: judge session {tag} touched off-limits information -> verdict discarded, incumbent kept "
              f"({aud['hits'][0]['kind']}: {aud['hits'][0]['match']})", flush=True)
        return incumbent
    if choice.exists():
        picked = choice.read_text().strip().split()[-1].strip("`'\"") if choice.read_text().strip() else ""
        if picked in cands:
            return picked
    return incumbent
