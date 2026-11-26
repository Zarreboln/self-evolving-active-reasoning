"""ACE (Agentic Context Engineering, arXiv 2510.04618) on experiment B's tasks.

How ACE's parts map onto our setting
------------------------------------
Generator  = the task's starting agent (a seed in mb/evolve/seeds/) playing one episode on Qwen3.5-9B,
             with the playbook in its prompt (harness.py). One "sample" of ACE = one problem.
Reflector  = Opus 5.5 with ACE's REFLECTOR_PROMPT_NO_GT (ace/prompts/reflector.py). In ACE's agent
             setting (ace-appworld) the Reflector reads the full conversation of the episode; here the
             episode's trace (every model call, reply, action and observation), with our inserted
             playbook block cut out. "Question" = the first prompt the agent saw; "Model's Predicted
             Answer" = the actions it took; "Part of Playbook ..." = the bullets whose ids appear in the
             agent's replies, else the whole playbook (our agents do not output ACE's bullet_ids field).
Curator    = Opus 5.5 with ACE's CURATOR_PROMPT_NO_GT (ace/prompts/curator.py); as in ace-appworld's
             curator_call, the episode's conversation follows the prompt. Its ADD operations are
             applied with ACE's own apply_curator_operations; bullet counters with update_bullet_counts.

Before deployment (offline adaptation, ACE._offline_train/_train_single_sample): the 7 development
tasks' search problems (20 each), interleaved task by task, for up to 5 epochs, batch size 1. Per
problem: play; if it failed, up to 5 rounds of reflect -> replay the same problem with the reflection
(each replay is an evaluated episode); if it succeeded, reflect once; then curate. Correctness =
the episode's `success` on the search split (scores are allowed before deployment). Stops when the
equal episode budget (8 rounds x 140 = 1,120 episodes) is used up. The final playbook is deployed
(ACE has no candidates, so no selection step).

At test time (online adaptation, ace-appworld solve_task_wo_gt; ACE paper: "for each sample, the model
first predicts with the current context, then updates its context based on that sample"): per new
task, its problems in the fixed order, one episode each; reflect on the trace with no correctness
signal (the environment's own messages in the trace are all there is), curate, next problem. Each
task starts from the offline playbook ("ace") or from ACE's empty playbook ("ace_online_only").

Everything is resumable: state.json is written after each problem, and every model-role call and
every episode is cached under its id, so a problem cut off midway is redone from the caches.

Deviations from ACE's code and paper, and why
---------------------------------------------
 1. Prompts: the task-neutral official Reflector/Curator prompts of the core repo (the _NO_GT
    variants, no answers exist here); ace-appworld's agent prompts are AppWorld-specific (API docs,
    Venmo/Spotify examples). The core repo's ground-truth-free mode still passes the correctness
    strings "Predicted answer (does not) match(es) ground truth"; we pass them before deployment
    (from the search split's `success`), plus "Episode score (0 to 1): x".
 2. Generator: our agents keep their own prompts and action formats, so the playbook is inserted
    with ace-appworld's sentence and BEGIN/END markers (harness.py) instead of ACE's QA generator
    prompt, and the agents do not return ACE's `bullet_ids` JSON field: bullet ids the agent cites in
    its replies are used if any, else the Reflector is shown the whole playbook.
 3. Post-curation generation (ace.py STEP 4) is skipped: it only measures post-train accuracy and
    does not change the playbook, but here it would cost an evaluated episode of the equal budget.
 4. Validation-based best-playbook choice (eval_steps) is skipped: the design gives ACE no
    selection step; the final playbook is deployed.
 5. Epochs/rounds: the paper's maxima (5 and 5; repo defaults are 1 and 3), cut by the equal episode
    budget of 1,120; replays during refinement rounds count as episodes.
 6. Online: one episode per problem (test-time rule), so no refinement rounds and no correctness
    signal; ACE's online loop (window test, then _train_single_sample with regeneration) would
    replay problems. Reflector feedback = TEST_FEEDBACK; the trace holds what the environment said.
    This is ace-appworld's solve_task_wo_gt (one attempt, reflect, curate). Online window = 1
    problem (repo default online_eval_frequency 15 tests 15 problems before learning from them;
    the paper describes per-sample updates).
 7. Playbook size: ACE's prompt states an 80,000-token budget and its de-duplication
    (BulletpointAnalyzer, embeddings) is off by default; our frozen agent has a 64K-token context
    that also holds the episode, so the stated budget is 16,000 tokens (paper A.6 finds 10K-100K
    equivalent) and it is enforced: over budget, the bullet with the lowest helpful-harmful count,
    oldest first, is dropped (the paper's lazy "prune older or low-utility entries"). Tokens are
    estimated as characters / 4 (ACE uses tiktoken).
 8. Role prompts carry at most 120,000 characters of a trace (first 20K + last 100K); ACE's AppWorld
    agent trims its history similarly (max_output_length) but at a different size.
 9. LLM: Opus 5.5 for both roles through headless Claude Code (llm.py): no temperature control
    (ACE uses 0), a one-line system prompt (ACE sends none), max output 4,096 tokens, no thinking.
10. Shared-store rules (guard.py): names redacted and answer-bearing ADDs dropped before applying.
11. Several tasks: ACE trains on one dataset; here the 7 development tasks' search problems are
    interleaved into one stream and one playbook (the design's "shared store").
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import threading
import time
from pathlib import Path

from mb.baselines.ace import guard, harness, official
from mb.baselines.ace.official import pb

TEST_FEEDBACK = ("No feedback beyond the trace: the environment's messages in the trace above are all "
                 "that is known about how this attempt went.")
TRACE_HEAD, TRACE_TAIL = 20_000, 100_000          # characters of a trace a role prompt may carry
BULLET_RE = re.compile(r"\[([a-z]{3,}-\d{5})\]")    # ace/core/generator.py _extract_bullet_ids_regex


def _atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent=1) if not isinstance(obj, str) else obj)
    os.replace(tmp, path)


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def approx_tokens(text):
    """ACE counts with tiktoken; ~4 characters per token here (tiktoken is not needed on the Mac)."""
    return len(text) // 4


# ---- reading a trace ----

def first_prompt(trace):
    m = re.search(r"## model call \(before action 1\)\n(.*?)\*\*reply:\*\*", trace, re.S)
    return m.group(1).strip() if m else ""


def actions(trace):
    return re.findall(r"## action \d+\n```\n(.*?)\n```", trace, re.S)


def clip(text):
    if len(text) <= TRACE_HEAD + TRACE_TAIL:
        return text
    cut = len(text) - TRACE_HEAD - TRACE_TAIL
    return text[:TRACE_HEAD] + f"\n[... {cut} characters of the trace omitted ...]\n" + text[-TRACE_TAIL:]


# ---- ACE's parsing, as in ace/core/reflector.py and ace/core/curator.py (non-JSON mode) ----

def extract_bullet_tags(response):
    try:
        start = response.find('"bullet_tags"')
        if start != -1:
            b = response.find("[", start)
            if b != -1:
                depth, end = 0, b
                for i in range(b, len(response)):
                    if response[i] == "[":
                        depth += 1
                    elif response[i] == "]":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                return json.loads(response[b:end])
    except Exception:
        pass
    return []


def extract_operations(response):
    info = pb.extract_json_from_text(response, "operations")
    if not info or not isinstance(info.get("reasoning"), str) or not isinstance(info.get("operations"), list):
        raise ValueError("curator reply is not the required JSON")
    for i, op in enumerate(info["operations"]):
        if not isinstance(op, dict) or "type" not in op:
            raise ValueError(f"operation {i} malformed")
        if op["type"] == "ADD" and {"type", "section", "content"} - set(op):
            raise ValueError(f"ADD operation {i} missing fields")
    return [op for op in info["operations"] if op["type"] == "ADD"]     # ACE applies ADD only


def prune(playbook, token_budget):
    """Size cap (stand-in for ACE's lazy grow-and-refine, paper Sec. 3.2 / A.6): while the playbook is
    over its token budget, drop the bullet with the lowest helpful - harmful count, oldest first."""
    dropped = []
    while approx_tokens(playbook) > token_budget:
        lines = playbook.split("\n")
        bullets = [(p["helpful"] - p["harmful"], i, p["id"]) for i, l in enumerate(lines)
                   if (p := pb.parse_playbook_line(l))]
        if not bullets:
            break
        _, i, bid = min(bullets)
        dropped.append(bid)
        playbook = "\n".join(lines[:i] + lines[i + 1:])
    return playbook, dropped


class ACERun:
    """One ACE adaptation stream with its own playbook, state and caches under `root`."""

    def __init__(self, root: Path, llm, backend, mailbox_prefix: str, token_budget: int,
                 seed_override: str | None = None, curator_history: bool = True, initial_playbook=None):
        self.root, self.llm, self.backend = Path(root), llm, backend
        self.prefix, self.token_budget = mailbox_prefix, token_budget
        self.seed_override, self.curator_history = seed_override, curator_history
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "playbook": initial_playbook if initial_playbook is not None else official.EMPTY_PLAYBOOK,
            "next_global_id": 1, "episodes": 0, "done": [], "played_answers": []}
        if initial_playbook is not None and not self.state_path.exists():
            self.state["next_global_id"] = pb.get_next_global_id(initial_playbook)

    # -- cached calls --
    def call(self, role, call_id, prompt):
        p = self.root / "calls" / f"{call_id}.json"
        if p.exists():
            c = json.loads(p.read_text())
            if c["prompt_sha"] == _sha(prompt):
                return c["reply"]
        reply = self.llm(role, f"{self.prefix}/{call_id}", prompt)
        _atomic(p, {"role": role, "prompt_sha": _sha(prompt), "prompt_chars": len(prompt), "reply": reply})
        return reply

    def play(self, step_id, cell, split, inst, playbook, reflection=None, visible_only=False):
        p = self.root / "plays" / f"{step_id}.json"
        if p.exists():
            return json.loads(p.read_text())
        src = harness.build(harness.seed_for(cell, self.seed_override), playbook, reflection)
        hp = self.root / "harness" / f"{step_id}.py"
        _atomic(hp, src)
        key = f"{self.prefix}/{step_id}"
        ep = self.backend.play(key, hp, cell, split, [inst], visible_only)[0]
        ep.update(cell=cell, key=key, playbook_used=playbook, reflection_used=reflection)
        _atomic(p, ep)
        return ep

    # -- ACE's roles --
    def visible(self, ep):
        # at test time the worker has already dropped the broker's score line (mb/evolve/worker.py)
        return harness.strip_block(ep["trace"], ep["playbook_used"], ep["reflection_used"])

    def reflect(self, call_id, ep, playbook, feedback):
        trace = self.visible(ep)
        q = first_prompt(trace)
        rest = trace.split("**reply:**", 1)[1] if "**reply:**" in trace else trace
        ids = sorted(set(BULLET_RE.findall(rest)))
        used = pb.extract_playbook_bullets(playbook, ids) if ids else playbook
        prompt = official.REFLECTOR_PROMPT_NO_GT.format(
            clip(q), clip("**reply:**" + rest), "\n".join(actions(trace)) or "(no action)", feedback, used)
        reply = self.call("reflector", call_id, prompt)
        return reply, extract_bullet_tags(reply)

    def curate(self, call_id, reflection, ep, step, total):
        trace = self.visible(ep)
        playbook = self.state["playbook"]
        prompt = official.CURATOR_PROMPT_NO_GT.format(
            current_step=step, total_samples=total, token_budget=self.token_budget,
            playbook_stats=json.dumps(pb.get_playbook_stats(playbook), indent=2),
            recent_reflection=reflection, current_playbook=playbook, question_context=clip(first_prompt(trace)))
        if self.curator_history:                    # ace-appworld curator_call appends the conversation
            prompt += "\n\n=== FULL CONVERSATION HISTORY ===\n" + clip(trace)
        reply = self.call("curator", call_id, prompt)
        try:
            ops = extract_operations(reply)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return {"curator_error": str(e)}         # ACE skips the update and continues
        ops, glog = guard.clean_operations(ops, self.state["played_answers"])
        with contextlib.redirect_stdout(io.StringIO()):          # ACE prints one line per bullet
            new, nid = pb.apply_curator_operations(playbook, ops, self.state["next_global_id"])
        new, dropped = prune(new, self.token_budget)
        self.state.update(playbook=new, next_global_id=nid)
        return {"added": len(ops), "guard": glog, "pruned": dropped}

    def update_counts(self, tags):
        if tags:
            with contextlib.redirect_stdout(io.StringIO()):
                self.state["playbook"] = pb.update_bullet_counts(self.state["playbook"], tags)

    def remember(self, inst):
        if inst not in self.state["played_answers"]:
            self.state["played_answers"].append(inst)

    def commit(self, done_id, event):
        self.state["done"].append(done_id)
        _atomic(self.state_path, self.state)
        _atomic(self.root / "playbook.txt", self.state["playbook"])
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "id": done_id, **event}) + "\n")


def offline(run: ACERun, problems, budget, epochs=official.NUM_EPOCHS, rounds=official.MAX_NUM_ROUNDS):
    """problems: [(cell, instance)] on the search split, in ACE's order. Returns the final playbook."""
    total = len(problems)
    for epoch in range(1, epochs + 1):
        for step, (cell, inst) in enumerate(problems, 1):
            sid = f"e{epoch}_s{step:03d}"
            if sid in run.state["done"]:
                continue
            if run.state["episodes"] >= budget:
                return _finish(run, "budget")
            st = run.state
            run.remember(inst)
            ep = run.play(f"{sid}_gen_initial", cell, "search", inst, st["playbook"])
            n, correct, first = 1, bool(ep["success"]), ep
            fb = lambda e: (official.FEEDBACK_RIGHT if e["success"] else official.FEEDBACK_WRONG) + \
                f"\nEpisode score (0 to 1): {e['score']}"
            if not correct:
                for r in range(rounds):
                    reflection, tags = run.reflect(f"{sid}_round_{r}", ep, st["playbook"], fb(ep))
                    run.update_counts(tags)
                    if st["episodes"] + n >= budget:
                        break
                    ep = run.play(f"{sid}_post_reflect_round_{r}", cell, "search", inst, st["playbook"], reflection)
                    n += 1
                    if ep["success"]:
                        correct = True
                        break
            else:
                reflection, tags = run.reflect(f"{sid}_reflect_on_correct", ep, st["playbook"], fb(ep))
                run.update_counts(tags)
            cur = {}
            if step % official.CURATOR_FREQUENCY == 0:
                cur = run.curate(sid, reflection, ep, step, total)
            st["episodes"] += n
            run.commit(sid, {"cell": cell, "instance": inst, "episodes": n, "first_success": first["success"],
                             "first_score": first["score"], "final_success": correct, "keys": [first["key"]],
                             **cur, "playbook_tokens": approx_tokens(st["playbook"])})
    return _finish(run, "epochs")


def _finish(run, why):
    _atomic(run.root / "final_playbook.txt", run.state["playbook"])
    _atomic(run.root / "offline_done.json", {"stopped_by": why, "episodes": run.state["episodes"]})
    return run.state["playbook"]


def online(run: ACERun, problems, split="test"):
    """problems: [(cell, instance)] of one new task in the fixed order; one episode each."""
    total = len(problems)
    for i, (cell, inst) in enumerate(problems):
        pid = f"p{i:03d}"
        if pid in run.state["done"]:
            continue
        st = run.state
        run.remember(inst)
        ep = run.play(pid, cell, split, inst, st["playbook"], visible_only=True)
        reflection, tags = run.reflect(f"{pid}_reflect", ep, st["playbook"], TEST_FEEDBACK)
        run.update_counts(tags)
        cur = run.curate(f"{pid}_curate", reflection, ep, i + 1, total)
        st["episodes"] += 1
        run.commit(pid, {"index": i, "cell": cell, "instance": inst, "key": ep["key"], **cur,
                         "playbook_tokens": approx_tokens(st["playbook"])})
    _atomic(run.root / "online_done.json", {"problems": total})
    return run.state["playbook"]
