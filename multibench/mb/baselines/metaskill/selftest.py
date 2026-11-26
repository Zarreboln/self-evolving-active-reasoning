"""CPU-only end-to-end test of the Meta-Skills integration: fake games, fake Target, fake Builder.

No model is called: the Builder is a Python function standing in for the Claude Code session, the
Target is a scripted chat, and the games are word-guessing stand-ins with the suite's cell names. Every
harness still runs through the real child.py and passes the real check.py, as on the cluster.

What it checks:
  1. bank phase: 7 tasks x rounds; one build per task per round (with an interface repair), plays only
     on the search split, one keep/add/revise per batch, bank versions and archive;
  2. the shared-store rules: a skill naming a benchmark and a skill copying revealed answers are both
     rejected; no bank version holds a name or a seen answer;
  3. resume: a crash inside a Builder session and a preempted play; the rerun finishes without
     replaying finished batches or rerunning finished sessions;
  4. early stop after 3 rounds without a gain;
  5. test phase: frozen bank, one harness per task (and per problem in the other mode), each problem
     played once, in the fixed order, results in that order; no selection split anywhere;
  6. the official prompts reach the workspaces verbatim; no real session object is ever made.

    python -m mb.baselines.metaskill.selftest
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from mb.baselines.metaskill import agents, driver, official, rules
from mb.baselines.metaskill.agents import FakeSession
from mb.baselines.metaskill.config import DEV_CELLS, TEST_TASKS
from mb.baselines.metaskill.executor import LocalExecutor

WORDS = ["apple", "bread", "crane", "delta", "eagle", "flame", "grape", "house", "ivory", "joker", "knife",
         "lemon", "mango", "nerve", "ocean", "piano", "queen", "river", "stone", "tiger", "ultra", "vivid",
         "whale", "xenon", "yacht", "zebra", "amber", "blaze", "cider", "dwell", "ember", "frost", "gloom",
         "haste", "inlet", "jolly", "karma", "lunar", "maple", "noble"]


def instances_of(cell, split, n):
    k = sum(map(ord, cell)) % 7
    words = WORDS[k:] + WORDS[:k]
    sl = {"search": words[:10], "selection": words[10:20], "test": words[20:]}[split]
    return sl if n is None else sl[:n]


class FakeGame:
    def __init__(self, cell, secret):
        self.cell, self.secret = cell, secret
        self.instructions = (f"You are playing Wordle ({cell}). Guess the secret five-letter word; write the guess "
                             f"in square brackets, e.g. [crane]. You have 3 guesses.")
        self.observation = "Make your first guess."
        self.max_steps, self.reads_full_reply = 3, True
        self.steps, self.done, self.won = 0, False, False

    def admissible(self):
        return None

    def act(self, action):
        if self.done:
            raise RuntimeError("over")
        self.steps += 1
        m = re.findall(r"\[([a-z]+)\]", action.lower())
        if m and m[-1] == self.secret:
            self.done = self.won = True
            return "Congratulations, you found the word."
        if self.steps >= self.max_steps:
            self.done = True
            return f"Out of guesses. The secret word was {self.secret}."
        return f"Not the word. Guess {self.steps} of 3 used."

    def result(self):
        return {"instance_id": self.secret[:2] + "x", "score": 1.0 if self.won else 0.0, "success": self.won,
                "steps": self.steps}


class FakeTarget:
    """Calls each listed harness tool once, then guesses; wins when the secret is 'apple' or 'crane'."""

    def __init__(self):
        self.called = set()

    def __call__(self, messages, temperature=None, max_tokens=None, **kw):
        system = messages[0]["content"]
        for m in re.finditer(r"^- ([A-Za-z][A-Za-z0-9_]*) (\{.*\}): ", system, re.M):
            if m.group(1) not in self.called:
                self.called.add(m.group(1))
                args = {"path": "notes.txt"} if m.group(1) == "workbench_read" else (
                    {"path": "a.txt", "content": "x"} if m.group(1) == "workbench_write" else {})
                return f"Tool: {m.group(1)} {json.dumps(args)}"
        n = sum(1 for x in messages if x["role"] == "assistant" and x["content"].startswith("Action"))
        return "Action: [%s]" % ["apple", "crane", "zzzzz"][min(n, 2) % 3]


GOOD = '''import json

INSTRUCTION = "Think briefly, then end with an Action line. Bank size: %d."


def mem_write(event, records):
    if event.get("type") == "primitive_result" and event.get("tool") == "act":
        return {"upsert": [{"id": "obs" + str(len(records)), "text": str(event["result"]["observation"])[:200]}],
                "delete": []}
    return {"upsert": [], "delete": []}


def mem_retrieve(query, records):
    return [r["id"] for r in records[-3:]]


def ctx_select(groups, retrieved):
    return [g["id"] for g in groups[-6:]]


def ctl_on_event(event, snapshot):
    state = dict(snapshot["controller"])
    state["events"] = state.get("events", 0) + 1
    return {"state": state, "phase": "play"}


def ver_before(action, snapshot):
    if not action.strip():
        return {"allow": False, "feedback": "Empty action; reply with an Action line."}
    return {"allow": True, "feedback": ""}


def ver_fail(failure, snapshot):
    return {"actions": [], "feedback": "The tool failed; continue."}


def ws_setup(task):
    return {"files": [{"path": "notes.txt", "content": "start"}], "actions": [{"tool": "workbench_list", "arguments": {}}]}


def read_notes(arguments, state, previous):
    if previous is None:
        return {"state": {"asked": 1}, "call": {"tool": "workbench_read", "arguments": {"path": "notes.txt"}}}
    return {"state": state, "return": {"notes": previous["observation"]}}


MEMORY = {"write": mem_write, "retrieve": mem_retrieve}
CONTEXT = {"select_context": ctx_select}
TOOLS = [{"name": "read_notes", "description": "Read the notes file.", "parameters": {"type": "object", "properties": {}},
          "next_action": read_notes}]
CONTROLLER = {"on_event": ctl_on_event}
VERIFICATION = {"before_action": ver_before, "on_failure": ver_fail}
WORKSPACE = {"setup": ws_setup}
'''
BROKEN = GOOD.replace("def mem_retrieve(query, records):", "def mem_retrieve(query):")


class FakeBuilder:
    """Stands in for the Claude Code session. Scripted: the first build of a run is broken (one repair);
    the first reflection names a benchmark, the second copies revealed answers (both must be rejected and
    retried), later ones alternate add / revise / keep."""

    def __init__(self, crash_at=None):
        self.n = {"build": 0, "build_repair": 0, "reflect": 0, "reflect_repair": 0}
        self.crash_at = crash_at
        self.total = 0
        self.seen_tasks = []

    def __call__(self, role, ws: Path):
        self.total += 1
        if self.crash_at and self.total == self.crash_at:
            raise KeyboardInterrupt("simulated crash inside a Builder session")
        self.n[role] += 1
        task = (ws / "TASK.md").read_text()
        self.seen_tasks.append(task)
        if role.startswith("build"):
            assert official.builder_system() in task and official.build_instruction() in task
            skills = json.loads((ws / "skills.json").read_text())
            src = BROKEN if (role == "build" and self.n["build"] == 1) else GOOD
            (ws / "harness" / "components.py").write_text(src.replace("%d", str(len(skills))))
            return
        assert official.reflection_system() in task
        ids = json.loads((ws / "batch.json").read_text())["episode_ids"]
        skills = json.loads((ws / "skills.json").read_text())
        k = self.n["reflect"]
        skill = {"skill_id": f"s{k}_{ids[0]}", "interface_version": "modular-v2.0", "evidence_ids": [ids[0]],
                 "when": "The target repeats a guess that the environment already rejected.",
                 "provide": "A memory of rejected guesses and a gate that blocks repeats.",
                 "use": "The target checks the memory before guessing and keeps choosing the word itself."}
        if role == "reflect" and k == 1:
            skill["when"] = "In Wordle the target repeats guesses."            # names a benchmark
            d = {"operation": "add", "skill": skill}
        elif role == "reflect" and k == 2:
            revealed = re.findall(r"The secret word was (\w+)", "".join(
                (p / "trace.md").read_text() for p in (ws / "episodes").iterdir()))
            words = " ".join(sorted(set(revealed)) + ["apple", "crane", "bread", "delta", "eagle"])
            skill["provide"] = ("Candidate list: " + words)[:700]                # copies answers
            d = {"operation": "add", "skill": skill}
        elif skills and k % 3 == 0:
            old = skills[-1]
            d = {"operation": "revise", "prior_skill_id": old["skill_id"],
                 "skill": {**skill, "skill_id": old["skill_id"],
                           "evidence_ids": sorted(set(old["evidence_ids"]) | {ids[0]})}}
        elif k % 3 == 2:
            d = {"operation": "keep"}
        else:
            d = {"operation": "add", "skill": skill}
        (ws / "decision.json").write_text(json.dumps(d))


class FlakyExecutor(LocalExecutor):
    """Raises once on the given play call (a preempted GPU job that the Mac sees as an error)."""

    def __init__(self, fail_on=None):
        super().__init__(lambda c, i: FakeGame(c, i), lambda c: FakeTarget(), instances_of)
        self.fail_on, self.plays = fail_on, 0

    def play(self, harness_path, cell, split, instances, tag):
        self.plays += 1
        if self.fail_on and self.plays == self.fail_on:
            raise RuntimeError("simulated preemption")
        return super().play(harness_path, cell, split, instances, tag)


def answers_for(n):
    return {c: {"search": instances_of(c, "search", n), "selection": instances_of(c, "selection", None)}
            for c in DEV_CELLS}


def main():
    def no_real(*a, **k):
        raise AssertionError("a real Claude Code session was requested")
    agents.ClaudeCodeSession = no_real
    root = Path(tempfile.mkdtemp(prefix="ms_selftest_"))
    driver.SEARCH_N = 4
    driver.TEST_PROBLEMS = {k: 3 for k in TEST_TASKS}
    answers = answers_for(driver.SEARCH_N)
    ok = []

    # ---- 1-3: bank phase with a crash in a session and a preempted play, then resume ----
    ex = FlakyExecutor(fail_on=5)
    b1 = FakeBuilder(crash_at=9)
    for attempt, (bld, exe) in enumerate([(b1, ex), (FakeBuilder(), ex), (FakeBuilder(), ex)]):
        m = driver.Method("t", exe, FakeSession(bld), answers, root=root)
        try:
            skills = m.bank_phase(rounds=2)
            break
        except (KeyboardInterrupt, RuntimeError) as e:
            print(f"  interrupted ({e}); resuming")
    d = root / "metaskill_t"
    plays = [x for x in ex.log if x[0].startswith("r")]
    assert {x[2] for x in plays} == {"search"}, "development plays must use the search split only"
    per_tag = {}
    for tag, cell, split, inst in plays:
        per_tag.setdefault((tag, cell), []).append(inst)
    for (tag, cell), insts in per_tag.items():
        assert insts == instances_of(cell, "search", driver.SEARCH_N), f"{tag}: replayed or reordered {insts}"
    assert len(per_tag) == 2 * len(DEV_CELLS), len(per_tag)
    ok.append(f"bank: 2 rounds x {len(DEV_CELLS)} tasks, each batch played once on the same search problems "
              f"({len(plays)} episodes), after a crash in a Builder session and a preempted play")
    b0 = json.loads((d / "rounds" / "r01" / "t0" / "build_progress.json").read_text())
    assert len(b0["attempts"]) == 2 and b0["attempts"][0]["problems"] and not b0["attempts"][1]["problems"]
    ok.append("interface repair: the broken first build was repaired once (" + b0["attempts"][0]["problems"][0] + ")")
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("rounds/r*/t*/reflect.json"))]
    rejected = [e for r in recs for e in r.get("rejected", [])]
    assert any("names a game, benchmark or task: wordle" in e for e in rejected), rejected
    assert any("shared-store rule: answers of" in e for e in rejected), rejected
    ops = [r["operation"] for r in recs]
    assert {"add", "revise", "keep"} <= set(ops), ops
    ok.append(f"reflections: {len(recs)} decisions {ops}; {len(rejected)} invalid decisions rejected and "
              "retried: " + " | ".join(sorted({e[:100] for e in rejected})))
    for p in sorted((d / "bank").glob("v*.json")) + [d / "bank" / "final.json"]:
        s = json.loads(p.read_text())
        assert not rules.bank_problems(s, answers), (p, rules.bank_problems(s, answers))
    hits = rules.answer_hits(skills, answers)
    ok.append(f"shared-store rules: no bank version names a benchmark or holds copied answers; final bank "
              f"{len(skills)} skills; answer hits {sum(v['search'] for v in hits.values())} seen / "
              f"{sum(v['selection'] for v in hits.values())} unseen")
    bad = rules.bank_problems([{"when": "In Wordle ...", "provide": "x", "use": "y", "skill_id": "a"}], answers)
    assert bad and "wordle" in bad[0]
    leak = rules.bank_problems([{"when": " ".join(instances_of("wordle:basic5", "search", 4)), "provide": "",
                                 "use": "", "skill_id": "b"}], answers)
    assert leak and "answers" in leak[0], leak
    ok.append("rule unit checks: a name and a list of seen answers are flagged")
    for ws in d.glob("rounds/r*/t*/ws"):
        text = " ".join(p.read_text(errors="ignore") for p in ws.rglob("*") if p.is_file())
        for c in DEV_CELLS:
            for w in instances_of(c, "selection", None) + instances_of(c, "test", None):
                assert f"secret word was {w}" not in text
    ok.append("build workspaces hold no selection or test problem")

    # ---- 4: early stop ----
    ex2 = FlakyExecutor()
    driver.SEARCH_N = 2
    m2 = driver.Method("stop", ex2, FakeSession(FakeBuilder()), answers_for(2), root=root)
    m2.bank_phase(rounds=8)
    man = json.loads((root / "metaskill_stop" / "bank" / "manifest.json").read_text())
    assert man["early_stop_after"] == 4, man
    ok.append(f"early stop: after round {man['early_stop_after']} of 8 (round scores {man['round_scores']})")
    driver.SEARCH_N = 4

    # ---- 5: test phase, fixed order, per task and per problem ----
    order = {}
    for key, cells in TEST_TASKS.items():
        pool = [{"cell": c, "instance": i} for c in cells for i in instances_of(c, "test", None)]
        order[key] = list(reversed(pool))[:3]
    bld = FakeBuilder()
    sess = FakeSession(bld)
    m = driver.Method("t", ex, sess, answers, root=root)
    n_before = len(ex.log)
    summ = m.test_phase(order, "task")
    tl = ex.log[n_before:]
    assert {x[2] for x in tl} == {"test"}
    for key in TEST_TASKS:
        played = [(x[1], x[3]) for x in tl if x[0].startswith(f"test_{key}_")]
        assert sorted(played) == sorted((p["cell"], p["instance"]) for p in order[key]), key
        assert len(set(played)) == len(played)
        res = json.loads((d / "test" / key / "results.json").read_text())
        assert [(r["cell"], r["instance"]) for r in res] == [(p["cell"], p["instance"]) for p in order[key]]
    builds = [c for c in sess.calls if c[0] == "build"]
    assert len(builds) == len(TEST_TASKS), builds
    ok.append(f"test (one harness per task): {len(builds)} builds, {len(tl)} episodes, each problem once, "
              f"results in the fixed order; unseen {summ['unseen']:.3f}, seen {summ['seen']:.3f}")
    n0 = len(ex.log)
    again = m.test_phase(order, "task")
    assert len(ex.log) == n0 and again == summ
    ok.append("test resume: a rerun replays nothing")
    m3 = driver.Method("pp", ex, FakeSession(FakeBuilder()), answers, root=root)
    shutil.copytree(d / "bank", root / "metaskill_pp" / "bank")
    n0 = len(ex.log)
    m3.test_phase(order, "problem", tasks=["twentyq_basic_pooled"])
    tl = ex.log[n0:]
    assert [(x[1], x[3]) for x in tl] == [(p["cell"], p["instance"]) for p in order["twentyq_basic_pooled"]]
    assert all(len([y for y in tl if y[0] == x[0]]) == 1 for x in tl)
    ok.append("test (one harness per problem): problems built and played one at a time in the fixed order")
    assert not any(x[2] == "selection" for x in ex.log)
    ok.append("no episode on the selection split; no real session was made")
    # ---- the mailbox backend's request and export parsing, with a fake remote ----
    from mb.baselines.metaskill.executor import MailboxExecutor

    class FakeRemote:
        def __init__(self):
            self.sent = []

        def submit(self, key, harness, params):
            self.sent.append((key, params))

        def result(self, key):
            return {"summary": {"infra_errors": 0, "mean_score": 0.5}}

        def fetch_export(self, key, dest):
            for i in range(2):
                e = Path(dest) / "episodes" / f"id{i}"
                e.mkdir(parents=True)
                (e / "result.json").write_text(json.dumps({"instance_id": f"id{i}", "score": i, "success": bool(i),
                                                           "steps": 3, "end": "end", "error": None, "model_calls": 4}))
                (e / "trace.md").write_text("# episode")
    mx = MailboxExecutor.__new__(MailboxExecutor)
    mx.remote, mx.run, mx.cards, mx.poll = FakeRemote(), "t", root, 0
    eps = mx.play(Path("h.py"), "wordle:basic5", "search", 20, "r01_t0")
    assert [e["score"] for e in eps] == [0, 1] and eps[0]["trace"] == "# episode"
    assert mx.remote.sent[0] == ("metaskill/t/r01_t0/search", {"cell": "wordle:basic5", "split": "search",
                                                              "export": True, "smoke": False, "n": 20})
    from mb.evolve import remote as R
    played = []

    def fake_play(box, key, harness, cell, split, instances, visible_only=True, **kw):
        played.append((key, cell, split, list(instances), visible_only))
        d = root / "play_results" / key
        (d / "visible").mkdir(parents=True, exist_ok=True)
        (d / "private.json").write_text(json.dumps({"infra_errors": 0, "per_episode": {
            i: {"score": float(j), "success": bool(j), "steps": 2, "end": "end", "error": None, "model_calls": 2}
            for j, i in enumerate(instances)}}))
        return d / "visible"
    real, R.play = R.play, fake_play
    try:
        eps = mx.play(Path("h.py"), "mastermind:basic", "test", ["a", "b"], "test_x")
    finally:
        R.play = real
    assert played == [("metaskill/t/test_x/test", "mastermind:basic", "test", ["a", "b"], True)]
    assert [(e["instance"], e["score"]) for e in eps] == [("a", 0.0), ("b", 1.0)] and "trace" not in eps[0]
    ok.append("mailbox backend: search request (first n, export) via the batch driver; test list via the "
              "worker, exactly those problems, traces withheld")
    print("\n".join("ok  " + x for x in ok))
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
