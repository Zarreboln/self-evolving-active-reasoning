"""Learning Meta-Skills in experiment B: the Mac-side control loop (entry point).

The Builder runs as Claude Code sessions in the Docker sandbox on the Mac (agents.py); episodes are
played on ORCD through an Executor (executor.py). Every step writes its result before the next one
starts, and a restarted driver skips finished steps, so a Mac restart, a dead session or a preempted
GPU job costs at most the step in flight.

Phase 1, the bank (paper §3.1, eq. (2)), on the 7 development tasks:
    bank S_0 = []
    for round r = 1..8 (= development passes; paper: 2; experiment B: the equal budget of 8 x 140):
        for each development task x (fixed order):
            H = Builder(H0, card(x), full current bank)            build + <= 2 interface repairs
            play H on x's 20 search problems                       20 episodes; the round has 140
            S = Builder.reflect(S, H, the 20 public episodes)      keep / add / revise, at most one
        early stop: 3 rounds in a row with no gain beyond noise (orchestrate.stalled, as every method)
    freeze S  -> bank/final.json (checked: no names, no answers)
Phase 2, test (paper §3.2, eq. (3)), on each of the 7 new tasks, bank frozen, full-bank mode:
    H* = Builder(H0, card(first problem), S*)   (or one per problem: --build-per problem)
    play H* once on each problem, in the fixed order; nothing adapts between problems

    python -m mb.baselines.metaskill.driver bank --run b1 --remote mailbox:~/mailbox --cards ~/ms_cards \
        --answers ~/ms_cards/answers.json
    python -m mb.baselines.metaskill.driver test --run b1 --remote mailbox:~/mailbox --cards ~/ms_cards \
        --order ~/ms_cards/test_order.json
    python -m mb.baselines.metaskill.driver status --run b1
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mb.baselines.metaskill import assemble, rules, workspaces
from mb.baselines.metaskill.bank import ReflectionError, apply_decision
from mb.baselines.metaskill.config import (DEV_CELLS, MAX_REPAIRS, PATIENCE, ROUNDS, RUNS, SEARCH_N,
                                           TEST_PROBLEMS, TEST_TASKS)
from mb.baselines.metaskill.executor import FIRST_SEARCH, LocalExecutor, mean_score
from mb.baselines.metaskill.mock_game import SCENARIOS, MockChat, MockEnv

UNSEEN = ["mastermind_basic", "alfworld_pick_and_place_simple", "twentyq_basic_pooled", "minesweeper_basic"]


def jload(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p: Path, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(p.suffix + ".tmp").write_text(json.dumps(v, indent=1))
    p.with_suffix(p.suffix + ".tmp").replace(p)


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def mock_problems(harness: Path) -> list[str]:
    """The assembled file through the real child.py against the mock game (what the broker will do)."""
    out = []
    for i, sc in enumerate(SCENARIOS):
        ex = LocalExecutor(lambda c, inst, sc=sc: MockEnv(**sc), lambda c: MockChat(), None)
        from mb.baselines.metaskill.executor import run_child
        r = run_child(harness, ex.make_env("mock", "mock"), ex.make_chat("mock"), timeout=120)
        if r["end"] != "end" or r.get("error"):
            out.append(f"mock {i}: {r['end']}: {(r.get('error') or '')[-400:]}")
        elif "runner stop" in r.get("harness_stderr", ""):
            out.append(f"mock {i}: " + r["harness_stderr"].strip().splitlines()[-1][-400:])
    return out


class Method:
    def __init__(self, run: str, executor, session, answers: dict | None = None, root: Path = RUNS):
        self.dir = Path(root) / f"metaskill_{run}"
        self.ex, self.session, self.answers = executor, session, answers
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- one Builder build with interface repairs (paper B.2) ----
    def build(self, sd: Path, card: dict, skills: list[dict], scope: str) -> Path | None:
        rec = jload(sd / "build.json")
        if rec is not None:
            return sd / "harness.py" if rec["ok"] else None
        prog = jload(sd / "build_progress.json", {"attempts": []})
        ws = sd / "ws"
        while True:
            k = len(prog["attempts"])
            if k > MAX_REPAIRS:
                break
            if k == 0:
                workspaces.fill_build(ws, card, skills, scope)
            else:
                workspaces.fill_build(ws, card, skills, scope, repair="; ".join(prog["attempts"][-1]["problems"]))
            s = self.session("build" if k == 0 else "build_repair", ws, self.dir / "private" / "sessions" / sd.relative_to(self.dir) / f"build{k}.jsonl")
            comp = (ws / "harness" / "components.py").read_text()
            probs = assemble.harness_problems(comp)
            if not probs:
                (sd / "harness.py").write_text(assemble.assemble(comp))
                probs = mock_problems(sd / "harness.py")
            prog["attempts"].append({"session": s, "problems": probs})
            jsave(sd / "build_progress.json", prog)
            self.cost(s)
            if not probs:
                (sd / "components.py").write_text(comp)
                jsave(sd / "build.json", {"ok": True, "repairs": k})
                return sd / "harness.py"
        jsave(sd / "build.json", {"ok": False, "repairs": MAX_REPAIRS, "problems": prog["attempts"][-1]["problems"]})
        return None

    # ---- one reflection: keep / add / revise (paper §3.1, released reflect_skill) ----
    def reflect(self, sd: Path, skills, comp: str, eps: list[dict], ids: list[str]):
        ws = sd / "reflect_ws"
        error, errors = None, []
        for k in range(2):                      # one retry of an invalid decision (ours; released: raises)
            workspaces.fill_reflect(ws, skills, comp, eps, ids, repair=error)
            s = self.session("reflect" if k == 0 else "reflect_repair", ws,
                             self.dir / "private" / "sessions" / sd.relative_to(self.dir) / f"reflect{k}.jsonl")
            self.cost(s)
            try:
                decision = json.loads((ws / "decision.json").read_text())
                new, old, op = apply_decision(skills, decision, ids, self.answers)
                return new, {"operation": op, "decision": decision, "superseded": old, "retries": k,
                             "rejected": errors}
            except (OSError, json.JSONDecodeError, ReflectionError) as e:
                error = f"{type(e).__name__}: {e}"
                errors.append(error)
        return list(skills), {"operation": "keep", "invalid": error, "retries": 1, "rejected": errors}

    def cost(self, s):
        with open(self.dir / "private" / "sessions.jsonl", "a") as f:
            f.write(json.dumps({**s, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")

    # ---- phase 1 ----
    def round_meta(self, r):
        per = {}
        for k, cell in enumerate(DEV_CELLS):
            for i, e in enumerate(jload(self.dir / "rounds" / f"r{r:02d}" / f"t{k}" / "play.json", [])):
                per[f"{cell}/{e.get('instance') or i}"] = {"score": e.get("score") or 0.0}
        sc = [v["score"] for v in per.values()]
        return {"id": f"r{r:02d}", "valid": True, "scores": {"mean_score": statistics.mean(sc) if sc else 0.0,
                                                             "per_episode": per}}

    def bank_phase(self, rounds=ROUNDS, patience=PATIENCE):
        from mb.evolve.orchestrate import stalled
        if (self.dir / "bank" / "final.json").exists():
            log("bank already frozen")
            return jload(self.dir / "bank" / "final.json")
        if self.answers is None:
            log("WARNING: no answer lists: only the name rule is checked on the bank")
        skills = []
        version = 0
        early = None
        for r in range(1, rounds + 1):
            done_rounds = [self.round_meta(q) for q in range(1, r)]
            if r > 1 and stalled(type("A", (), {"all": lambda self_: done_rounds})(), patience):
                early = r - 1
                log(f"early stop after round {early}: {patience} rounds in a row without a gain beyond noise")
                break
            for k, cell in enumerate(DEV_CELLS):
                sd = self.dir / "rounds" / f"r{r:02d}" / f"t{k}"
                if (sd / "done.json").exists():
                    skills = jload(sd / "bank_after.json")
                    version = jload(sd / "done.json")["bank_version"]
                    continue
                card = self.card(cell, FIRST_SEARCH)
                harness = self.build(sd, card, skills, "task")
                if harness is not None:
                    eps = jload(sd / "play.json")
                    if eps is None:
                        eps = self.ex.play(harness, cell, "search", SEARCH_N, tag=f"r{r:02d}_t{k}")
                        jsave(sd / "play.json", eps)
                    ids = [f"r{r:02d}t{k}e{i:02d}" for i in range(len(eps))]
                    comp = (sd / "components.py").read_text()
                    skills_new, rec = self.reflect(sd, skills, comp, eps, ids)
                else:                            # construction failure: the batch scores zero (paper B.3)
                    eps = [{"instance": f"fail{i}", "score": 0.0, "end": "construction_failure"} for i in range(SEARCH_N)]
                    jsave(sd / "play.json", eps)
                    skills_new, rec = list(skills), {"operation": "keep", "reason": "no harness"}
                if rec["operation"] != "keep":
                    version += 1
                    jsave(self.dir / "bank" / f"v{version:03d}.json", skills_new)
                    if rec.get("superseded"):
                        with open(self.dir / "bank" / "archive.jsonl", "a") as f:   # paper B.1
                            f.write(json.dumps({"version": version, "skill": rec["superseded"]}) + "\n")
                skills = skills_new
                jsave(sd / "reflect.json", rec)
                jsave(sd / "bank_after.json", skills)
                jsave(sd / "done.json", {"bank_version": version, "mean_score": mean_score(eps),
                                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
                log(f"r{r:02d} t{k} {cell}: {mean_score(eps):.3f}, {rec['operation']}, bank {len(skills)} skills")
            jsave(self.dir / "rounds" / f"r{r:02d}" / "summary.json", self.round_meta(r)["scores"]["mean_score"])
        problems = rules.bank_problems(skills, self.answers)
        manifest = {"skills": len(skills), "bank_version": version, "rounds_run": early or rounds,
                    "early_stop_after": early, "answer_check": self.answers is not None,
                    "answer_hits": rules.answer_hits(skills, self.answers), "rule_problems": problems,
                    "round_scores": {f"r{q:02d}": self.round_meta(q)["scores"]["mean_score"]
                                     for q in range(1, (early or rounds) + 1)}}
        jsave(self.dir / "bank" / "manifest.json", manifest)
        if problems:
            raise RuntimeError(f"final bank breaks the shared-store rules: {problems}")
        jsave(self.dir / "bank" / "final.json", skills)
        log(f"bank frozen: {len(skills)} skills")
        return skills

    def card(self, cell, instance):
        p = self.dir / "private" / "cards" / f"{cell.replace(':', '_')}__{str(instance).replace(':', '_').replace('/', '_')}.json"
        c = jload(p)
        if c is None:
            c = self.ex.public_card(cell, instance)
            jsave(p, c)
        return c

    # ---- phase 2 ----
    def test_phase(self, order: dict, build_per="task", tasks=None):
        skills = jload(self.dir / "bank" / "final.json")
        if skills is None:
            raise RuntimeError("no frozen bank: run the bank phase first")
        bad = rules.bank_problems(skills, self.answers)
        if bad:
            raise RuntimeError(f"bank breaks the shared-store rules: {bad}")
        out = {}
        for key in (tasks or TEST_TASKS):
            probs = order[key]
            want = TEST_PROBLEMS.get(key, 50)
            if len(probs) != want:
                raise ValueError(f"{key}: the order file has {len(probs)} problems, the design says {want}")
            if any(p["cell"] not in TEST_TASKS[key] for p in probs):
                raise ValueError(f"{key}: the order file names a cell outside the task")
            td = self.dir / "test" / key
            res = jload(td / "results.json")
            if res is None:
                res = self.test_task(td, probs, skills, build_per)
                jsave(td / "results.json", res)
            out[key] = mean_score(res) if all("score" in e for e in res) else None
            log(f"test {key}: {out[key]}")
        groups = {"unseen": [out[k] for k in out if k in UNSEEN and out[k] is not None],
                  "seen": [out[k] for k in out if k not in UNSEEN and out[k] is not None]}
        summ = {"per_task": out, **{g: (statistics.mean(v) if v else None) for g, v in groups.items()},
                "build_per": build_per}
        jsave(self.dir / "test" / "summary.json", summ)
        return summ

    def test_task(self, td: Path, probs, skills, build_per):
        if build_per == "task":
            harness = self.build(td / "build", self.card(probs[0]["cell"], probs[0]["instance"]), skills, "task")
            if harness is None:                  # paper B.3: no executable harness -> zero
                return [{**p, "score": 0.0, "end": "construction_failure"} for p in probs]
            by_cell = {}
            for p in probs:                      # nothing adapts between problems: one batch per cell
                by_cell.setdefault(p["cell"], []).append(p["instance"])
            got = {}
            for cell, ids in by_cell.items():
                f = td / f"play_{cell.replace(':', '_')}.json"
                eps = jload(f)
                if eps is None:
                    eps = self.ex.play(harness, cell, "test", ids, tag=f"test_{td.name}_{cell.replace(':', '_')}")
                    jsave(f, eps)
                if len(eps) == len(ids) and all(e.get("instance") is not None for e in eps):
                    got.update({(cell, e["instance"]): e for e in eps})
                else:                            # batch backend: per-episode results stay on the cluster
                    return [{"summary": e.get("summary")} for e in eps]
            return [{**p, **{k: v for k, v in got[(p["cell"], p["instance"])].items() if k != "trace"}} for p in probs]
        res = []
        for i, p in enumerate(probs):            # one harness per problem, built and played in order
            f = td / f"p{i:03d}.json"
            e = jload(f)
            if e is None:
                harness = self.build(td / f"p{i:03d}", self.card(p["cell"], p["instance"]), skills, "problem")
                if harness is None:
                    e = {**p, "score": 0.0, "end": "construction_failure"}
                else:
                    ep = self.ex.play(harness, p["cell"], "test", [p["instance"]], tag=f"test_{td.name}_p{i:03d}")[0]
                    e = {**p, **{k: v for k, v in ep.items() if k != "trace"}}
                jsave(f, e)
            res.append(e)
        return res


def status(run):
    d = RUNS / f"metaskill_{run}"
    for r in sorted((d / "rounds").glob("r*")):
        steps = [jload(t / "done.json") for t in sorted(r.glob("t*"))]
        print(r.name, [round(s["mean_score"], 3) if s else None for s in steps])
    print("bank:", jload(d / "bank" / "manifest.json"))
    print("test:", jload(d / "test" / "summary.json"))
    s = [json.loads(l) for l in open(d / "private" / "sessions.jsonl")] if (d / "private" / "sessions.jsonl").exists() else []
    tot = {k: sum((x.get("usage") or {}).get(k, 0) or 0 for x in s) for k in ("input", "cache_write", "cache_read", "output")}
    print(f"sessions: {len(s)}, tokens: {tot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "test", "status"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--remote", default="mailbox:~/mailbox", help="mb/evolve/remote.py spec")
    ap.add_argument("--cards", type=Path, help="cards made on the cluster by prep_cluster.py")
    ap.add_argument("--answers", type=Path, help="answers.json from prep_cluster.py (search + selection keys)")
    ap.add_argument("--no-answer-check", action="store_true")
    ap.add_argument("--order", type=Path, help="the fixed test order shared by every method")
    ap.add_argument("--build-per", choices=["task", "problem"], default="task")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--tasks", default="", help="test: only these test tasks (comma-separated), to run tasks in parallel")
    ap.add_argument("--model", default="claude-opus-5-5")
    a = ap.parse_args()
    if a.cmd == "status":
        return status(a.run)
    from mb.baselines.metaskill.agents import ClaudeCodeSession
    from mb.baselines.metaskill.executor import MailboxExecutor
    if not a.answers and not a.no_answer_check:
        ap.error("--answers is required (or pass --no-answer-check)")
    answers = jload(a.answers) if a.answers else None
    m = Method(a.run, MailboxExecutor(a.remote, a.run, a.cards), ClaudeCodeSession(a.model), answers)
    if a.cmd == "bank":
        m.bank_phase(a.rounds)
    else:
        m.test_phase(jload(a.order), a.build_per, tasks=[t for t in a.tasks.split(",") if t] or None)


if __name__ == "__main__":
    main()
