"""Our method, version 3 (user, 2026-11-27): every task keeps its own best harness; shared code is distilled
from those harnesses into a layered graph (a soft hierarchy) of modules and workflows, used to start new tasks.

Parts
  base agent   TextArena's standard agent (seeds/textarena_standard.py): observe, call the model, act. Fixed,
               the same start as every other method; not a node.
  leaves       one complete harness per task (code + description), its own. Changed only by its own edits.
  nodes        shared code distilled from the leaves by a review, each = code + a description:
                 module    a small reusable piece of functionality (e.g. keep a compact record of past turns)
                 workflow  how a family of tasks puts modules together
               A node lists the tasks it was distilled from; per task, what part of the functionality it implements,
               evidence (names of its functions, or verbatim excerpts of its harness) and how it differs from the
               node. A link to a parent node says how the node builds on that parent. All of these are required.
               Nodes do not run inside the tasks' harnesses: they are what new tasks (and test time) start from.

For each task, in a fixed random order (seed "exp_b/incremental/0"):
  1. probe: 3 search problems with the base agent; the code-editing model writes a task description and the
     required capabilities, which are embedded
  2. retrieve: similar tasks and workflow nodes, top-k by similarity, all module
     nodes with their match to the required capabilities ("hard": one path; "flat": nothing; "retrieve": the
     similar tasks only, no nodes and no review)
  3. compose: the model builds the task's first harness and its description from what was retrieved, adapting
     only what is needed to run on this task; no new optimisation. Skipped when nothing is retrieved
  4. start: 20 search problems (the transfer measurement); on the 3 probe problems the start is compared with
     the base agent, and the result is recorded with each node used (node_usage)
  5. up to 5 edits of the task's own harness, in Meta-Harness's own workspace (its TASK template, the archive of
     this task's candidates with code, proposals, scores and traces), each kept if its mean rises; early stop
     after 3 in a row not kept. Every new harness is replayed on recorded episodes first; a crash goes back to
     the model until it runs
  6. review (from the 2nd task on, not "flat"): the model reads every task's harness and the graph, and may add,
     change, merge, split or delete nodes and links. No task's harness changes, so nothing is re-scored. Nodes
     failing the checks above (valid code, a description, at least 2 members with evidence) are not kept.
After the last task: the versions holding every task are ranked by mean search score; the top 3 go to the
selection split, as for every method.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

from mb.evolve import remote, soft_tree2 as st2
from mb.evolve.check import problems
from mb.evolve.soft_tree import HERE, dist, embed, paired
from mb.evolve.orchestrate import CHAT_CAP, MODEL

RUNS = Path.home() / "evo_b_runs"
BASE_AGENT = HERE / "seeds" / "textarena_standard.py"
SIGMA, K_NODES, K_TASKS = 0.3, 3, 3          # top 3 of each, with their similarities; the model decides (user, 2026-11-27)
EDITS = 5
MAX_FIXES = 10          # repair sessions until the harness runs (user, 2026-11-27: fix until it passes; a safety cap)
ABLATION = ""
ORDER_SEED = st2.ORDER_SEED
NODE_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def defined_functions(src):
    try:
        return {n.name for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    except SyntaxError:
        return set()


def normalise(text):
    return " ".join(text.split())


def evidence_problems(names, quotes, defined, code):
    """A member's evidence that it really has the node's functionality: names of its own functions, or verbatim
    excerpts of its harness (code, prompt text or settings; whitespace ignored, at least 40 characters each)."""
    if not names and not quotes:
        return ["no evidence (functions or excerpts of its harness)"]
    out = [f"function {q!r} is not defined in its harness" for q in names if q not in defined]
    body = normalise(code)
    out += [f"excerpt shorter than 40 characters: {q[:30]!r}" for q in quotes if len(normalise(q)) < 40]
    out += [f"excerpt not found verbatim in its harness: {q[:40]!r}" for q in quotes
            if len(normalise(q)) >= 40 and normalise(q) not in body]
    return out


def d_problems(spec, code):
    """Checks of the D fields of one node; returns (problems, fields to store)."""
    why, out = [], {}
    defined = defined_functions(code)
    if spec.get("kind") == "workflow":
        sl = spec.get("slots")
        if not isinstance(sl, dict) or not sl:
            why.append("a skeleton needs `slots`")
        else:
            out["slots"] = {}
            for k, v in sl.items():
                v = v if isinstance(v, dict) else {}
                if v.get("role") not in ROLES:
                    why.append(f"slot {k}: role must be one of {', '.join(ROLES)}")
                out["slots"][str(k)] = {"role": v.get("role"), "default": v.get("default") or None,
                                        "what": str(v.get("what", ""))[:200]}
    else:
        if spec.get("role") not in ROLES:
            why.append(f"`role` must be one of {', '.join(ROLES)}")
        if spec.get("entry") not in defined:
            why.append("`entry` must name a function the node defines")
        pr = spec.get("params", {})
        if not isinstance(pr, dict):
            why.append("`params` must be an object")
            pr = {}
        for k, v in pr.items():
            if not isinstance(v, dict) or "type" not in v or "default" not in v:
                why.append(f"param {k}: needs `type` and `default`")
        out.update(role=spec.get("role"), entry=spec.get("entry"), params=pr)
    return why, out


def config_problems(cf, nodes):
    """A game's config must point at kept nodes: a skeleton, modules per slot, parameters of those modules."""
    why = []
    if not isinstance(cf, dict):
        return ["not an object"]
    wf = cf.get("workflow")
    if wf is not None and (wf not in nodes or nodes[wf].get("kind") != "workflow"):
        why.append(f"workflow {wf!r} is not a kept skeleton")
    for sl, m in (cf.get("slots") or {}).items():
        if m != "own" and m not in nodes:
            why.append(f"slot {sl}: {m!r} is not a kept module")
    for m, pv in (cf.get("params") or {}).items():
        if m not in nodes or not isinstance(pv, dict):
            why.append(f"params for {m!r}: not a kept module")
        elif set(pv) - set(nodes[m].get("params", {})):
            why.append(f"params for {m}: unknown {sorted(set(pv) - set(nodes[m].get('params', {})))}")
    return why


def node_problems(code):
    """Static rules for a node: a module of functions (no run_episode needed)."""
    out = [p for p in problems(code) if "run_episode" not in p]
    if not defined_functions(code):
        out.append("defines no function")
    return out


class Run(st2.Run):
    """Reuses soft_tree2.Run's proposer, records, evidence and selection plumbing; state is one graph.json."""

    # ----- state -----
    def versions(self):
        return sorted(p for p in (self.dir / "state").iterdir() if (p / "graph.json").exists())

    def load(self, v):
        return json.loads((v / "graph.json").read_text()), json.loads((v / "meta.json").read_text())

    def save_state(self, g, meta):
        d = self.dir / "state" / f"v{len(self.versions()):03d}"
        d.mkdir(parents=True, exist_ok=True)
        meta["units"] = self.units
        (d / "graph.json").write_text(json.dumps(g, indent=1))
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    # ----- evaluation (each task runs its own harness) -----
    def evaluate(self, g, cells, tag, n=None, srcs=None):
        folder = self.dir / "tmp" / tag
        folder.mkdir(parents=True, exist_ok=True)
        hm = {}
        for c in cells:
            src = (srcs or {}).get(c) or g["leaves"].get(c, {}).get("code") or BASE_AGENT.read_text()
            bad = problems(src)
            if bad:
                return None, "; ".join(bad)[:300]
            p = folder / f"{c.replace(':', '_')}.py"
            p.write_text(src)
            hm[c] = p
            lb = self.dir / "private" / f"leak_baseline_{c.replace(':', '_')}.py"
            if lb.exists():     # code written before the model saw this task: its words are not answers
                hm["leak_baseline_" + c] = lb
        n = n or self.a.n
        key = f"{self.a.run}/{tag}/search"
        t0 = time.time()
        self.box.submit(key, hm, {"cell": ",".join(cells), "split": "search", "n": n, "export": True, "smoke": False})
        while (r := self.box.result(key)) is None:
            time.sleep(30)
        self.timed("evaluation", tag, t0)
        if "rejected" in r:
            return None, "rejected on the cluster: " + "; ".join(r["rejected"])[:300]
        dest = self.dir / "exports" / tag
        self.box.fetch_export(key, dest)
        self.units += len(cells) * n / self.a.n
        out = {}
        for c in cells:
            f = (dest / "cells" / c.replace(":", "_") / "scores.json") if len(cells) > 1 else dest / "scores.json"
            out[c] = json.loads(f.read_text())
        return out, None

    def timed(self, kind, tag, t0):
        t1 = time.time()
        with (self.dir / "private" / "times.jsonl").open("a") as f:
            f.write(json.dumps({"kind": kind, "tag": tag, "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                                "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1)),
                                "seconds": round(t1 - t0, 1)}) + "\n")

    def propose(self, ws, tag):
        """soft_tree2's proposer call, timed (user, 2026-11-27: record time as well as tokens)."""
        if (self.dir / "private" / f"{tag}_proposed.json").exists() and ws.exists():
            return super().propose(ws, tag)
        t0 = time.time()
        rc = super().propose(ws, tag)
        self.timed("opus", tag, t0)
        return rc

    def record(self, tag, rec):
        rec.setdefault("finished_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        super().record(tag, rec)

    def ws(self, tag):
        ws = self.dir / "workspaces" / tag
        shutil.rmtree(ws, ignore_errors=True)
        ws.mkdir(parents=True)
        for f in ("harness_lib.py", "mock_env.py"):
            shutil.copy(HERE / "ws_template" / f, ws / f)
        shutil.copy(HERE / "check.py", ws / "check_harness.py")
        shutil.copy(HERE / "ws_template" / "replay_test.py", ws / "replay_test.py")
        return ws

    def crash_report(self, code, traces):
        """'' if `code` passes the static rules and runs on the recorded traces (replay_test.py), else the
        report: the static problems, or the crash output."""
        bad = problems(code)
        if bad:
            return "Static check failed:\n" + "\n".join(bad)
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for f in ("harness_lib.py", "replay_test.py"):
                shutil.copy(HERE / "ws_template" / f, d / f)
            (d / "harness.py").write_text(code)
            for k, t in enumerate(traces[:3]):
                (d / "t" / str(k)).mkdir(parents=True)
                shutil.copy(t, d / "t" / str(k) / "trace.md")
            args = [sys.executable, "-I", "replay_test.py", "harness.py"] + [f"t/{k}/trace.md" for k in range(len(traces[:3]))]
            if not traces:
                return ""
            try:
                r = subprocess.run(args, cwd=d, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                return "The harness did not finish replaying a recorded episode within 300 s (an endless loop?)."
            return "" if r.returncode == 0 else (r.stdout + r.stderr)[-4000:]

    def until_it_runs(self, tag, c, episodes, code, doc):
        """Repair sessions until the harness runs on this game's recorded episodes (`episodes`: a folder of
        episodes/<id>/trace.md) (user, 2026-11-27). Returns (code, doc, fixes), or (None, None, fixes) if it
        still crashes after MAX_FIXES."""
        traces = sorted(str(p) for p in Path(episodes).glob("*/trace.md")) if episodes else []
        fixes = 0
        while (report := self.crash_report(code, traces)):
            if fixes >= MAX_FIXES:
                print(f"{tag}: still crashing after {fixes} repairs", flush=True)
                return None, None, fixes
            fixes += 1
            ftag = f"{tag}_fix{fixes}"
            fws = self.dir / "workspaces" / ftag
            if self.fresh(ftag):
                fws = self.ws(ftag)
                (fws / "harness.py").write_text(code)
                (fws / "harness.md").write_text(doc or "")
                (fws / "crash.txt").write_text(report)
                if episodes and Path(episodes).exists():
                    shutil.copytree(episodes, fws / "evidence" / self.tid(c) / "episodes")
                (fws / "TASK.md").write_text(TASK_FIX.format(tid=self.tid(c)))
            self.propose(fws, ftag)
            code = (fws / "harness.py").read_text() if (fws / "harness.py").exists() else code
            doc = (fws / "harness.md").read_text().strip() if (fws / "harness.md").exists() else doc
        if fixes:
            print(f"{tag}: runs after {fixes} repair(s)", flush=True)
        return code, doc, fixes

    def fresh(self, tag):
        """True if the proposer session of `tag` has not run yet (the workspace must be built)."""
        return not ((self.dir / "private" / f"{tag}_proposed.json").exists() and (self.dir / "workspaces" / tag).exists())

    # ----- the graph -----
    @staticmethod
    def sim(a, b):
        """Cosine similarity of two normalised embedding vectors."""
        return round(1 - dist(a, b) ** 2 / 2, 3)

    def retrieve(self, g, c):
        """What a new task is built from (user, 2026-11-27):
        - similar earlier tasks, by description similarity (top K_TASKS; "hard": 1);
        - workflow nodes, by the similarity of their nearest member to the task (top K_NODES; "hard": 1);
        - module nodes: all of them (short and general: a single similarity would bury the most shared ones), each
          with its match to the task's required capabilities, the tasks using it and their similarity, and whether
          a retrieved workflow builds on it; "hard": only the modules on the retrieved workflow's path.
        "flat": nothing. "retrieve": the similar tasks only (their full code), no nodes."""
        if ABLATION == "flat":
            return {}, {}, {}
        v = g["vecs"][c]
        tasks = sorted(((x, self.sim(v, g["vecs"][x])) for x in g["order"]), key=lambda t: -t[1])
        tasks = dict(tasks[:1 if ABLATION == "hard" else K_TASKS])
        if ABLATION == "retrieve":              # retrieval without the hierarchy (user, 2026-11-27)
            return tasks, {}, {}
        workflows = {}                          # (an affinity weight was dropped, 2026-11-27: never read again)
        for n, nd in g["nodes"].items():
            if nd.get("kind") == "workflow":
                workflows[n] = max(self.sim(v, g["vecs"][x]) for x in nd["members"])
        workflows = dict(sorted(workflows.items(), key=lambda t: -t[1])[:1 if ABLATION == "hard" else K_NODES])
        def up(n, seen):
            for p in g["nodes"].get(n, {}).get("parents", []):
                if p in g["nodes"] and p not in seen:
                    seen.add(p)
                    up(p, seen)
            return seen
        on_path = set().union(*(up(n, set()) for n in workflows)) if workflows else set()
        modules = {}
        for n, nd in g["nodes"].items():
            if nd.get("kind") == "workflow" or (ABLATION == "hard" and n not in on_path):
                continue
            modules[n] = {"matches_capabilities": self.capability_matches(g, c, nd),
                      "used_by": {self.tid(x): self.sim(v, g["vecs"][x]) for x in nd["members"]},
                      "games_using_it": len(nd["members"]),
                      "used_by_retrieved_workflows": sorted(m for m in workflows if n in up(m, set()))}
        return tasks, workflows, modules

    def capability_matches(self, g, c, nd):
        """Each required capability of the new task with its similarity to this module's description (retrieval by
        sub-goals, as in HyperSkill; user 2026-11-27): direct evidence, unlike similarity through the games using it."""
        if not nd.get("desc_vec"):
            return {}
        return dict(sorted(((x["capability"], self.sim(x["vec"], nd["desc_vec"])) for x in g.get("capabilities", {}).get(c, [])),
                           key=lambda t: -t[1]))

    def graph_md(self, g, meta):
        L = ["# Shared code graph", "", "Nodes are code distilled from the games' own harnesses. `module`: a small",
             "reusable piece; `workflow`: modules put together for a family of games.", ""]
        for n, nd in g["nodes"].items():
            mem = ", ".join(f"{self.tid(x)}" for x in nd.get("members", {}))
            L.append(f"- `{n}` ({nd.get('kind')}): {nd.get('desc', '')} | parents {nd.get('parents', [])} | "
                     f"distilled from {mem}")
        if not g["nodes"]:
            L.append("(no shared node yet)")
        L += ["", "## Games", ""]
        L += [f"- {self.tid(c)}: {g['descriptions'].get(c, '')} (search score {meta['scores'].get(c)})" for c in g["order"]]
        return "\n".join(L) + "\n"

    # ----- the loop -----
    def loop(self):
        vs = self.versions()
        if vs:
            g, meta = self.load(vs[-1])
        else:
            g = {"nodes": {}, "leaves": {}, "vecs": {}, "descriptions": {}, "order": [], "affinity": {}}
            meta = {"scores": {}, "per_episode": {}, "last_export": {}, "start": {}, "used": {}, "leaf_cand": {}, "units": 0}
            self.save_state(g, meta)
        self.units = meta.get("units", 0)
        for k, c in enumerate(self.order):
            if self.done(f"t{k + 1}_review"):
                continue
            for x in g["order"]:               # a harness imported without a description gets one
                if not (g["leaves"][x].get("doc") or "").strip():
                    g = self.document(x, g, meta)
            g, meta = self.add_task(k + 1, c, g, meta)
        print("all tasks added; units used:", self.units, flush=True)

    def add_task(self, k, c, g, meta):
        tp = f"t{k}"
        slug = c.replace(":", "_")
        if c not in g["order"]:
            # 1 probe with the base agent; task description and required capabilities
            probe, err = self.evaluate(g, [c], f"{tp}_probe", n=3, srcs={c: BASE_AGENT.read_text()})
            if probe is None:
                raise RuntimeError(f"{tp}_probe failed: {err}")
            desc, caps = self.describe(c, f"{tp}_probe")
            vecs = embed([desc] + caps)
            g["vecs"][c], g["descriptions"][c] = vecs[0], desc
            if PROFILE:
                g.setdefault("profiles", {})[c] = self.profile(f"{tp}_probe")
            g.setdefault("capabilities", {})[c] = [{"capability": t, "vec": v} for t, v in zip(caps, vecs[1:])]
            if c in IMPORT:                  # cold start (task k <= K): the task's own evolved harness, no transfer
                g, meta = self.import_leaf(tp, c, g, meta)
                if ABLATION not in ("flat", "retrieve") and len(g["order"]) >= 2:
                    g, meta = self.review(f"{tp}_review", c, g, meta)
                else:
                    self.record(f"{tp}_review", {"outcome": "skipped (first task, flat or retrieve)"})
                return g, meta
            # 2-3 retrieve and compose
            tasks, workflows, modules = self.retrieve(g, c)
            code, used, doc = BASE_AGENT.read_text(), {}, BASE_AGENT_DOC
            if tasks or workflows or modules:
                code, used, doc = self.compose(f"{tp}_compose", c, g, meta, tasks, workflows, modules)
            else:
                (self.dir / "private" / f"leak_baseline_{slug}.py").write_text(BASE_AGENT.read_text())
            g["order"].append(c)
            g["leaves"][c] = {"code": code, "doc": doc}
            # 4 start
            start, err = self.evaluate(g, [c], f"{tp}_start")
            if start is None:
                raise RuntimeError(f"{tp}_start failed: {err}")
            meta["scores"][c] = meta["start"][c] = start[c]["mean_score"]
            meta["per_episode"][c] = start[c]["per_episode"]
            meta["last_export"][c] = f"{tp}_start"
            meta["used"][c] = used
            meta.setdefault("leaf_cand", {})[c] = "c000"
            self.arc_add(c, "c000", None, code, START_NOTE.format(how=("composed from earlier games' code: " + json.dumps(
                sorted(used))) if used or tasks else "the base agent"), export_tag=f"{tp}_start")
            us = meta.setdefault("node_usage", {})
            for n in list(workflows) + list(modules):
                us.setdefault(n, {"retrieved": 0, "chosen": 0, "start_vs_base_agent": []})["retrieved"] += 1
            pw, pl = paired({i: e for i, e in start[c]["per_episode"].items() if i in probe[c]["per_episode"]},
                            probe[c]["per_episode"])
            for n in used:                     # evidence about the nodes: did starting from them beat the base agent?
                meta["node_usage"][n]["chosen"] += 1
                meta["node_usage"][n]["start_vs_base_agent"].append(
                    {"game": self.tid(c), "wins": pw, "losses": pl, "verdict": "better" if pw > pl else "worse" if pl > pw else "same"})
            self.save_state(g, meta)
            self.record(f"{tp}_start", {"task": c, "similar_tasks": tasks, "workflows": workflows,
                                        "modules": sorted(modules), "used": used,
                                        "probe_vs_start_on_probe_problems": [pw, pl],
                                        "start_score": meta["scores"][c], "outcome": f"joined; start {meta['scores'][c]}"})
        # 5 own edits
        for e in range(1, EDITS + 1):
            tag = f"{tp}_e{e}"
            if self.done(tag):
                continue
            if STOP_AT_FULL:
                if meta["scores"][c] >= 1.0:
                    self.record(tag, {"outcome": "skipped: full score reached"})
                    continue
                g, meta = self.edit(tag, c, g, meta)
                continue
            last3 = [self.done(f"{tp}_e{j}") for j in range(e - 3, e) if j >= 1]
            if len(last3) == 3 and all(r and not str(r.get("outcome", "")).startswith("kept") for r in last3):
                self.record(tag, {"outcome": "skipped: early stop (3 edits in a row not kept, as every method)"})
                continue
            g, meta = self.edit(tag, c, g, meta)
        # 6 review
        if ABLATION not in ("flat", "retrieve") and len(g["order"]) >= 2:
            g, meta = self.review(f"{tp}_review", c, g, meta)
        else:
            self.record(f"{tp}_review", {"outcome": "skipped (first task, flat or retrieve)"})
        return g, meta

    def import_leaf(self, tp, c, g, meta):
        """Cold start (user, 2026-11-27): the task joins with the harness its own evolution chose (a Meta-Harness
        run: archive/<cand>/ with per-task scores and traces under cells/<task>/). Every candidate of that run is
        copied into this task's archive, so the review's history summary shows how the harness was evolved."""
        run, chosen = IMPORT[c]
        slug = c.replace(":", "_")
        src = Path(run).expanduser() / "archive"
        for d in sorted(p for p in src.iterdir() if (p / "meta.json").exists()):
            m = json.loads((d / "meta.json").read_text())
            cell = d / "cells" / slug
            if not (d / "cells").exists():   # a run on this task alone keeps its scores and traces at the top
                cell = d
            sc = json.loads((cell / "scores.json").read_text()) if (cell / "scores.json").exists() else None
            dst = self.arc(c) / m["id"]
            shutil.rmtree(dst, ignore_errors=True)
            (dst / "harness").mkdir(parents=True)
            shutil.copy(d / "harness" / "harness.py", dst / "harness" / "harness.py")
            (dst / "proposal.md").write_text((d / "proposal.md").read_text() if (d / "proposal.md").exists() else "")
            if sc is not None:
                (dst / "scores.json").write_text(json.dumps(sc, indent=1))
            if (cell / "episodes").exists():
                shutil.copytree(cell / "episodes", dst / "episodes")
            info = {"id": m["id"], "parent": m.get("parent"), "valid": m.get("valid", True), "reason": m.get("reason")}
            (dst / "meta.json").write_text(json.dumps(info, indent=1))
            (dst / "full.json").write_text(json.dumps({**info, "scores": sc}, indent=1))
        sc = json.loads((self.arc(c) / chosen / "full.json").read_text())["scores"]
        start = json.loads((self.arc(c) / "c000" / "full.json").read_text())["scores"] if (self.arc(c) / "c000").exists() else None
        g = json.loads(json.dumps(g))
        g["order"].append(c)
        g["leaves"][c] = {"code": (self.arc(c) / chosen / "harness" / "harness.py").read_text(), "doc": ""}
        meta["scores"][c], meta["per_episode"][c] = sc["mean_score"], sc["per_episode"]
        meta["start"][c] = (start or {}).get("mean_score")
        meta["last_export"][c] = f"import:{Path(run).name}/{chosen}"
        meta["used"][c] = {}
        meta.setdefault("leaf_cand", {})[c] = chosen
        self.save_state(g, meta)
        self.record(f"{tp}_start", {"task": c, "imported": f"{Path(run).name}/{chosen}", "start_score": meta["start"][c],
                                    "outcome": f"imported {Path(run).name}/{chosen}; search {sc['mean_score']}"})
        return self.document(c, g, meta), meta

    def document(self, c, g, meta):
        tag = f"doc_{c.replace(':', '_')}_{meta['leaf_cand'].get(c, 'leaf')}"
        ws = self.dir / "workspaces" / tag
        if self.fresh(tag):
            ws = self.ws(tag)
            (ws / "harness.py").write_text(g["leaves"][c]["code"])
            (ws / "TASK.md").write_text(TASK_DOCUMENT)
        self.propose(ws, tag)
        doc = (ws / "harness.md").read_text().strip() if (ws / "harness.md").exists() else ""
        g = json.loads(json.dumps(g))
        g["leaves"][c]["doc"] = doc or "(no description)"
        self.save_state(g, meta)
        self.record(tag, {"task": c, "kind": "document", "outcome": f"described ({len(doc)} chars)"})
        return g

    def describe(self, c, probe_tag):
        tag = f"{probe_tag}_description"
        ws = self.dir / "workspaces" / tag
        if self.fresh(tag):
            ws = self.ws(tag)
            self.add_evidence(ws, c, probe_tag)
            (ws / "TASK.md").write_text(TASK_DESCRIBE.format(tid=self.tid(c)) + (DESCRIBE_PROFILE if PROFILE else ""))
        self.propose(ws, tag)
        try:
            desc = (ws / "description.txt").read_text().strip()[:600] or "(no description)"
        except FileNotFoundError:
            desc = "(no description)"
        try:
            caps = [str(x)[:200] for x in json.loads((ws / "capabilities.json").read_text()) if str(x).strip()][:8]
        except Exception:
            caps = []
        return desc, caps

    def profile(self, probe_tag):
        """The game profile written with the description (PROFILE runs), {} if none: shown to compose and review,
        not used for retrieval (user, 2026-11-27)."""
        try:
            p = json.loads((self.dir / "workspaces" / f"{probe_tag}_description" / "profile.json").read_text())
            return {str(k): str(v)[:300] for k, v in p.items()} if isinstance(p, dict) else {}
        except Exception:
            return {}

    def lay_out(self, d, g, meta, tasks, workflows, modules):
        """Write what a game is built from into d: nodes/ (code, description, members, similarity) and
        similar_games/ (full harnesses with their descriptions and similarity). Returns the code shown."""
        shown = []
        (d / "nodes").mkdir()
        listing = {}
        for n in list(workflows) + list(modules):
            nd = g["nodes"][n]
            (d / "nodes" / f"{n}.py").write_text(nd["code"])
            info = {"kind": nd.get("kind"), "desc": nd.get("desc"), "parents": nd.get("builds_on") or nd.get("parents", []),
                    **{k: nd[k] for k in ("role", "entry", "params", "slots") if k in nd},
                    "members": {self.tid(x): v for x, v in nd.get("members", {}).items()}}
            if n in workflows:
                info["similarity"] = workflows[n]
            else:
                info.update(modules[n])
            (d / "nodes" / f"{n}.json").write_text(json.dumps(info, indent=1))
            listing[n] = {k: info[k] for k in info if k not in ("members",)}
            shown.append(nd["code"])
        (d / "nodes" / "index.json").write_text(json.dumps(listing, indent=1))
        (d / "similar_games").mkdir()
        for x, sv in tasks.items():
            (d / "similar_games" / f"{self.tid(x)}.py").write_text(g["leaves"][x]["code"])
            (d / "similar_games" / f"{self.tid(x)}.md").write_text(g["leaves"][x].get("doc", ""))
            shown.append(g["leaves"][x]["code"])
        (d / "similar_games" / "about.json").write_text(json.dumps(
            {self.tid(x): {"problem": g["descriptions"].get(x, ""), "search_score": meta["scores"].get(x),
                           "similarity": sv, **({"profile": g["profiles"][x]} if g.get("profiles", {}).get(x) else {}),
                           **({"config": g["configs"][x]} if g.get("configs", {}).get(x) else {})}
             for x, sv in tasks.items()}, indent=1))
        return shown

    def share_with_edit(self, ws, c, g, meta):
        """An edit sees what composing saw (user, 2026-11-27: what the model sees before deployment it sees at
        test time, and the hierarchy is visible to edits too): the same retrieval, without the game itself, in
        shared/ (read-only). The leak check's baseline grows by that code: it was written before this game."""
        others = dict(g, order=[x for x in g["order"] if x != c])
        tasks, workflows, modules = self.retrieve(others, c)
        if not (tasks or workflows or modules):
            return False
        (ws / "shared").mkdir()
        shown = self.lay_out(ws / "shared", g, meta, tasks, workflows, modules)
        base = self.dir / "private" / f"leak_baseline_{c.replace(':', '_')}.py"
        have = base.read_text() if base.exists() else ""
        add = [x for x in shown if x not in have]
        if add:
            base.write_text(have + "\n" + "\n".join(add))
        return True

    def suggest_params(self, g, c, modules):
        """Soft inheritance of parameters (user, 2026-11-28): for each retrieved module, the values its member games
        use, averaged with their similarity to the new game as weights (numbers: weighted mean; others: weighted
        vote). A suggestion only: the composer may ignore it."""
        v, out = g["vecs"][c], {}
        for n in modules:
            nd = g["nodes"].get(n, {})
            spec = nd.get("params") or {}
            for p in spec:
                vals = [(max(self.sim(v, g["vecs"][x]), 0.0), m["params"][p]) for x, m in nd.get("members", {}).items()
                        if x in g["vecs"] and p in (m.get("params") or {})]
                if not vals or sum(w for w, _ in vals) <= 0:
                    continue
                tot = sum(w for w, _ in vals)
                if all(isinstance(q, (int, float)) and not isinstance(q, bool) for _, q in vals):
                    val = sum(w * q for w, q in vals) / tot
                    val = round(val) if all(isinstance(q, int) for _, q in vals) else round(val, 3)
                else:
                    votes = {}
                    for w, q in vals:
                        votes[json.dumps(q)] = votes.get(json.dumps(q), 0) + w
                    val = json.loads(max(votes, key=votes.get))
                out.setdefault(n, {})[p] = {"suggested": val, "from": {self.tid(x): m["params"][p]
                                            for x, m in nd["members"].items() if p in (m.get("params") or {})}}
        return out

    def compose(self, tag, c, g, meta, tasks, workflows, modules, adapt_only=False):
        ws = self.dir / "workspaces" / tag
        slug = c.replace(":", "_")
        if self.fresh(tag):
            ws = self.ws(tag)
            shown = [BASE_AGENT.read_text()]
            (ws / "base_agent.py").write_text(BASE_AGENT.read_text())
            shown += self.lay_out(ws, g, meta, tasks, workflows, modules)
            (ws / "this_game.txt").write_text(g["descriptions"][c] + "\n\nRequired capabilities:\n" + "".join(
                f"- {x['capability']}\n" for x in g.get("capabilities", {}).get(c, []))
                + (("\nProfile:\n" + json.dumps(g["profiles"][c], indent=1) + "\n") if g.get("profiles", {}).get(c) else ""))
            self.add_evidence(ws, c, tag.replace("_compose", "_probe"))
            (ws / "harness.py").write_text(BASE_AGENT.read_text())
            task = TASK_COMPOSE
            if adapt_only:                  # single inheritance: one earlier game's harness, adapted to run here
                (ws / "TASK.md").write_text(task.format(tid=self.tid(c)))
            else:
                if COMPOSE_FREE:
                    assert COMPOSE_AS_REFERENCE[0] in task
                    task = task.replace(*COMPOSE_AS_REFERENCE)
                sp = self.suggest_params(g, c, modules) if REPR_D and modules else {}
                if sp:
                    (ws / "nodes" / "suggested_params.json").write_text(json.dumps(sp, indent=1))
                (ws / "TASK.md").write_text(task.format(tid=self.tid(c)) + (COMPOSE_D_NOTE if REPR_D else "")
                                            + (SUGGEST_NOTE if sp else "")
                                            + (COMPOSE_STRUCTURE_NOTE if STRUCT_HINT else ""))
            # the leak check counts only what was written after the model saw this game
            (self.dir / "private" / f"leak_baseline_{slug}.py").write_text("\n".join(shown))
        rc = self.propose(ws, tag)
        code = (ws / "harness.py").read_text() if (ws / "harness.py").exists() else ""
        doc0 = (ws / "harness.md").read_text().strip() if (ws / "harness.md").exists() else ""
        code, doc0, fixes = self.until_it_runs(tag, c, ws / "evidence" / self.tid(c) / "episodes", code, doc0)
        if code is None:
            print(f"{tag}: composed harness never ran; base agent used", flush=True)
            code, doc0 = BASE_AGENT.read_text(), BASE_AGENT_DOC
        try:
            used = [n for n in json.loads((ws / "compose.json").read_text()).get("used_nodes", [])
                    if n in workflows or n in modules]
        except Exception:
            used = []
        used = {n: workflows.get(n) for n in used}
        doc = doc0
        if code == BASE_AGENT.read_text() and not doc:
            doc = BASE_AGENT_DOC
        print(f"{tag}: rc {rc}, used nodes {sorted(used)}, description {len(doc)} chars", flush=True)
        return code, used, doc

    # ----- this task's candidates, laid out as Meta-Harness's archive (the inner loop is Meta-Harness's) -----
    def arc(self, c):
        return self.dir / "archive" / c.replace(":", "_")

    def arc_add(self, c, cid, parent, code, proposal, export_tag=None, valid=True, reason=None):
        d = self.arc(c) / cid
        shutil.rmtree(d, ignore_errors=True)
        (d / "harness").mkdir(parents=True)
        (d / "harness" / "harness.py").write_text(code)
        (d / "proposal.md").write_text(proposal or "")
        scores = None
        if export_tag:
            src = self.dir / "exports" / export_tag
            if (src / "scores.json").exists():
                shutil.copy(src / "scores.json", d / "scores.json")
                scores = json.loads((src / "scores.json").read_text())
            if (src / "episodes").exists():
                shutil.copytree(src / "episodes", d / "episodes")
        (d / "meta.json").write_text(json.dumps({"id": cid, "parent": parent, "valid": valid, "reason": reason}, indent=1))
        (d / "full.json").write_text(json.dumps({"id": cid, "parent": parent, "valid": valid, "reason": reason,
                                                 "scores": scores}, indent=1))

    def arc_fill(self, ws, c, parent):
        """archive/summary.md and archive/<id>/ in the workspace, exactly as Meta-Harness shows its archive."""
        arc = ws / "archive"
        arc.mkdir()
        rows = ["| id | parent | valid | mean score | success | reason |", "|---|---|---|---|---|---|"]
        for d in sorted(p for p in self.arc(c).iterdir() if (p / "full.json").exists()):
            m = json.loads((d / "full.json").read_text())
            sc = m.get("scores") or {}
            rows.append(f"| {m['id']} | {m.get('parent') or '-'} | {m['valid']} | {sc.get('mean_score', '-')} | "
                        f"{sc.get('success_rate', '-')} | {m.get('reason') or ''} |")
            dst = arc / m["id"]
            shutil.copytree(d, dst, ignore=shutil.ignore_patterns("full.json"))
        (arc / "summary.md").write_text("\n".join(rows) + "\n")
        m = json.loads((self.arc(c) / parent / "full.json").read_text())
        sc = [e["score"] for e in (m.get("scores") or {}).get("per_episode", {}).values() if e.get("score") is not None]
        return statistics.stdev(sc) / math.sqrt(len(sc)) if len(sc) > 1 else 0.0

    def edit(self, tag, c, g, meta):
        parent = meta["leaf_cand"][c]
        cid = "c%03d" % int(tag.rsplit("_e", 1)[1])
        ws = self.dir / "workspaces" / tag
        if self.fresh(tag):
            ws = self.ws(tag)
            (ws / "harness").mkdir()
            (ws / "harness" / "harness.py").write_text(g["leaves"][c]["code"])
            (ws / "harness.md").write_text(g["leaves"][c].get("doc") or "")
            noise = self.arc_fill(ws, c, parent)
            shared = self.share_with_edit(ws, c, g, meta)
            (ws / "TASK.md").write_text(
                (HERE / "ws_template" / "TASK_template.md").read_text().format(
                    model=MODEL, parent=parent, n=self.a.n, noise=noise, chat_cap=CHAT_CAP)
                + (EDIT_NOTE_TIE if TIE_COST else EDIT_NOTE).format(tid=self.tid(c))
                + (SHARED_NOTE if shared else "") + (EDIT_STRUCTURE_NOTE if STRUCT_HINT else ""))
        t0 = time.time()
        rc = self.propose(ws, tag)
        rec = {"task": c, "kind": "edit", "candidate": cid, "parent": parent, "proposer_rc": rc,
               "proposer_seconds": round(time.time() - t0)}
        hp = ws / "harness" / "harness.py"
        new_code = hp.read_text() if hp.exists() else ""
        new_doc = (ws / "harness.md").read_text().strip() if (ws / "harness.md").exists() else ""
        proposal = (ws / "proposal.md").read_text() if (ws / "proposal.md").exists() else ""
        if new_code == g["leaves"][c]["code"]:
            if new_doc and new_doc != g["leaves"][c].get("doc"):   # a description alone changes no behaviour
                g = json.loads(json.dumps(g))
                g["leaves"][c]["doc"] = new_doc
                self.save_state(g, meta)
            rec["outcome"] = "rejected: no change"
            self.record(tag, rec)
            return g, meta
        fixed, fixed_doc, fixes = self.until_it_runs(tag, c, self.arc(c) / parent / "episodes", new_code, new_doc)
        rec["repairs"] = fixes
        if fixed is None:
            self.arc_add(c, cid, parent, new_code, proposal, valid=False, reason=f"crashes after {fixes} repairs")
            rec["outcome"] = f"rejected: still crashing after {fixes} repairs"
            self.record(tag, rec)
            return g, meta
        new_code, new_doc = fixed, fixed_doc
        new, err = self.evaluate(g, [c], tag, srcs={c: new_code})
        if new is None:
            self.arc_add(c, cid, parent, new_code, proposal, valid=False, reason=err[:200])
            rec["outcome"] = "rejected: " + err
            self.record(tag, rec)
            return g, meta
        self.arc_add(c, cid, parent, new_code, proposal, export_tag=tag)
        w, l = paired(new[c]["per_episode"], meta["per_episode"][c])
        rec["tests"] = {"wins": w, "losses": l}
        nt, ot = token_costs(new[c]["per_episode"], meta["per_episode"][c])
        rec["tokens"] = {"new": nt, "parent": ot}
        tie_cheaper = (TIE_COST and new[c]["mean_score"] == meta["scores"][c]
                       and nt is not None and ot is not None and nt < ot)
        if new[c]["mean_score"] > meta["scores"][c] or tie_cheaper:
            g = json.loads(json.dumps(g))
            g["leaves"][c]["code"] = new_code
            g["leaves"][c]["doc"] = new_doc or g["leaves"][c].get("doc", "")
            meta["scores"][c], meta["per_episode"][c], meta["last_export"][c] = \
                new[c]["mean_score"], new[c]["per_episode"], tag
            meta["leaf_cand"][c] = cid
            self.save_state(g, meta)
            rec["outcome"] = (f"kept ({c} {new[c]['mean_score']}, tie, tokens/episode {nt:.0f} < {ot:.0f})"
                              if tie_cheaper else f"kept ({c} {new[c]['mean_score']})")
        else:
            rec["outcome"] = f"rejected ({new[c]['mean_score']} <= {meta['scores'][c]})"
        self.record(tag, rec)
        return g, meta

    def history_md(self, x, meta):
        """A game's history (user, 2026-11-27): every version with its parent, its search score, whether its
        change is in the final harness, and its author's notes on what changed and why."""
        d = self.arc(x)
        if not d.exists():
            return "(no recorded versions)\n"
        final = meta.get("leaf_cand", {}).get(x)
        par = {}
        rows, notes = ["| version | parent | search score | its change is in the final harness |", "|---|---|---|---|"], []
        for v in sorted(p for p in d.iterdir() if (p / "full.json").exists()):
            m = json.loads((v / "full.json").read_text())
            par[m["id"]] = m.get("parent")
        line, cur = set(), final
        while cur:
            line.add(cur)
            cur = par.get(cur)
        for v in sorted(p for p in d.iterdir() if (p / "full.json").exists()):
            m = json.loads((v / "full.json").read_text())
            sc = (m.get("scores") or {}).get("mean_score", "-")
            note = m.get("reason") or ""
            rows.append(f"| {m['id']} | {m.get('parent') or '-'} | {sc} | {'yes' if m['id'] in line else 'no'}"
                        + (f" ({note})" if note else "") + " |")
            prop = (v / "proposal.md").read_text().strip() if (v / "proposal.md").exists() else ""
            if prop:
                notes.append(f"### {m['id']}\n\n{prop[:2500]}")
        return "# Versions\n\n" + "\n".join(rows) + "\n\n# What each version changed and why\n\n" + "\n\n".join(notes) + "\n"

    def node_history_md(self, n, g, meta):
        """A node's history (user, 2026-11-27): where it came from (each member's part, evidence and differences;
        the games' own history files say which version introduced that code), every review that created or
        changed it with that review's notes, and how it has been used."""
        nd = g["nodes"].get(n)
        L = [f"# Node `{n}`" + ("" if nd else " (removed)"), ""]
        if nd:
            L += [f"Kind: {nd.get('kind')}. Description: {nd.get('desc', '')}", "", "## Distilled from", ""]
            for x, v in nd.get("members", {}).items():
                ev = v.get("evidence", {})
                L.append(f"- {self.tid(x)}: {v.get('implements', '')} | functions {ev.get('functions', [])} | "
                         f"differences: {v.get('differences', '')}")
            L += ["", "(Each game's `history_summary/taskN.md` says which of its versions introduced that code, and what "
                  "it did to the score.)", ""]
        L += ["## Changes", ""]
        for e in g.get("node_history", {}).get(n, []):
            L.append(f"- {e['review']} (after {e['after_game']}): {', '.join(e['change'])}; members {e['members']}")
            note = g.get("review_notes", {}).get(e["review"])
            if note:
                L.append("  That review's notes: " + note.replace("\n", " ")[:1500])
        u = meta.get("node_usage", {}).get(n)
        L += ["", "## Use by new games", "", json.dumps(u) if u else "(not offered to a new game yet)"]
        return "\n".join(L) + "\n"

    def review(self, tag, c, g, meta):
        ws = self.dir / "workspaces" / tag
        tids = {self.tid(x): x for x in g["order"]}
        if self.fresh(tag):
            ws = self.ws(tag)
            (ws / "games").mkdir()
            for x in g["order"]:
                (ws / "games" / f"{self.tid(x)}.py").write_text(g["leaves"][x]["code"])
                (ws / "games" / f"{self.tid(x)}.md").write_text(g["leaves"][x].get("doc", ""))
            hs = ws / "history_summary"
            (hs / "nodes").mkdir(parents=True)
            for x in g["order"]:
                (hs / f"{self.tid(x)}.md").write_text(self.history_md(x, meta))
            for n in sorted(set(g["nodes"]) | set(g.get("node_history", {}))):
                (hs / "nodes" / f"{n}.md").write_text(self.node_history_md(n, g, meta))
            (ws / "games" / "about.json").write_text(json.dumps(
                {self.tid(x): {"problem": g["descriptions"].get(x, ""), "search_score": meta["scores"].get(x),
                               **({"profile": g["profiles"][x]} if g.get("profiles", {}).get(x) else {})}
                 for x in g["order"]}, indent=1))
            (ws / "nodes").mkdir()
            for n, nd in g["nodes"].items():
                (ws / "nodes" / f"{n}.py").write_text(nd["code"])
                (ws / "nodes" / f"{n}.json").write_text(json.dumps(
                    {"kind": nd.get("kind"), "desc": nd.get("desc"), "parents": nd.get("builds_on") or nd.get("parents", []),
                     **{k: nd[k] for k in ("role", "entry", "params", "slots") if k in nd},
                     "members": {self.tid(x): v for x, v in nd.get("members", {}).items()},
                     "usage": meta.get("node_usage", {}).get(n, {"retrieved": 0, "chosen": 0, "start_vs_base_agent": []})},
                    indent=1))
            (ws / "graph.md").write_text(self.graph_md(g, meta))
            (ws / "base_agent.py").write_text(BASE_AGENT.read_text())
            (ws / "TASK.md").write_text(TASK_REVIEW.format(tid=self.tid(c), hard=HARD_NOTE if ABLATION == "hard" else "")
                                        + (REVIEW_D_NOTE if REPR_D else ""))
            if REPR_D:
                (ws / "leaves").mkdir()
                for x, cf in g.get("configs", {}).items():
                    (ws / "leaves" / f"{self.tid(x)}.json").write_text(json.dumps(cf, indent=1))
        rc = self.propose(ws, tag)
        rec = {"task": c, "kind": "review", "proposer_rc": rc}
        nodes, dropped = {}, {}
        fns = {x: defined_functions(g["leaves"][x]["code"]) for x in g["order"]}
        for f in sorted((ws / "nodes").glob("*.json")):
            n = f.stem
            why = []
            try:
                spec = json.loads(f.read_text())
                code = (ws / "nodes" / f"{n}.py").read_text()
            except Exception as e:
                dropped[n] = [f"unreadable: {e!r}"[:120]]
                continue
            if not NODE_ID.match(n):
                why.append("bad id")
            why += node_problems(code)
            mem = {}
            for t, v in (spec.get("members") or {}).items():
                x = tids.get(t)
                v = v if isinstance(v, dict) else {}
                ev = v.get("evidence") if isinstance(v.get("evidence"), dict) else {}
                names = [q for q in (ev.get("functions") or []) if isinstance(q, str)]
                quotes = [q for q in (ev.get("excerpts") or []) if isinstance(q, str)]
                diff = str(v.get("differences", "")).strip()
                bad = evidence_problems(names, quotes, fns[x], g["leaves"][x]["code"]) if x else ["unknown game"]
                if bad:
                    why.append(f"member {t}: " + "; ".join(bad))
                elif len(diff) < 10:
                    why.append(f"member {t}: no `differences` (how its version differs from this node)")
                else:
                    mem[x] = {"implements": str(v.get("implements", ""))[:300],
                              "evidence": {"functions": names, "excerpts": quotes}, "differences": diff[:600]}
                    if REPR_D and isinstance(v.get("params"), dict):
                        mem[x]["params"] = {str(k): q for k, q in v["params"].items()}
            dfields = d_problems(spec, code) if REPR_D else ([], {})
            why += dfields[0]
            if len(mem) < 2:
                why.append(f"distilled from {len(mem)} task(s); shared code needs at least 2")
            if len(str(spec.get("desc", "")).strip()) < 20:
                why.append("no description (a node is code plus a description)")
            if why:
                dropped[n] = why
                continue
            par = spec.get("parents") or {}
            par = {q: "" for q in par} if isinstance(par, list) else par
            nodes[n] = {"kind": spec.get("kind") if spec.get("kind") in ("module", "workflow") else "module",
                        "desc": str(spec.get("desc", ""))[:400], "code": code, "members": mem,
                        "builds_on": {p: str(d).strip()[:600] for p, d in par.items() if isinstance(p, str)},
                        **dfields[1]}
        for n, nd in nodes.items():           # links only to kept nodes, each with what this node adds to its parent
            links = {p: d for p, d in nd.pop("builds_on").items() if p in nodes and p != n}
            for p, d in list(links.items()):
                if len(d) < 10:
                    dropped[f"{n}<-{p}"] = ["link without a derivation note (what this node adds to that parent)"]
                    del links[p]
            links = dict(list(links.items())[:1]) if ABLATION == "hard" else links
            nd["parents"], nd["builds_on"] = list(links), links
            nd["centre"] = [sum(g["vecs"][x][i] for x in nd["members"]) / len(nd["members"])
                            for i in range(len(g["vecs"][c]))]
        if REPR_D:
            for n, nd in nodes.items():
                for sl, sv in nd.get("slots", {}).items():
                    dm = sv.get("default")
                    if dm and (dm not in nodes or nodes[dm].get("role") != sv.get("role")):
                        dropped[f"{n}.{sl}"] = [f"default {dm!r} is not a kept module of role {sv.get('role')}"]
                        sv["default"] = None
            configs = {}
            for f in sorted((ws / "leaves").glob("*.json")) if (ws / "leaves").exists() else []:
                x = tids.get(f.stem)
                try:
                    cf = json.loads(f.read_text())
                except Exception as e:
                    dropped[f"leaf {f.stem}"] = [f"unreadable: {e!r}"[:120]]
                    continue
                bad = config_problems(cf, nodes) if x else ["unknown game"]
                if bad:
                    dropped[f"leaf {f.stem}"] = bad
                else:
                    configs[x] = cf
        descs = [n for n in nodes if nodes[n]["desc"]]
        for n, dv in zip(descs, embed([nodes[n]["desc"] for n in descs]) if descs else []):
            nodes[n]["desc_vec"] = dv
        new_g = json.loads(json.dumps(g))
        new_g["nodes"] = nodes
        if REPR_D:
            new_g["configs"] = configs
        rec.update(nodes={n: {"kind": nd["kind"], "members": [self.tid(x) for x in nd["members"]],
                              "lines": len(nd["code"].splitlines()), "parents": nd["parents"]} for n, nd in nodes.items()},
                   dropped=dropped, before=sorted(g["nodes"]))
        hist = new_g.setdefault("node_history", {})
        notes = (ws / "review.md").read_text().strip()[:3000] if (ws / "review.md").exists() else ""
        for n in sorted(set(nodes) | set(g["nodes"])):
            a, b = g["nodes"].get(n), nodes.get(n)
            ch = (["created"] if a is None else ["removed"] if b is None else
                  [k for k, what in (("code", "code changed"), ("desc", "description changed"),
                                     ("members", "members changed"), ("parents", "parents changed"))
                   if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True)])
            if ch:
                hist.setdefault(n, []).append({"review": tag, "after_game": self.tid(c), "change": ch,
                                               "members": [self.tid(x) for x in (b or a).get("members", {})]})
        if notes and any(e["review"] == tag for v in hist.values() for e in v):
            new_g.setdefault("review_notes", {})[tag] = notes
        strip = lambda d: {n: {k: v for k, v in nd.items() if k != "desc_vec"} for n, nd in d.items()}
        if strip(nodes) == strip(g["nodes"]):
            rec["outcome"] = "no change to the graph" + (f" (dropped {sorted(dropped)})" if dropped else "")
            self.record(tag, rec)
            return g, meta
        self.save_state(new_g, meta)
        rec["outcome"] = f"graph now {sorted(nodes)}" + (f"; dropped {sorted(dropped)}" if dropped else "")
        self.record(tag, rec)
        return new_g, meta

    def final(self):
        full = []
        for v in self.versions():
            g, meta = self.load(v)
            if set(g["order"]) >= set(self.order) and all(meta["scores"].get(c) is not None for c in self.order):
                full.append((statistics.mean(meta["scores"][c] for c in self.order), v.name))
        top = [n for _, n in sorted(full, reverse=True)[:3]]
        sel = {}
        for name in top:
            g, _ = self.load(self.dir / "state" / name)
            folder = self.dir / "tmp" / f"sel_{name}"
            folder.mkdir(parents=True, exist_ok=True)
            hm = {}
            for c in self.order:
                p = folder / f"{c.replace(':', '_')}.py"
                p.write_text(g["leaves"][c]["code"])
                hm[c] = p
            key = f"{self.a.run}/{name}/selection"
            self.box.submit(key, hm, {"cell": ",".join(self.order), "split": "selection", "n": self.a.n_selection,
                                      "export": False, "smoke": False})
            while (r := self.box.result(key)) is None:
                time.sleep(30)
            sel[name] = r["summary"]["mean_score"] if "summary" in r else -1
        best = max(sel, key=sel.get) if sel else None
        (self.dir / "chosen.json").write_text(json.dumps({"best": best, "selection": sel, "candidates": full}, indent=1))
        print("selection:", sel, "best", best, flush=True)


TASK_DESCRIBE = """# Describe a new game

`evidence/{tid}/` has a few traces of a plain agent playing a game (its instructions are the first message).
Write `description.txt`: one or two sentences on the underlying problem and what it takes to solve it, not its
surface (for example "find a hidden item by proposing candidates and narrowing them down from structured
feedback", or "move through a small world to reach and act on a described object"). Games that are the same
problem should read alike even if their symbols, formats or stories differ.
Write `capabilities.json`: a list of 2 to 6 short sentences, each one capability the game requires (for example
"turn each round's feedback into constraints on the hidden answer", "remember what was already asked").
General capabilities, not the game's surface. No game or benchmark names.
"""
DESCRIBE_PROFILE = """Write `profile.json`, the game's profile as the traces show it, each field one short phrase:
{"observation": "<what each observation holds: only the latest result / the whole history so far / a local view
                 around the agent / ...>",
 "action": "<the form of an action: one of a fixed list of commands / free text / a word or letter in a required
            format / a command with arguments / ...>",
 "episode_length": "<the turn limit if stated, and how many turns the traces actually took>",
 "feedback": "<what the game reports after an action: structured per-element feedback / only win or lose at the
              end / a message when an action is invalid / ...>",
 "base_agent_failures": "<how the plain agent fails in these traces: its replies are not accepted as actions /
                         it repeats actions that change nothing / it forgets earlier information / it reasons
                         wrongly / ...>"}
"""


TASK_COMPOSE = """# Build a new game's first harness from shared code

A frozen open model plays games through harnesses (`run_episode(env, chat)`; `base_agent.py` is the plain starting
agent). Game `{tid}` is new: `this_game.txt` gives its description and required capabilities, and
`evidence/{tid}/` has a few traces of the base agent on it. You are given what was learned on earlier games:
- `similar_games/`: the complete harnesses of the most similar earlier games (`taskN.py` with its description
  `taskN.md`; `about.json`: their problems, scores and similarity to this game, 1 = same problem);
- `nodes/`: shared code distilled from earlier games (`<id>.py` code; `<id>.json` its description, the games it
  came from and how each differs, the nodes it builds on; `index.json` an overview). `workflow` nodes (how a
  family of games puts modules together) carry their similarity to this game. `module` nodes (small reusable
  pieces) carry their match to each required capability of this game, the games that use them with those games'
  similarity to this game, how many games use them (more = more general), and which retrieved workflows use them.

Write `harness.py` for game `{tid}` by putting together what fits from these, adapting only what this game needs
to run: its formats, symbols, lengths, wording of its messages. Do not add new strategies or tune it to the
traces; improving it is the next step. If nothing fits, keep the base agent. Write `harness.md`: how the harness
works, its main functions and what each does, its parameters. Write `compose.json`:
{{"used_nodes": ["<node id>", ...], "used_games": ["taskN", ...], "notes": "<what you took from where>"}}.
No game or benchmark names or answers anywhere. `check_harness.py` checks the static rules; `replay_test.py`
replays this game's recorded episodes to it (`python replay_test.py harness.py`): it must pass.
"""

COMPOSE_AS_REFERENCE = (
    "Write `harness.py` for game `{tid}` by putting together what fits from these, adapting only what this game needs\n"
    "to run: its formats, symbols, lengths, wording of its messages. Do not add new strategies or tune it to the\n"
    "traces; improving it is the next step. If nothing fits, keep the base agent.",
    "Write `harness.py` for game `{tid}` as this game actually needs it, using what you know about games of this kind.\n"
    "The harnesses and nodes above are references, and you reuse them first: take what fits, as it is or adapted.\n"
    "Write new code only where they do not fit this game, or where it needs something they do not have; say in\n"
    "`compose.json` `notes` which parts are new and why. The evidence traces show what the game is like; do not\n"
    "hard-code anything about specific instances. Improving the harness on its scores is the next step.")

COMPOSE_D_NOTE = """
## How the shared code is laid out

`workflow` nodes are skeletons: a family's main loop with named `slots` (each with a role and a default module).
`module` nodes fill slots; each has a `role`, an `entry` function and `params` with defaults; members list the
values their games use. `similar_games/about.json` gives each similar game's `config`: which skeleton, which module
in each slot, which parameter values, and what stays specific to that game. Where a skeleton fits this game, build
from it: choose a module for each slot and set its parameters. Also write `config.json` in the same form:
{"workflow": "<id or null>", "slots": {"<slot>": "<module id or own>"}, "params": {"<module id>": {...}},
 "own": "<what is specific to this game>"}.
"""

SUGGEST_NOTE = """
`nodes/suggested_params.json` suggests a starting value for each parameter of the retrieved modules: the values
their member games use, weighted by how similar each of those games is to this one. Use it where the parameter
means the same here; ignore it where it does not.
"""

COMPOSE_STRUCTURE_NOTE = """
## A checklist, not a required structure

A harness usually has some of these parts. Go through them for this game so that nothing it needs from the shared
code is missed; for each, take it from the shared code above, adapt it as described above, or leave it out, and say
which in `compose.json` `notes`. They are prompts for your judgement, not an interface: the rules above still apply,
and the code need not be organised this way.
- instruction: what the model is told about the game and the reply format;
- memory: what is kept from earlier turns, and in what form (raw messages, facts, a map, a list of candidates);
- context: what the model sees each turn and how long that gets over a long episode;
- tools: code that computes something for the model (parsing, search, counting);
- controller: code that decides actions or steps itself, and when it hands the decision to the model;
- verification: checks on the model's reply or action before it is sent (format, repeats, contradictions);
- workspace: anything set up once at the start of an episode.
"""

EDIT_STRUCTURE_NOTE = """
## A checklist for finding what is missing

When you look for the failure pattern, it can help to go through the parts a harness usually has and ask which one
the failures point to (one missing or weak, or one that grows too large over a long episode). This is a reference,
not a required structure, and the rules above still apply (one mechanism per change).
""" + "\n".join(l for l in COMPOSE_STRUCTURE_NOTE.splitlines() if l.startswith("- ")) + "\n"

EDIT_NOTE = """
## In this setting

- This harness belongs to game `{tid}` alone: other games have their own harnesses, so your change affects this
  game only. Candidates in `archive/` are this game's.
- Also keep `harness.md` (in the workspace root) up to date with the code: how the harness works, its main
  functions and what each does, its parameters. Write it if it is empty. You may edit it besides
  `harness/harness.py` and `proposal.md`.
- Keep functions small and named for what they do, with symbols, lengths and formats as named parameters: later,
  what several games share is distilled from their harnesses. No game or benchmark names anywhere.
- `replay_test.py` replays this game's recorded episodes (from the archive) to a harness:
  `python replay_test.py harness/harness.py` must pass, as well as `check_harness.py` and `mock_env.py`.
- Your change is kept only if this game's mean score on the same problems rises above the parent's.
"""
EDIT_NOTE_TIE = EDIT_NOTE.replace(
    "rises above the parent's.\n",
    "rises above the parent's, or equals it\n  with fewer model tokens per episode (input plus output, "
    "`model_in_tokens` + `model_out_tokens` in `scores.json`).\n")
TIE_COST = False
IMPORT = {}            # cell -> (Meta-Harness run folder, chosen candidate): the cold-start tasks
COMPOSE_FREE = False
STRUCT_HINT = False
STOP_AT_FULL = False
REPR_D = False          # skeletons with slots, typed modules with parameters, per-game configs (user, 2026-11-28)
PROFILE = False         # the describe step also writes a game profile (user, 2026-11-27)


def mean_tokens(per_episode, with_input=True):
    """Mean model tokens per episode (input + output, or output only), or None if any episode lacks them."""
    keys = ("model_in_tokens", "model_out_tokens") if with_input else ("model_out_tokens",)
    t = [[e.get(k) for k in keys] for e in per_episode.values()]
    if not t or any(x is None for row in t for x in row):
        return None
    return sum(sum(row) for row in t) / len(t)


def token_costs(new, old):
    """(new, parent) mean tokens per episode on the same footing: input + output when both records have
    input counts (exports from 2026-11-27 on), else output only."""
    both = mean_tokens(new) is not None and mean_tokens(old) is not None
    return mean_tokens(new, both), mean_tokens(old, both)

TASK_DOCUMENT = """# Describe a harness

`harness.py` is a game's harness (`run_episode(env, chat)` drives a frozen open model through a text game). Write
`harness.md`: how the harness works, its main functions and what each does, its parameters. Describe the code
as it is; do not change it. No game or benchmark names.
"""

SHARED_NOTE = """- `shared/` (read-only) holds what this game was built from: the most similar other games' harnesses with
  their descriptions and similarity (`shared/similar_games/`) and shared code distilled from earlier games, each
  node with its description, the games using it and its similarity (`shared/nodes/`). You may reuse any of it.
"""

START_NOTE = "Starting harness of this game ({how}); not written by an edit.\n"

TASK_REVIEW = """# Review: distil what the games share into shared code

Game `{tid}` has just finished its own edits. `games/` holds every game's own harness (`taskN.py`, described in
`taskN.md`; `about.json` says what each problem is), `nodes/` the shared code so far (`<id>.py` + `<id>.json`),
`graph.md` an overview. The games' harnesses stay as they are; the nodes are what new games are built from.

Distil the functionality the games share into real code, layered by generality:
- `module`: a small piece of functionality several games have (for example keeping a compact record of past
  turns, re-asking after a malformed reply, narrowing a candidate set from feedback), written once with what
  differs between the games as parameters.
- `workflow`: how a family of games puts modules together (a loop, an order of steps). Its parents are the
  modules it builds on; the more general a node, the higher it sits.
Put each piece on the node shared by exactly the games that have it. Functionality counts wherever it lives in a
harness: a function, a few lines of the main loop, prompt text, settings. You may add, change, merge, split or
delete nodes. A node is a pair `nodes/<id>.py` (code: functions, no `run_episode` needed) + `nodes/<id>.json`:
{{"kind": "module" | "workflow",
  "desc": "<required: what it does, when it fits, its parameters>",
  "parents": {{"<node id>": "<required: how this node builds on that parent>", ...}},
  "members": {{"taskN": {{"implements": "<which part of this node's functionality that game has>",
                         "evidence": {{"functions": ["<names of that game's functions>"],
                                      "excerpts": ["<verbatim code, prompt text or settings from its harness>"]}},
                         "differences": "<required: how that game's version differs from this node>"}}, ...}}}}
Each member needs evidence: function names defined in its harness, or excerpts copied verbatim from it (at least
40 characters each), or both. A node is kept only if its code is valid, it has a description, and at least two
games are members with valid evidence. Only what is genuinely shared: no node is better than a forced one.
Each node's `usage` says how often it was offered to a new game (`retrieved`), picked (`chosen`), and whether the
games started from it beat the base agent on the same problems. A node never retrieved or never chosen is NOT a
reason to delete it: later games may simply not have resembled it yet. Rewrite or remove a node only on evidence
against it (chosen at least twice and worse than the base agent each time) or when it duplicates another node.
`history_summary/` is there if you want it: `taskN.md` lists each version of a game's harness with its parent, its
score (before deployment) or the judge's choice (at test time), whether its change is in the final harness, and
its author's notes; `nodes/<id>.md` says where a node came from, which reviews changed it and how new games used
it. You may consult it, for example to see which change introduced a piece of code and what it did; you need not.
The base agent (`base_agent.py`) is fixed and is not a node. No game or benchmark names or answers anywhere.
Write `review.md`.
{hard}"""

BASE_AGENT_DOC = ("The base agent: each turn it sends a fixed system prompt and the current observation to the "
                "model, with no memory of earlier turns, and sends the model's whole reply as the action. One function, "
                "run_episode; no parameters.")

TASK_FIX = """# Fix a crash

Game `{tid}`'s harness `harness.py` (described in `harness.md`) crashes or breaks the static rules: `crash.txt`
has the output. Fix the bug so that it runs; change nothing else (no new strategy). `replay_test.py` replays
this game's recorded episodes (`evidence/`) to the harness: run `python replay_test.py harness.py` until it
passes; `check_harness.py` checks the static rules. Keep `harness.md` accurate.
"""

ROLES = ("instruction", "memory", "context", "tools", "controller", "verification", "workspace")
# the component families of Learning Meta-Skills (Qian et al., 2026, Table 1), used here as role labels only

REVIEW_D_NOTE = """
## How shared code is laid out (skeletons, slots, modules, parameters)

Lay the shared code out as a few skeletons with slots that modules fill:
- A `workflow` node is a **skeleton**: the main loop of a family of games (who decides each step, in what order
  the steps run), with named **slots** where modules plug in. Its code takes the slot implementations as arguments
  (functions or objects), so a game picks the modules. Add to its json
  `"slots": {"<slot name>": {"role": "<role>", "default": "<module id or null>", "what": "<what the slot does>"}}`.
- A `module` node fills a slot. Add to its json `"role": "<role>"`, `"entry": "<its main function>"` and
  `"params": {"<name>": {"type": "int | float | bool | str | choice", "default": <value>,
  "range": [lo, hi] or "options": [...], "what": "<what it controls>"}}`, and to each member
  `"params": {"<name>": <the value that game uses>}`.
- Roles (what a part does in a harness; labels, not a required structure): instruction, memory, context, tools,
  controller, verification, workspace.
- Modules are shared across skeletons where games of different families have the same part.
- Slots say what a skeleton is assembled from; `parents` now say what a node specializes: a more general node of the
  same kind (a skeleton for a narrower family built on a broader skeleton, a module built on a more general
  module), with what it adds. The hierarchy has no fixed depth: make a level only where the games show one, and a
  node may specialize several parents.
- Write `leaves/taskN.json` for every game: `{"workflow": "<skeleton id or null>", "slots": {"<slot>": "<module id
  or own>"}, "params": {"<module id>": {"<name>": <value>}}, "own": "<what stays specific to this game>"}`: how that
  game's harness is put together from the shared parts, as far as it is.
"""

HARD_NOTE = "\nEach node has at most one parent (a strict tree): the first one listed is kept.\n"


def main():
    global ABLATION, EDITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--remote", default="mailbox:~/evo_b_mailbox")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-selection", type=int, default=20)
    ap.add_argument("--max-tasks", type=int, default=0, help="pilot: only the first k tasks of the order")
    ap.add_argument("--edits", type=int, default=5)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--ablation", default="", choices=["", "hard", "flat", "retrieve"])
    ap.add_argument("--tie-cost", action="store_true",
                    help="an edit that ties the parent's mean score is kept if it uses fewer model tokens per "
                         "episode (user, 2026-11-27); off for the runs started before")
    ap.add_argument("--repr-d", action="store_true",
                    help="shared code as skeletons (workflows with slots) and typed modules with parameters, plus a "
                         "config per game saying how its harness is put together (user, 2026-11-28)")
    ap.add_argument("--profile", action="store_true",
                    help="the describe step also writes a game profile (observation, action, episode length, feedback, "
                         "base agent failures), shown to compose and review, not used for retrieval")
    ap.add_argument("--compose-free", action="store_true",
                    help="compose writes the harness the game needs, reusing the retrieved code first and writing new "
                         "code where it does not fit (user, 2026-11-27)")
    ap.add_argument("--import-from", default="",
                    help="cold start: JSON {cell: [Meta-Harness run folder, chosen candidate]}; these tasks join with "
                         "that harness (no compose, no edits) and are reviewed as usual")
    ap.add_argument("--stop-at-full", action="store_true",
                    help="a task's edits stop only when its search score reaches 1.0; no early stop after 3 edits "
                         "not kept (user, 2026-11-27)")
    ap.add_argument("--struct-hint", action="store_true",
                    help="compose only: add a checklist of harness parts (instruction, memory, context, tools, "
                         "controller, verification, workspace) as prompts, not a required structure (user, 2026-11-27)")
    a = ap.parse_args()
    global TIE_COST, STRUCT_HINT, STOP_AT_FULL, COMPOSE_FREE, IMPORT, PROFILE, REPR_D
    TIE_COST, STRUCT_HINT, STOP_AT_FULL, COMPOSE_FREE = a.tie_cost, a.struct_hint, a.stop_at_full, a.compose_free
    PROFILE, REPR_D = a.profile, a.repr_d
    IMPORT = {k: tuple(v) for k, v in json.loads(Path(a.import_from).read_text()).items()} if a.import_from else {}
    ABLATION, EDITS = a.ablation, a.edits
    st2.ABLATION = a.ablation
    run = Run(a)
    run.final() if a.final else run.loop()


if __name__ == "__main__":
    main()
