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
  2. retrieve: similar tasks and workflow nodes, top-k by similarity (workflows weighted by affinity), all module
     nodes with their match to the required capabilities ("hard": one path; "flat": nothing)
  3. compose: the model builds the task's first harness and its description from what was retrieved, adapting
     only what is needed to run on this task; no new optimisation. Skipped when nothing is retrieved
  4. start: 20 search problems (the transfer measurement); on the 3 probe problems the start is compared with
     the base agent: better -> affinity to the used nodes x1.5 (max 3), worse -> x0.5
  5. up to 5 edits of the task's own harness, each kept if its mean rises (as Meta-Harness keeps its best
     candidate); early stop after 3 in a row not kept
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

RUNS = Path.home() / "evo_b_runs"
BASE_AGENT = HERE / "seeds" / "textarena_standard.py"
SIGMA, K_NODES, K_TASKS = 0.3, 3, 2
AFF_UP, AFF_DOWN, AFF_MAX = 1.5, 0.5, 3.0
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
        self.box.submit(key, hm, {"cell": ",".join(cells), "split": "search", "n": n, "export": True, "smoke": False})
        while (r := self.box.result(key)) is None:
            time.sleep(30)
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

    def until_it_runs(self, tag, c, ws, code, doc):
        """Repair sessions until the harness runs on this game's recorded traces (user, 2026-11-27).
        Returns (code, doc, fixes) or (None, None, fixes) if it still crashes after MAX_FIXES."""
        traces = sorted(str(p) for p in (ws / "evidence").glob("*/episodes/*/trace.md"))
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
                if (ws / "evidence").exists():
                    shutil.copytree(ws / "evidence", fws / "evidence")
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
        - workflow nodes, by the similarity of their nearest member to the task, times the task's affinity to the
          node (top K_NODES; "hard": 1);
        - module nodes: all of them (short and general: a single similarity would bury the most shared ones), each
          with its match to the task's required capabilities, the tasks using it and their similarity, and whether
          a retrieved workflow builds on it; "hard": only the modules on the retrieved workflow's path.
        "flat": nothing."""
        if ABLATION == "flat":
            return {}, {}, {}
        v = g["vecs"][c]
        tasks = sorted(((x, self.sim(v, g["vecs"][x])) for x in g["order"]), key=lambda t: -t[1])
        tasks = dict(tasks[:1 if ABLATION == "hard" else K_TASKS])
        aff = g.get("affinity", {}).get(c, {})
        workflows = {}
        for n, nd in g["nodes"].items():
            if nd.get("kind") == "workflow" and aff.get(n, 1.0) > 0:
                workflows[n] = max(self.sim(v, g["vecs"][x]) for x in nd["members"])
        workflows = dict(sorted(workflows.items(), key=lambda t: -(t[1] * aff.get(t[0], 1.0)))[:1 if ABLATION == "hard" else K_NODES])
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
            meta = {"scores": {}, "per_episode": {}, "last_export": {}, "start": {}, "used": {}, "units": 0}
            self.save_state(g, meta)
        self.units = meta.get("units", 0)
        for k, c in enumerate(self.order):
            if self.done(f"t{k + 1}_review"):
                continue
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
            g.setdefault("capabilities", {})[c] = [{"capability": t, "vec": v} for t, v in zip(caps, vecs[1:])]
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
            us = meta.setdefault("node_usage", {})
            for n in list(workflows) + list(modules):
                us.setdefault(n, {"retrieved": 0, "chosen": 0, "start_vs_base_agent": []})["retrieved"] += 1
            pw, pl = paired({i: e for i, e in start[c]["per_episode"].items() if i in probe[c]["per_episode"]},
                            probe[c]["per_episode"])
            for n in used:                     # evidence about the nodes: did starting from them beat the base agent?
                meta["node_usage"][n]["chosen"] += 1
                meta["node_usage"][n]["start_vs_base_agent"].append(
                    {"game": self.tid(c), "wins": pw, "losses": pl, "verdict": "better" if pw > pl else "worse" if pl > pw else "same"})
                a = g["affinity"].setdefault(c, {}).get(n, 1.0)
                if pw > pl:
                    g["affinity"][c][n] = min(AFF_MAX, a * AFF_UP)
                elif pl > pw:
                    g["affinity"][c][n] = a * AFF_DOWN
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
            last3 = [self.done(f"{tp}_e{j}") for j in range(e - 3, e) if j >= 1]
            if len(last3) == 3 and all(r and not str(r.get("outcome", "")).startswith("kept") for r in last3):
                self.record(tag, {"outcome": "skipped: early stop (3 edits in a row not kept, as every method)"})
                continue
            g, meta = self.edit(tag, c, g, meta)
        # 6 review
        if ABLATION != "flat" and len(g["order"]) >= 2:
            g, meta = self.review(f"{tp}_review", c, g, meta)
        else:
            self.record(f"{tp}_review", {"outcome": "skipped (first task or flat)"})
        return g, meta

    def describe(self, c, probe_tag):
        tag = f"{probe_tag}_description"
        ws = self.dir / "workspaces" / tag
        if self.fresh(tag):
            ws = self.ws(tag)
            self.add_evidence(ws, c, probe_tag)
            (ws / "TASK.md").write_text(TASK_DESCRIBE.format(tid=self.tid(c)))
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

    def compose(self, tag, c, g, meta, tasks, workflows, modules):
        ws = self.dir / "workspaces" / tag
        slug = c.replace(":", "_")
        if self.fresh(tag):
            ws = self.ws(tag)
            shown = [BASE_AGENT.read_text()]
            (ws / "base_agent.py").write_text(BASE_AGENT.read_text())
            (ws / "nodes").mkdir()
            listing = {}
            for n in list(workflows) + list(modules):
                nd = g["nodes"][n]
                (ws / "nodes" / f"{n}.py").write_text(nd["code"])
                info = {"kind": nd.get("kind"), "desc": nd.get("desc"), "parents": nd.get("builds_on") or nd.get("parents", []),
                        "members": {self.tid(x): v for x, v in nd.get("members", {}).items()}}
                if n in workflows:
                    info["similarity"] = workflows[n]
                else:
                    info.update(modules[n])
                (ws / "nodes" / f"{n}.json").write_text(json.dumps(info, indent=1))
                listing[n] = {k: info[k] for k in info if k not in ("members",)}
                shown.append(nd["code"])
            (ws / "nodes" / "index.json").write_text(json.dumps(listing, indent=1))
            (ws / "similar_games").mkdir()
            for x, sv in tasks.items():
                (ws / "similar_games" / f"{self.tid(x)}.py").write_text(g["leaves"][x]["code"])
                (ws / "similar_games" / f"{self.tid(x)}.md").write_text(g["leaves"][x].get("doc", ""))
                shown.append(g["leaves"][x]["code"])
            (ws / "similar_games" / "about.json").write_text(json.dumps(
                {self.tid(x): {"problem": g["descriptions"].get(x, ""), "search_score": meta["scores"].get(x),
                               "similarity": sv} for x, sv in tasks.items()}, indent=1))
            (ws / "this_game.txt").write_text(g["descriptions"][c] + "\n\nRequired capabilities:\n" + "".join(
                f"- {x['capability']}\n" for x in g.get("capabilities", {}).get(c, [])))
            self.add_evidence(ws, c, tag.replace("_compose", "_probe"))
            (ws / "harness.py").write_text(BASE_AGENT.read_text())
            (ws / "TASK.md").write_text(TASK_COMPOSE.format(tid=self.tid(c)))
            # the leak check counts only what was written after the model saw this game
            (self.dir / "private" / f"leak_baseline_{slug}.py").write_text("\n".join(shown))
        rc = self.propose(ws, tag)
        code = (ws / "harness.py").read_text() if (ws / "harness.py").exists() else ""
        doc0 = (ws / "harness.md").read_text().strip() if (ws / "harness.md").exists() else ""
        code, doc0, fixes = self.until_it_runs(tag, c, ws, code, doc0)
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

    def edit(self, tag, c, g, meta):
        ws = self.dir / "workspaces" / tag
        if self.fresh(tag):
            ws = self.ws(tag)
            (ws / "harness.py").write_text(g["leaves"][c]["code"])
            (ws / "harness.md").write_text(g["leaves"][c].get("doc") or "")
            self.add_evidence(ws, c, meta["last_export"][c])
            (ws / "TASK.md").write_text(TASK_EDIT.format(tid=self.tid(c), n=self.a.n))
        t0 = time.time()
        rc = self.propose(ws, tag)
        rec = {"task": c, "kind": "edit", "proposer_rc": rc, "proposer_seconds": round(time.time() - t0)}
        new_code = (ws / "harness.py").read_text() if (ws / "harness.py").exists() else ""
        new_doc = (ws / "harness.md").read_text().strip() if (ws / "harness.md").exists() else ""
        if new_code != g["leaves"][c]["code"]:
            fixed, fixed_doc, fixes = self.until_it_runs(tag, c, ws, new_code, new_doc)
            rec["repairs"] = fixes
            if fixed is None:
                rec["outcome"] = f"rejected: still crashing after {fixes} repairs"
                self.record(tag, rec)
                return g, meta
            new_code, new_doc = fixed, fixed_doc
        if new_code == g["leaves"][c]["code"]:
            if new_doc and new_doc != g["leaves"][c].get("doc"):   # a description alone changes no behaviour
                g = json.loads(json.dumps(g))
                g["leaves"][c]["doc"] = new_doc
                self.save_state(g, meta)
            rec["outcome"] = "rejected: no change"
            self.record(tag, rec)
            return g, meta
        new, err = self.evaluate(g, [c], tag, srcs={c: new_code})
        if new is None:
            rec["outcome"] = "rejected: " + err
            self.record(tag, rec)
            return g, meta
        w, l = paired(new[c]["per_episode"], meta["per_episode"][c])
        rec["tests"] = {"wins": w, "losses": l}
        if new[c]["mean_score"] > meta["scores"][c]:
            g = json.loads(json.dumps(g))
            g["leaves"][c]["code"] = new_code
            g["leaves"][c]["doc"] = new_doc or g["leaves"][c].get("doc", "")
            meta["scores"][c], meta["per_episode"][c], meta["last_export"][c] = \
                new[c]["mean_score"], new[c]["per_episode"], tag
            self.save_state(g, meta)
            rec["outcome"] = f"kept ({c} {new[c]['mean_score']})"
        else:
            rec["outcome"] = f"rejected ({new[c]['mean_score']} <= {meta['scores'][c]})"
        self.record(tag, rec)
        return g, meta

    def review(self, tag, c, g, meta):
        ws = self.dir / "workspaces" / tag
        tids = {self.tid(x): x for x in g["order"]}
        if self.fresh(tag):
            ws = self.ws(tag)
            (ws / "games").mkdir()
            for x in g["order"]:
                (ws / "games" / f"{self.tid(x)}.py").write_text(g["leaves"][x]["code"])
                (ws / "games" / f"{self.tid(x)}.md").write_text(g["leaves"][x].get("doc", ""))
            (ws / "games" / "about.json").write_text(json.dumps(
                {self.tid(x): {"problem": g["descriptions"].get(x, ""), "search_score": meta["scores"].get(x)}
                 for x in g["order"]}, indent=1))
            (ws / "nodes").mkdir()
            for n, nd in g["nodes"].items():
                (ws / "nodes" / f"{n}.py").write_text(nd["code"])
                (ws / "nodes" / f"{n}.json").write_text(json.dumps(
                    {"kind": nd.get("kind"), "desc": nd.get("desc"), "parents": nd.get("builds_on") or nd.get("parents", []),
                     "members": {self.tid(x): v for x, v in nd.get("members", {}).items()},
                     "usage": meta.get("node_usage", {}).get(n, {"retrieved": 0, "chosen": 0, "start_vs_base_agent": []})},
                    indent=1))
            (ws / "graph.md").write_text(self.graph_md(g, meta))
            (ws / "base_agent.py").write_text(BASE_AGENT.read_text())
            (ws / "TASK.md").write_text(TASK_REVIEW.format(tid=self.tid(c), hard=HARD_NOTE if ABLATION == "hard" else ""))
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
                        "builds_on": {p: str(d).strip()[:600] for p, d in par.items() if isinstance(p, str)}}
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
        descs = [n for n in nodes if nodes[n]["desc"]]
        for n, dv in zip(descs, embed([nodes[n]["desc"] for n in descs]) if descs else []):
            nodes[n]["desc_vec"] = dv
        new_g = json.loads(json.dumps(g))
        new_g["nodes"] = nodes
        rec.update(nodes={n: {"kind": nd["kind"], "members": [self.tid(x) for x in nd["members"]],
                              "lines": len(nd["code"].splitlines()), "parents": nd["parents"]} for n, nd in nodes.items()},
                   dropped=dropped, before=sorted(g["nodes"]))
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

TASK_EDIT = """# Improve one game's harness

A frozen open model plays game `{tid}` through `harness.py` (`run_episode(env, chat)`), described in `harness.md`.
This harness is this game's own: change it freely. Evidence: `evidence/{tid}/` has its scores ({n} problems) and
traces. Tabulate how episodes end before reading traces; read the last steps of failing traces. Code may decide
actions; a part that cannot read an observation should hand the decision back to the model. Write mechanisms for
the problem, not its surface: keep symbols, lengths and formats as named parameters, and keep functions small and
named for what they do (later, what several games share is distilled from their harnesses). No game or benchmark
names or answers anywhere. `check_harness.py` checks the static rules; `replay_test.py` replays this game's recorded
episodes to the harness (`python replay_test.py harness.py`): it must pass. Keep `harness.md` (how the
harness works, its main functions and what each does, its parameters) up to date with the code; write it if it is
empty. Write `proposal.md` (what you saw, what you changed, why). The change is kept only if this game's mean
score on the same problems rises.
"""

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
    ap.add_argument("--ablation", default="", choices=["", "hard", "flat"])
    a = ap.parse_args()
    ABLATION, EDITS = a.ablation, a.edits
    st2.ABLATION = a.ablation
    run = Run(a)
    run.final() if a.final else run.loop()


if __name__ == "__main__":
    main()
