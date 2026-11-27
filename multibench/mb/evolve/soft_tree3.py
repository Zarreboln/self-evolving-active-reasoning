"""Our method, version 3 (user, 2026-11-27): every task keeps its own best harness; shared code is distilled
from those harnesses into a layered graph of operations and combinations, used to start new tasks.

Parts
  skeleton     TextArena's standard agent (seeds/textarena_standard.py): observe, call the model, act. Fixed,
               the same start as every other method; not part of the graph.
  leaves       one complete harness per task, its own. Changed only by that task's own edits.
  nodes        shared code distilled from the leaves by a review, each = real code + a description:
                 operation    a small reusable piece (e.g. keep a compact record of past turns)
                 combination  a way of putting operations together for a family of problems
               A node lists the tasks it was distilled from and, per task, which of the task's own functions it
               generalises and how the task differs from it (required); a link to a parent node says what the
               node adds to that parent (required). Every harness and node carries a description. Nodes do not run inside the tasks' harnesses:
               they are templates for new tasks (and for test time). Several parents are allowed (a graph).

For each task, in a fixed random order (seed "exp_b/incremental/0"):
  1. probe: 3 search problems with the skeleton; the code-editing model writes what the problem is
     (essence), which is embedded
  2. retrieve: the nearest nodes (softmax over essence distances, sigma 0.3, times the task's affinity to
     each node; up to 3; "hard": 1) and the nearest tasks (up to 2; "hard": 1)
  3. compose: the model builds the task's first harness from those nodes' code and those tasks' harnesses,
     adapting only what is needed to run on this task (formats, symbols, lengths); no new optimisation.
     Skipped when there is nothing to retrieve, and in "flat" (the skeleton is the start)
  4. start: 20 search problems (the transfer measurement); on the 3 probe problems the start is compared
     with the skeleton: better -> affinity to the used nodes x1.5 (max 3), worse -> x0.5
  5. up to 5 edits of the task's own harness, each kept if its mean rises (as Meta-Harness keeps its best
     candidate); early stop after 3 in a row not kept
  6. review (from the 2nd task on, not "flat"): the model reads every task's harness and the graph, and may
     add, change, merge, split or delete nodes and their links. It changes no task's harness, so nothing is
     re-scored. A node is kept only if it is real shared code: it compiles, passes the static rules, defines
     functions, and is distilled from at least 2 tasks whose harnesses really define the functions it names.
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
SKELETON = HERE / "seeds" / "textarena_standard.py"
SIGMA, K_NODES, K_TASKS = 0.3, 3, 2
AFF_UP, AFF_DOWN, AFF_MAX = 1.5, 0.5, 3.0
EDITS = 5
ABLATION = ""
ORDER_SEED = st2.ORDER_SEED
NODE_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def defined_functions(src):
    try:
        return {n.name for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    except SyntaxError:
        return set()


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
            src = (srcs or {}).get(c) or g["leaves"].get(c, {}).get("code") or SKELETON.read_text()
            bad = problems(src)
            if bad:
                return None, "; ".join(bad)[:300]
            p = folder / f"{c.replace(':', '_')}.py"
            p.write_text(src)
            hm[c] = p
            lb = self.dir / "private" / f"leakbase_{c.replace(':', '_')}.py"
            if lb.exists():     # code written before the model saw this task: its words are not answers
                hm["leakbase_" + c] = lb
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
        return ws

    def fresh(self, tag):
        """True if the proposer session of `tag` has not run yet (the workspace must be built)."""
        return not ((self.dir / "private" / f"{tag}_proposed.json").exists() and (self.dir / "workspaces" / tag).exists())

    # ----- the graph -----
    @staticmethod
    def sim(a, b):
        """Cosine similarity of two normalised essence vectors."""
        return round(1 - dist(a, b) ** 2 / 2, 3)

    def retrieve(self, g, c):
        """What a new task is built from (user, 2026-11-27):
        - similar earlier tasks, by essence similarity (top K_TASKS; "hard": 1);
        - combination nodes, by the similarity of their nearest member to the task, times the task's affinity
          to the node (top K_NODES; "hard": 1);
        - operation nodes: all of them (short and general: a similarity would bury the most shared ones),
          each with the tasks using it and their similarity, and whether a retrieved combination builds on it;
          "hard": only the operations on the retrieved combination's path (one parent each).
        "flat": nothing."""
        if ABLATION == "flat":
            return {}, {}, {}
        v = g["vecs"][c]
        tasks = sorted(((x, self.sim(v, g["vecs"][x])) for x in g["order"]), key=lambda t: -t[1])
        tasks = dict(tasks[:1 if ABLATION == "hard" else K_TASKS])
        aff = g.get("affinity", {}).get(c, {})
        combos = {}
        for n, nd in g["nodes"].items():
            if nd.get("kind") == "combination" and aff.get(n, 1.0) > 0:
                combos[n] = max(self.sim(v, g["vecs"][x]) for x in nd["members"])
        combos = dict(sorted(combos.items(), key=lambda t: -(t[1] * aff.get(t[0], 1.0)))[:1 if ABLATION == "hard" else K_NODES])
        def up(n, seen):
            for p in g["nodes"].get(n, {}).get("parents", []):
                if p in g["nodes"] and p not in seen:
                    seen.add(p)
                    up(p, seen)
            return seen
        on_path = set().union(*(up(n, set()) for n in combos)) if combos else set()
        ops = {}
        for n, nd in g["nodes"].items():
            if nd.get("kind") == "combination" or (ABLATION == "hard" and n not in on_path):
                continue
            ops[n] = {"matches_needs": self.need_matches(g, c, nd),
                      "used_by": {self.tid(x): self.sim(v, g["vecs"][x]) for x in nd["members"]},
                      "games_using_it": len(nd["members"]),
                      "built_on_by_retrieved_combinations": sorted(m for m in combos if n in up(m, set()))}
        return tasks, combos, ops

    def need_matches(self, g, c, nd):
        """Each of the new task's stated needs with its similarity to this operation's description (HyperSkill-style
        retrieval by sub-needs, user 2026-11-27): direct evidence, unlike similarity through the games using it."""
        if not nd.get("desc_vec"):
            return {}
        return dict(sorted(((x["need"], self.sim(x["vec"], nd["desc_vec"])) for x in g.get("needs", {}).get(c, [])),
                           key=lambda t: -t[1]))

    def graph_md(self, g, meta):
        L = ["# Shared code graph", "", "Nodes are code distilled from the games' own harnesses. `operation`: a small",
             "reusable piece; `combination`: operations put together for a family of games.", ""]
        for n, nd in g["nodes"].items():
            mem = ", ".join(f"{self.tid(x)}" for x in nd.get("members", {}))
            L.append(f"- `{n}` ({nd.get('kind')}): {nd.get('desc', '')} | parents {nd.get('parents', [])} | "
                     f"distilled from {mem}")
        if not g["nodes"]:
            L.append("(no shared node yet)")
        L += ["", "## Games", ""]
        L += [f"- {self.tid(c)}: {g['essence'].get(c, '')} (search score {meta['scores'].get(c)})" for c in g["order"]]
        return "\n".join(L) + "\n"

    # ----- the loop -----
    def loop(self):
        vs = self.versions()
        if vs:
            g, meta = self.load(vs[-1])
        else:
            g = {"nodes": {}, "leaves": {}, "vecs": {}, "essence": {}, "order": [], "affinity": {}}
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
            # 1 probe with the skeleton, essence
            probe, err = self.evaluate(g, [c], f"{tp}_probe", n=3, srcs={c: SKELETON.read_text()})
            if probe is None:
                raise RuntimeError(f"{tp}_probe failed: {err}")
            ess, needs = self.essence(c, f"{tp}_probe")
            vecs = embed([ess] + needs)
            g["vecs"][c], g["essence"][c] = vecs[0], ess
            g.setdefault("needs", {})[c] = [{"need": t, "vec": v} for t, v in zip(needs, vecs[1:])]
            # 2-3 retrieve and compose
            tasks, combos, ops = self.retrieve(g, c)
            code, used, doc = SKELETON.read_text(), {}, SKELETON_DOC
            if tasks or combos or ops:
                code, used, doc = self.compose(f"{tp}_compose", c, g, meta, tasks, combos, ops)
            else:
                (self.dir / "private" / f"leakbase_{slug}.py").write_text(SKELETON.read_text())
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
            for n in list(combos) + list(ops):
                us.setdefault(n, {"retrieved": 0, "chosen": 0, "start_vs_skeleton": []})["retrieved"] += 1
            pw, pl = paired({i: e for i, e in start[c]["per_episode"].items() if i in probe[c]["per_episode"]},
                            probe[c]["per_episode"])
            for n in used:                     # evidence about the nodes: did starting from them beat the skeleton?
                meta["node_usage"][n]["chosen"] += 1
                meta["node_usage"][n]["start_vs_skeleton"].append(
                    {"game": self.tid(c), "wins": pw, "losses": pl, "verdict": "better" if pw > pl else "worse" if pl > pw else "same"})
                a = g["affinity"].setdefault(c, {}).get(n, 1.0)
                if pw > pl:
                    g["affinity"][c][n] = min(AFF_MAX, a * AFF_UP)
                elif pl > pw:
                    g["affinity"][c][n] = a * AFF_DOWN
            self.save_state(g, meta)
            self.record(f"{tp}_start", {"task": c, "similar_tasks": tasks, "combinations": combos,
                                        "operations": sorted(ops), "used": used,
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

    def essence(self, c, probe_tag):
        tag = f"{probe_tag}_essence"
        ws = self.dir / "workspaces" / tag
        if self.fresh(tag):
            ws = self.ws(tag)
            self.add_evidence(ws, c, probe_tag)
            (ws / "TASK.md").write_text(st2.TASK_ESSENCE.format(tid=self.tid(c)) + NEEDS_NOTE)
        self.propose(ws, tag)
        try:
            ess = (ws / "essence.txt").read_text().strip()[:600] or "(no description)"
        except FileNotFoundError:
            ess = "(no description)"
        try:
            needs = [str(x)[:200] for x in json.loads((ws / "needs.json").read_text()) if str(x).strip()][:8]
        except Exception:
            needs = []
        return ess, needs

    def compose(self, tag, c, g, meta, tasks, combos, ops):
        ws = self.dir / "workspaces" / tag
        slug = c.replace(":", "_")
        if self.fresh(tag):
            ws = self.ws(tag)
            shown = [SKELETON.read_text()]
            (ws / "skeleton.py").write_text(SKELETON.read_text())
            (ws / "nodes").mkdir()
            listing = {}
            for n in list(combos) + list(ops):
                nd = g["nodes"][n]
                (ws / "nodes" / f"{n}.py").write_text(nd["code"])
                info = {"kind": nd.get("kind"), "desc": nd.get("desc"), "parents": nd.get("derives") or nd.get("parents", []),
                        "members": {self.tid(x): v for x, v in nd.get("members", {}).items()}}
                if n in combos:
                    info["similarity"] = combos[n]
                else:
                    info.update(ops[n])
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
                {self.tid(x): {"problem": g["essence"].get(x, ""), "search_score": meta["scores"].get(x),
                               "similarity": sv} for x, sv in tasks.items()}, indent=1))
            (ws / "this_game.txt").write_text(g["essence"][c] + "\n\nWhat it needs:\n" + "".join(
                f"- {x['need']}\n" for x in g.get("needs", {}).get(c, [])))
            self.add_evidence(ws, c, tag.replace("_compose", "_probe"))
            (ws / "harness.py").write_text(SKELETON.read_text())
            (ws / "TASK.md").write_text(TASK_COMPOSE.format(tid=self.tid(c)))
            # the leak check counts only what was written after the model saw this game
            (self.dir / "private" / f"leakbase_{slug}.py").write_text("\n".join(shown))
        rc = self.propose(ws, tag)
        code = (ws / "harness.py").read_text() if (ws / "harness.py").exists() else ""
        if problems(code):
            print(f"{tag}: composed harness fails the static rules {problems(code)[:2]}; skeleton used", flush=True)
            code = SKELETON.read_text()
        try:
            used = [n for n in json.loads((ws / "compose.json").read_text()).get("used_nodes", [])
                    if n in combos or n in ops]
        except Exception:
            used = []
        used = {n: combos.get(n) for n in used}
        doc = (ws / "harness.md").read_text().strip() if (ws / "harness.md").exists() else ""
        if code == SKELETON.read_text() and not doc:
            doc = SKELETON_DOC
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
                {self.tid(x): {"problem": g["essence"].get(x, ""), "search_score": meta["scores"].get(x)}
                 for x in g["order"]}, indent=1))
            (ws / "nodes").mkdir()
            for n, nd in g["nodes"].items():
                (ws / "nodes" / f"{n}.py").write_text(nd["code"])
                (ws / "nodes" / f"{n}.json").write_text(json.dumps(
                    {"kind": nd.get("kind"), "desc": nd.get("desc"), "parents": nd.get("derives") or nd.get("parents", []),
                     "members": {self.tid(x): v for x, v in nd.get("members", {}).items()},
                     "usage": meta.get("node_usage", {}).get(n, {"retrieved": 0, "chosen": 0, "start_vs_skeleton": []})},
                    indent=1))
            (ws / "graph.md").write_text(self.graph_md(g, meta))
            (ws / "skeleton.py").write_text(SKELETON.read_text())
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
                names = [s for s in (v.get("functions") or []) if isinstance(s, str)] if isinstance(v, dict) else []
                diff = str(v.get("difference", "")).strip() if isinstance(v, dict) else ""
                if not (x and names and all(s in fns[x] for s in names)):
                    why.append(f"member {t}: functions {names} not all defined in its harness")
                elif len(diff) < 10:
                    why.append(f"member {t}: no difference note (how its version differs from this node)")
                else:
                    mem[x] = {"functions": names, "difference": diff[:600]}
            if len(mem) < 2:
                why.append(f"distilled from {len(mem)} task(s); shared code needs at least 2")
            if len(str(spec.get("desc", "")).strip()) < 20:
                why.append("no description (a node is code plus a description)")
            if why:
                dropped[n] = why
                continue
            par = spec.get("parents") or {}
            par = {p: "" for p in par} if isinstance(par, list) else par
            nodes[n] = {"kind": spec.get("kind") if spec.get("kind") in ("operation", "combination") else "operation",
                        "desc": str(spec.get("desc", ""))[:400], "code": code, "members": mem,
                        "derives": {p: str(d).strip()[:600] for p, d in par.items() if isinstance(p, str)}}
        for n, nd in nodes.items():           # links only to kept nodes, each with what this node adds to its parent
            links = {p: d for p, d in nd.pop("derives").items() if p in nodes and p != n}
            for p, d in list(links.items()):
                if len(d) < 10:
                    dropped[f"{n}<-{p}"] = ["link without a derivation note (what this node adds to that parent)"]
                    del links[p]
            links = dict(list(links.items())[:1]) if ABLATION == "hard" else links
            nd["parents"], nd["derives"] = list(links), links
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


TASK_COMPOSE = """# Build a new game's first harness from shared code

A frozen open model plays games through harnesses (`run_episode(env, chat)`; `skeleton.py` is the plain
starting agent). Game `{tid}` is new: `this_game.txt` says what its problem is and `evidence/{tid}/` has a
few traces of the plain agent on it. You are given what was learned on earlier games:
- `similar_games/`: the complete harnesses of the most similar earlier games (`about.json`: their problems,
  scores and similarity to this game, 1 = same problem);
- `nodes/`: shared code distilled from earlier games (`<id>.py` real code; `<id>.json` what it does, which games
  it came from and how each differs; `index.json` an overview). `combination` nodes (ways a family of games puts
  operations together) carry their similarity to this game. `operation` nodes (small reusable pieces) carry how
  well they match each of this game's needs (`this_game.txt`), the games that use them with those games' similarity
  to this game, how many games use them (more = more general), and which retrieved combinations build on them.

Write `harness.py` for game `{tid}` by putting together what fits from these, adapting only what this game
needs to run: its formats, symbols, lengths, wording of its messages. Do not add new strategies or tune it
to the traces; improving it is the next step. If nothing fits, keep the skeleton. Write `harness.md`: how the
harness works, its main functions and what each does, its parameters (each earlier game's `.md` next to its
code is such a description). Write `compose.json`:
{{"used_nodes": ["<node id>", ...], "used_games": ["taskN", ...], "notes": "<what you took from where>"}}.
No game or benchmark names or answers anywhere. `check_harness.py` and `mock_env.py` check that it runs.
"""

TASK_EDIT = """# Improve one game's harness

A frozen open model plays game `{tid}` through `harness.py` (`run_episode(env, chat)`). This harness is this
game's own: change it freely. Evidence: `evidence/{tid}/` has its scores ({n} problems) and traces. Tabulate how
episodes end before reading traces; read the last steps of failing traces. Code may decide actions; a part that
cannot read an observation should hand the decision back to the model. Write mechanisms for the problem, not
its surface: keep symbols, lengths and formats as named parameters, and keep functions small and named for what
they do (later, what several games share is distilled from their harnesses). No game or benchmark names or
answers anywhere. `check_harness.py` and `mock_env.py` check that it runs. Keep `harness.md` (how the harness
works, its main functions and what each does, its parameters) up to date with the code; write it if it is
empty. Write `proposal.md` (what you saw,
what you changed, why). The change is kept only if this game's mean score on the same problems rises.
"""

TASK_REVIEW = """# Review: distil what the games share into shared code

Game `{tid}` has just finished its own edits. `games/` holds every game's own harness (`taskN.py`, described in
`taskN.md`; `about.json`
says what each problem is), `nodes/` the shared code so far (`<id>.py` + `<id>.json`), `graph.md` an overview.
The games' harnesses stay as they are; the nodes are what a new game will be built from.

Distil real shared code, layered by generality:
- `operation`: a small piece several games do alike (for example keeping a compact record of past turns,
  re-asking on a malformed reply, narrowing a candidate set from feedback). Make what differs between
  the games parameters, so that one function serves all of them.
- `combination`: how a family of games puts operations together (a loop, an order of steps). Its `parents`
  are the operations it builds on; the more general a node, the higher it sits.
Put each thing on the node shared by exactly the games that have it. You may add, change, merge, split or delete
nodes. A node is a pair `nodes/<id>.py` (code: functions, no `run_episode` needed) + `nodes/<id>.json`:
{{"kind": "operation" | "combination", "desc": "<required: what it does, when it fits, its parameters>",
  "parents": {{"<node id>": "<required: what this node adds to that parent, how it builds on it>", ...}},
  "members": {{"taskN": {{"functions": ["<names of that game's own functions this generalises>"],
                         "difference": "<required: how that game's version differs from this node>"}}, ...}}}}
Each node's `usage` says how often it was offered to a new game (`retrieved`), picked (`chosen`), and whether the
games started from it beat the plain skeleton on the same problems. A node never retrieved or never chosen is NOT
a reason to delete it: later games may simply not have resembled it yet. Rewrite or remove a node only on evidence
against it (chosen at least twice and worse than the skeleton each time) or when it duplicates another node.
A node is kept only if its code is valid and it is distilled from at least two games whose harnesses really
define the functions you list. Only what is genuinely shared: no node is better than a forced one. The skeleton
(`skeleton.py`) is fixed and is not a node. No game or benchmark names or answers anywhere. Write `review.md`.
{hard}"""

SKELETON_DOC = ("The plain starting agent: each turn it sends a fixed system prompt and the current observation to the "
                "model, with no memory of earlier turns, and sends the model's whole reply as the action. One function, "
                "run_episode; no parameters.")

NEEDS_NOTE = """
Also write `needs.json`: a list of 2 to 6 short sentences, each one capability the game needs (for example
"turn each round's feedback into constraints on the hidden answer", "remember what was already asked"). Generic
capabilities, not the game's surface; no game or benchmark names.
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
