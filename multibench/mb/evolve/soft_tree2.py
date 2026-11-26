"""Our method, incremental version (2026-11-26, user): tasks join one at a time; the hierarchy is extracted
bottom-up from the harnesses themselves, and nodes inherit softly from several parents.

Structure (RUNS/<run>/state/vNNN/tree.json + shared.py):
  shared.py   one code base for every task; behaviour set by CONFIG (new modules behind switches)
  nodes       root, internal nodes and one leaf per task. A node has `parents` {id: weight} (weights sum to 1;
              the root has none), its own `config` overrides, a one-line `desc`, a `centre` in the essence space.
  resolve(n)  the parents' resolved configs mixed by weight (numbers averaged, switches by weighted vote,
              other values from the heaviest parent), then n's own overrides on top
  a task's harness = shared.py with CONFIG = resolve(its leaf)

Tasks are leaves and always sit at the bottom: a new task never inherits from another task's leaf, only from
the internal nodes (the level above the leaves) or the root. Its parent weights are a softmax over distances
from its essence vector to those nodes' centres (sigma 0.3, distances divided by the median pairwise
distance of the tasks seen so far), over the 3 nearest.

Tasks join in a fixed random order. For each new task (budget in units of "one task x 20 search problems"):
  1. probe: 3 search problems with the current root config (counts as 3 episodes)
  2. essence: the code-editing model writes what the problem is; it is embedded; parents are set
  3. start: the task's 20 search problems with its inherited config (1 unit) -- the transfer measurement
  4. edit (EDITS units): the model edits this task's leaf (or adds a module to shared.py behind a switch);
     kept if the paired sign test shows a gain and nothing else changes (other tasks are not affected:
     their resolved configs do not include this leaf)
  5. extract (from the second task on): the model compares this leaf with the nearest other leaves and moves
     what they have in common into an internal node (new, or an existing one), with a description. The code
     re-scores every task whose resolved config changes (1 unit each) and keeps the extraction only if none of
     them gets worse; then the involved leaves inherit softly from that node
Ablations: "flat" never extracts (every leaf inherits from the root only; its extract budget goes to edits);
"hard" extracts but gives each leaf a single parent (the nearest).
After the last task: versions that hold every task are ranked by mean search score; the top 3 go to the
selection split, as for every method.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

from mb.evolve import remote
from mb.evolve.check import problems
from mb.evolve.soft_tree import (HERE, SANDBOX, assemble, dist, embed, paired, sign_test_p)

RUNS = Path.home() / "evo_b_runs"
SIGMA = 0.3
K_PARENTS = 3
P_NOISE = 0.1
DROP_TOL = 0.1       # a mean fall this large also counts as worse (user, 2026-11-26)
EDITS = 5              # leaf edit rounds per task, as MH's 5 rounds per task (user); early stop below
BUDGET_UNITS = float("inf")  # no cap for now (user, 2026-11-26): run every step, record the units used; compare later
ABLATION = ""          # "" | "flat" | "hard"
ORDER_SEED = "exp_b/incremental/0"


# ---------- configs ----------

def mix(cfgs: dict, weights: dict) -> dict:
    """Weighted mix of configurations: a key absent from a config counts as 'not set' (its default)."""
    tot = sum(weights.values()) or 1.0
    keys = set().union(*cfgs.values()) if cfgs else set()
    top = max(weights, key=weights.get) if weights else None
    out = {}
    for k in keys:
        vals = {p: c[k] for p, c in cfgs.items() if k in c}
        sample = next(iter(vals.values()))
        if isinstance(sample, bool):
            on = sum(weights[p] for p, v in vals.items() if v is True) / tot
            off = sum(weights[p] for p, v in vals.items() if v is False) / tot
            if on > 0.5:
                out[k] = True
            elif off > 0.5:
                out[k] = False
        elif isinstance(sample, (int, float)) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                     for v in vals.values()):
            if sum(weights[p] for p in vals) / tot > 0.5:          # set by most of the weight: weighted mean
                w = sum(weights[p] for p in vals)
                avg = sum(weights[p] * v for p, v in vals.items()) / w
                out[k] = round(avg) if all(isinstance(v, int) for v in vals.values()) else avg
        else:
            heaviest = max(vals, key=lambda p: weights[p])
            if top in vals or sum(weights[p] for p in vals) / tot > 0.5:
                out[k] = vals[top] if top in vals else vals[heaviest]
    return out


def resolve(tree, nid, memo=None):
    memo = {} if memo is None else memo
    if nid in memo:
        return memo[nid]
    nd = tree["nodes"][nid]
    base = mix({p: resolve(tree, p, memo) for p in nd["parents"]}, nd["parents"]) if nd["parents"] else {}
    base.update(nd["config"])
    memo[nid] = base
    return base


def leaf(cell):
    return "leaf:" + cell


def internal(tree):
    return [n for n, nd in tree["nodes"].items() if not n.startswith("leaf:") and n != "root"]


def scale(tree):
    vs = list(tree["vecs"].values())
    ds = [dist(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]]
    return statistics.median(ds) if ds else 1.0


AFF_UP, AFF_MAX = 1.5, 3.0


def affinity(tree, task, node):
    return tree.get("affinity", {}).get(task, {}).get(node, 1.0)


def set_affinity(tree, task, node, value):
    tree.setdefault("affinity", {}).setdefault(task, {})[node] = value


def parent_weights(tree, vec, exclude=(), task=None):
    """Soft parents of a leaf among the internal nodes and the root: a softmax over distances (the prior:
    how alike the problems are) times the task's learned affinity to each node (the evidence: whether it
    gained or lost when the node changed; 0 = it left the node). The root's centre is the mean of all
    tasks seen, so a task unlike every internal node mostly inherits from the root."""
    cands = [n for n in internal(tree) if n not in exclude and (task is None or affinity(tree, task, n) > 0)]
    if ABLATION == "flat" or not cands:
        return {"root": 1.0}
    vs = list(tree["vecs"].values())
    tree["nodes"]["root"]["centre"] = [sum(v[i] for v in vs) / len(vs) for i in range(len(vs[0]))]
    cands = cands + ["root"]
    s = scale(tree) or 1.0
    near = sorted(cands, key=lambda n: dist(vec, tree["nodes"][n]["centre"]))[:K_PARENTS]
    logits = {n: -(dist(vec, tree["nodes"][n]["centre"]) / s / SIGMA) ** 2 / 2 for n in near}
    m = max(logits.values())
    raw = {n: math.exp(v - m) * (affinity(tree, task, n) if task else 1.0) for n, v in logits.items()}
    if ABLATION == "hard":
        best = max(raw, key=raw.get)
        return {best: 1.0}
    z = sum(raw.values()) or 1.0
    w = {n: v / z for n, v in raw.items()}
    return {n: round(x, 4) for n, x in w.items() if x >= 0.05} or {near[0]: 1.0}


def members(tree, nid):
    """Tasks whose leaf inherits from nid (directly or through other nodes), with the inherited weight."""
    out = {}
    for c in tree["order"]:
        if leaf(c) not in tree["nodes"]:
            continue
        w = reach(tree, leaf(c), nid)
        if w > 0:
            out[c] = w
    return out


def reach(tree, frm, to):
    if frm == to:
        return 1.0
    return sum(w * reach(tree, p, to) for p, w in tree["nodes"][frm]["parents"].items())


def recentre(tree):
    for n in internal(tree):
        mem = members(tree, n)
        if mem:
            tot = sum(mem.values())
            tree["nodes"][n]["centre"] = [sum(mem[c] * tree["vecs"][c][i] for c in mem) / tot
                                          for i in range(len(next(iter(tree["vecs"].values()))))]


# ---------- the run ----------

class Run:
    def __init__(self, a):
        self.a, self.dir = a, RUNS / a.run
        self.cells = a.cells.split(",")
        self.box = remote.make(a.remote)
        for d in ("state", "private", "rounds", "exports", "workspaces", "tmp"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        order = list(self.cells)
        random.Random(ORDER_SEED).shuffle(order)
        self.order = order[:a.max_tasks] if a.max_tasks else order
        self.units = 0

    # state
    def versions(self):
        return sorted(p for p in (self.dir / "state").iterdir() if (p / "tree.json").exists())

    def load(self, v):
        return (v / "shared.py").read_text(), json.loads((v / "tree.json").read_text()), json.loads((v / "meta.json").read_text())

    def save(self, name, shared, tree, meta):
        d = self.dir / "state" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "shared.py").write_text(shared)
        (d / "tree.json").write_text(json.dumps(tree, indent=1))
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    def tid(self, c):
        return f"task{self.cells.index(c) + 1}"

    # evaluation
    def harness_map(self, shared, tree, cells, tag):
        folder = self.dir / "tmp" / tag
        folder.mkdir(parents=True, exist_ok=True)
        out = {}
        for c in cells:
            cfg = resolve(tree, leaf(c)) if leaf(c) in tree["nodes"] else resolve(tree, "root")
            src = assemble(shared, cfg)
            bad = problems(src)
            if bad:
                return None, bad
            p = folder / f"{c.replace(':', '_')}.py"
            p.write_text(src)
            out[c] = p
        return out, []

    def evaluate(self, shared, tree, cells, tag, n=None, instances=None):
        """Score `cells` (each with its own harness); returns {cell: scores.json} or (None, error)."""
        hm, bad = self.harness_map(shared, tree, cells, tag)
        if bad:
            return None, "; ".join(bad)[:300]
        n = n or self.a.n
        key = f"{self.a.run}/{tag}/search"
        params = {"cell": ",".join(cells), "split": "search", "n": n, "export": True, "smoke": False}
        self.box.submit(key, hm, params)
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

    def propose(self, ws, tag):
        done = self.dir / "private" / f"{tag}_proposed.json"
        if done.exists() and ws.exists():
            return json.loads(done.read_text())["rc"]
        sys.path.insert(0, str(SANDBOX))
        import run_proposer
        rc = run_proposer.run(ws, self.dir / "private" / f"{tag}_proposer.jsonl")
        done.write_text(json.dumps({"rc": rc}, default=str))
        return rc

    def record(self, tag, rec):
        (self.dir / "rounds" / f"{tag}.json").write_text(json.dumps(rec, indent=1))
        print(f"{tag}: {rec.get('outcome')}", flush=True)

    def done(self, tag):
        f = self.dir / "rounds" / f"{tag}.json"
        return json.loads(f.read_text()) if f.exists() else None

    # workspace pieces
    def tree_md(self, tree, meta):
        lines = ["# The tree", "",
                 "Leaves are games (one each, task ids as in the folders); internal nodes hold what several games",
                 "share. Each node inherits softly from its parents (weights shown).", ""]
        for n, nd in tree["nodes"].items():
            who = self.tid(n[5:]) if n.startswith("leaf:") else n
            par = ", ".join(f"{p if not p.startswith('leaf:') else self.tid(p[5:])} ({w:.2f})" for p, w in nd["parents"].items())
            lines.append(f"- `{who}`: parents [{par}]; {nd.get('desc') or ''}; own config {json.dumps(nd['config'])}")
        lines += ["", "## Latest search score per game", ""]
        lines += [f"- {self.tid(c)}: {s}" for c, s in meta["scores"].items()]
        return "\n".join(lines) + "\n"

    def modules_md(self, shared, tree):
        """The shared code's switches and which games turn each on (their resolved configuration)."""
        import ast
        defaults = {}
        for node in ast.parse(shared).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_CONFIG" for t in node.targets):
                try:
                    defaults = ast.literal_eval(node.value)
                except Exception:
                    pass
        lines = ["# Modules of shared.py and the games that use them", "",
                 "Every game runs this same code; a CONFIG key switches a module on or sets it.", ""]
        for k, d in defaults.items():
            users = [self.tid(c) for c in tree["order"] if leaf(c) in tree["nodes"]
                     and resolve(tree, leaf(c)).get(k, d) != d]
            lines.append(f"- `{k}` (default {d!r}): {', '.join(users) if users else 'used by no game yet'}")
        return "\n".join(lines) + "\n"

    def base_ws(self, tag, shared, tree, meta):
        ws = self.dir / "workspaces" / tag
        shutil.rmtree(ws, ignore_errors=True)
        ws.mkdir(parents=True)
        for f in ("harness_lib.py", "mock_env.py"):
            shutil.copy(HERE / "ws_template" / f, ws / f)
        shutil.copy(HERE / "check.py", ws / "check_harness.py")
        (ws / "shared.py").write_text(shared)
        (ws / "tree.md").write_text(self.tree_md(tree, meta))
        (ws / "modules.md").write_text(self.modules_md(shared, tree))
        return ws

    def add_evidence(self, ws, c, export_tag, k=None):
        src = self.dir / "exports" / export_tag
        src = src / "cells" / c.replace(":", "_") if (src / "cells").exists() else src
        dst = ws / "evidence" / self.tid(c)
        dst.mkdir(parents=True, exist_ok=True)
        if (src / "scores.json").exists():
            shutil.copy(src / "scores.json", dst / "scores.json")
        eps = sorted((src / "episodes").iterdir()) if (src / "episodes").exists() else []
        for e in eps[:k] if k else eps:
            shutil.copytree(e, dst / "episodes" / e.name, dirs_exist_ok=True)

    # ----- the loop -----
    def loop(self):
        vs = self.versions()
        if vs:
            shared, tree, meta = self.load(vs[-1])
        else:
            shared = (HERE / "seeds" / "soft_tree_shared.py").read_text()
            tree = {"nodes": {"root": {"parents": {}, "config": {}, "desc": "every game", "centre": []}},
                    "vecs": {}, "order": [], "essence": {}}
            meta = {"scores": {}, "last_export": {}, "units": 0}
            self.save("v000", shared, tree, meta)
        self.units = meta.get("units", 0)
        for k, c in enumerate(self.order):
            if self.done(f"t{k + 1}_extract"):           # this task finished joining (resume)
                continue
            shared, tree, meta = self.add_task(k + 1, c, shared, tree, meta)
        print("all tasks added; units used:", self.units, flush=True)

    def add_task(self, k, c, shared, tree, meta):
        tagp = f"t{k}"
        # 1-3: probe, essence, parents, start score
        if c not in tree["order"]:
            probe, err = self.evaluate(shared, tree, [c], f"{tagp}_probe", n=3)
            ess = self.essence(c, f"{tagp}_probe")
            vec = embed([ess])[0]
            tree["vecs"][c], tree["essence"][c] = vec, ess
            if not tree["nodes"]["root"]["centre"]:
                tree["nodes"]["root"]["centre"] = vec
            tree["order"].append(c)
            tree["nodes"][leaf(c)] = {"parents": parent_weights(tree, vec, task=c), "config": {}, "desc": ess[:200],
                                      "centre": vec}
            start, err = self.evaluate(shared, tree, [c], f"{tagp}_start")
            meta["scores"][c] = start[c]["mean_score"] if start else None
            meta["per_episode"] = meta.get("per_episode", {})
            meta["per_episode"][c] = start[c]["per_episode"] if start else {}
            meta["last_export"][c] = f"{tagp}_start"
            meta["start"] = meta.get("start", {}) | {c: meta["scores"][c]}
            meta["units"] = self.units
            self.save(f"v{len(self.versions()):03d}", shared, tree, meta)
            self.record(f"{tagp}_start", {"task": c, "parents": tree["nodes"][leaf(c)]["parents"],
                                          "start_score": meta["scores"][c], "outcome": f"joined; start {meta['scores'][c]}"})
        # 4: edits of this task's leaf
        for e in range(1, EDITS + (1 if ABLATION == "flat" else 0) + 1):
            tag = f"{tagp}_e{e}"
            if self.done(tag):
                continue
            if self.units >= BUDGET_UNITS - 1:          # keep the budget equal to every other method's
                self.record(tag, {"outcome": "skipped: episode budget used up"})
                continue
            last3 = [self.done(f"{tagp}_e{k}") for k in range(e - 3, e) if k >= 1]
            if len(last3) == 3 and all(r and not str(r.get("outcome", "")).startswith("kept") for r in last3):
                self.record(tag, {"outcome": "skipped: early stop (3 edits in a row not kept, as every method)"})
                continue
            shared, tree, meta = self.edit_leaf(tag, c, shared, tree, meta)
        # 5: extraction
        if ABLATION != "flat" and len(tree["order"]) >= 2 and not self.done(f"{tagp}_extract"):
            shared, tree, meta = self.extract(f"{tagp}_extract", c, shared, tree, meta)
        elif not self.done(f"{tagp}_extract"):
            self.record(f"{tagp}_extract", {"outcome": "skipped (first task or flat)"})
        return shared, tree, meta

    def essence(self, c, probe_tag):
        ws = self.dir / "workspaces" / f"{probe_tag}_essence"
        if not ((self.dir / "private" / f"{probe_tag}_essence_proposed.json").exists() and ws.exists()):
            shutil.rmtree(ws, ignore_errors=True)
            ws.mkdir(parents=True)
            self.add_evidence(ws, c, probe_tag)
            (ws / "TASK.md").write_text(TASK_ESSENCE.format(tid=self.tid(c)))
        self.propose(ws, f"{probe_tag}_essence")
        try:
            return (ws / "essence.txt").read_text().strip()[:600] or c
        except FileNotFoundError:
            return c

    def edit_leaf(self, tag, c, shared, tree, meta):
        lf = leaf(c)
        ws = self.dir / "workspaces" / tag
        if (self.dir / "private" / f"{tag}_proposed.json").exists() and ws.exists():
            return self.score_edit(tag, c, shared, tree, meta, ws, resumed=True)
        ws = self.base_ws(tag, shared, tree, meta)
        inherited = mix({p: resolve(tree, p) for p in tree["nodes"][lf]["parents"]}, tree["nodes"][lf]["parents"])
        (ws / "inherited_config.json").write_text(json.dumps(inherited, indent=1))
        (ws / "leaf_config.json").write_text(json.dumps(tree["nodes"][lf]["config"], indent=1))
        self.add_evidence(ws, c, meta["last_export"][c])
        (ws / "TASK.md").write_text(TASK_LEAF.format(tid=self.tid(c), n=self.a.n))
        return self.score_edit(tag, c, shared, tree, meta, ws)

    def score_edit(self, tag, c, shared, tree, meta, ws, resumed=False):
        lf = leaf(c)
        t0 = time.time()
        rc = self.propose(ws, tag)                     # returns at once when the session already ran
        rec = {"task": c, "kind": "edit", "proposer_rc": rc, "proposer_seconds": round(time.time() - t0),
               "resumed": resumed}
        try:
            new_cfg = json.loads((ws / "leaf_config.json").read_text())
        except Exception as e:
            rec["outcome"] = f"rejected: bad leaf_config.json {e!r}"[:200]
            self.record(tag, rec)
            return shared, tree, meta
        new_shared = (ws / "shared.py").read_text()
        if new_cfg == tree["nodes"][lf]["config"] and new_shared == shared:
            rec["outcome"] = "rejected: no change"
            self.record(tag, rec)
            return shared, tree, meta
        cand = json.loads(json.dumps(tree))
        cand["nodes"][lf]["config"] = new_cfg
        others = [x for x in tree["order"] if x != c]
        cells = [c] + (others if new_shared != shared else [])   # shared code changed: every task re-scored
        new, err = self.evaluate(new_shared, cand, cells, tag)
        if new is None:
            rec["outcome"] = "rejected: " + err
            self.record(tag, rec)
            return shared, tree, meta
        tests = {x: dict(zip(("wins", "losses"), paired(new[x]["per_episode"], meta["per_episode"][x]))) for x in cells}
        drops = [x for x in cells if sign_test_p(tests[x]["losses"], tests[x]["wins"]) < P_NOISE]
        # kept when the edited task's score rises, as Meta-Harness keeps its best candidate (user, 2026-11-26);
        # protection of the others is strict: if shared.py changed, every task was re-scored and none may drop
        gain = new[c]["mean_score"] > meta["scores"][c] and not drops
        rec["tests"] = tests
        if gain and not drops:
            for x in cells:
                meta["scores"][x] = new[x]["mean_score"]
                meta["per_episode"][x] = new[x]["per_episode"]
                meta["last_export"][x] = tag
            meta["units"] = self.units
            self.save(f"v{len(self.versions()):03d}", new_shared, cand, meta)
            rec["outcome"] = f"kept ({c} {new[c]['mean_score']})"
            return_value = (new_shared, cand, meta)
        else:
            rec["outcome"] = f"rejected (gain {gain}, dropped {drops})"
            return_value = (shared, tree, meta)
        meta["units"] = self.units
        self.record(tag, rec)
        return return_value

    def extract(self, tag, c, shared, tree, meta):
        """The model moves what this leaf and its nearest other leaves share into an internal node."""
        # every task already in the tree, nearest first (at most 6 here): the model decides what is shared
        others = sorted([x for x in tree["order"] if x != c], key=lambda x: dist(tree["vecs"][x], tree["vecs"][c]))
        ws = self.dir / "workspaces" / tag
        if not ((self.dir / "private" / f"{tag}_proposed.json").exists() and ws.exists()):
            ws = self.base_ws(tag, shared, tree, meta)
            self.fill_extract_ws(ws, c, others, tree, meta)
        return self.finish_extract(tag, c, others, shared, tree, meta, ws)

    def fill_extract_ws(self, ws, c, others, tree, meta):
        leaves = {self.tid(x): {"own": tree["nodes"][leaf(x)]["config"], "resolved": resolve(tree, leaf(x)),
                                "essence": tree["essence"].get(x, ""),
                                "distance_to_new_game": round(dist(tree["vecs"][x], tree["vecs"][c]), 3)}
                  for x in [c] + others}
        (ws / "leaves.json").write_text(json.dumps(leaves, indent=1))
        (ws / "internal_nodes.json").write_text(json.dumps(
            {n: {"config": tree["nodes"][n]["config"], "desc": tree["nodes"][n].get("desc", "")} for n in internal(tree)}, indent=1))
        for x in [c] + others:
            self.add_evidence(ws, x, meta["last_export"][x], k=3)
        (ws / "TASK.md").write_text(TASK_EXTRACT.format(tid=self.tid(c)))

    def finish_extract(self, tag, c, others, shared, tree, meta, ws):
        rc = self.propose(ws, tag)
        rec = {"task": c, "kind": "extract", "proposer_rc": rc, "neighbours": others}
        try:
            prop = json.loads((ws / "extract.json").read_text())
        except Exception:
            prop = {}
        node = prop.get("node")
        cfg = prop.get("config") if isinstance(prop.get("config"), dict) else None
        ids = {self.tid(x): x for x in [c] + others}
        who = [ids[t] for t in prop.get("tasks", []) if t in ids]
        updates = {ids[t]: u for t, u in (prop.get("leaf_updates") or {}).items() if t in ids and isinstance(u, dict)}
        new_shared = (ws / "shared.py").read_text() if (ws / "shared.py").exists() else shared
        if new_shared != shared and problems(assemble(new_shared, {})):
            new_shared = shared                         # a refactor that breaks the static rules is dropped
        has_node = bool(node) and cfg is not None and len(who) >= 2
        if not has_node and not updates and new_shared == shared:
            rec["outcome"] = "nothing extracted"
            self.record(tag, rec)
            return shared, tree, meta
        cand = json.loads(json.dumps(tree))
        nid = None
        if has_node:
            same = [n for n in internal(cand) if all(cand["nodes"][n]["config"].get(k) == v for k, v in cfg.items())]
            nid = node if node in internal(cand) else (same[0] if same else f"n{len(internal(cand)) + 1}")
            if nid not in cand["nodes"]:
                cand["nodes"][nid] = {"parents": {"root": 1.0}, "config": {}, "desc": "", "centre": []}
            cand["nodes"][nid]["config"].update(cfg)
            cand["nodes"][nid]["desc"] = str(prop.get("description", ""))[:300]
            for x in who:                              # the extracted keys leave the leaves
                for key in cfg:
                    cand["nodes"][leaf(x)]["config"].pop(key, None)
            cand["nodes"][nid]["centre"] = [sum(cand["vecs"][x][i] for x in who) / len(who)
                                            for i in range(len(cand["vecs"][c]))]
        for x, u in updates.items():                   # a merged module: the leaves switch to its keys
            for k, v in u.items():
                if v is None:
                    cand["nodes"][leaf(x)]["config"].pop(k, None)
                else:
                    cand["nodes"][leaf(x)]["config"][k] = v
        for x in cand["order"]:                        # every leaf re-derives its soft parents
            cand["nodes"][leaf(x)]["parents"] = parent_weights(cand, cand["vecs"][x], task=x)
        recentre(cand)
        if has_node:                                   # a member that would lose an extracted setting keeps it
            for x in who:
                before, after = resolve(tree, leaf(x)), resolve(cand, leaf(x))
                for key in cfg:
                    if after.get(key) != before.get(key) and key in before:
                        cand["nodes"][leaf(x)]["config"][key] = before[key]
        if new_shared != shared:                       # the code changed: every task is re-scored
            changed = list(cand["order"])
        else:
            changed = [x for x in cand["order"] if resolve(cand, leaf(x)) != resolve(tree, leaf(x))]
        if not changed:
            rec["outcome"] = "extraction changes no task's configuration: kept without scoring"
            meta["units"] = self.units
            self.save(f"v{len(self.versions()):03d}", shared, cand, meta)
            self.record(tag, rec)
            return shared, cand, meta
        new, err = self.evaluate(new_shared, cand, changed, tag)
        if new is None:
            rec["outcome"] = "rejected: " + err
            self.record(tag, rec)
            return shared, tree, meta
        tests = {x: dict(zip(("wins", "losses"), paired(new[x]["per_episode"], meta["per_episode"][x]))) for x in changed}
        # a task is worse if it drops significantly or its mean falls by DROP_TOL or more
        worse = [x for x in changed if sign_test_p(tests[x]["losses"], tests[x]["wins"]) < P_NOISE
                 or new[x]["mean_score"] <= meta["scores"][x] - DROP_TOL]
        rec.update(tests=tests, node=nid, tasks=who, changed=changed, refactored=new_shared != shared,
                   leaf_updates=list(updates), worse=worse)
        if worse and new_shared != shared:             # shared code cannot be split per task: all or nothing
            rec["outcome"] = f"rejected: the shared-code change hurt {worse}"
            self.record(tag, rec)
            return shared, tree, meta
        keep = [x for x in changed if x not in worse]
        if not keep:
            rec["outcome"] = f"rejected: every re-scored task got worse {worse}"
            self.record(tag, rec)
            return shared, tree, meta
        mean_new = statistics.mean(new[x]["mean_score"] for x in keep)
        mean_old = statistics.mean(meta["scores"][x] for x in keep)
        rec.update(mean_old=mean_old, mean_new=mean_new)
        if mean_new < mean_old:
            rec["outcome"] = f"rejected: mean over the tasks that stay {mean_old:.3f} -> {mean_new:.3f}"
            self.record(tag, rec)
            return shared, tree, meta
        for x in worse:                                # it leaves the node: its leaf as before, affinity 0
            cand["nodes"][leaf(x)] = json.loads(json.dumps(tree["nodes"][leaf(x)]))
            if nid:
                set_affinity(cand, x, nid, 0.0)
                old_par = {p: w for p, w in cand["nodes"][leaf(x)]["parents"].items() if p != nid}
                tot = sum(old_par.values()) or 1.0
                cand["nodes"][leaf(x)]["parents"] = {p: w / tot for p, w in old_par.items()} or {"root": 1.0}
        if nid:
            for x in keep:                             # it gained or held: a stronger tie to the node
                if new[x]["mean_score"] > meta["scores"][x]:
                    set_affinity(cand, x, nid, min(AFF_MAX, affinity(cand, x, nid) * AFF_UP))
            recentre(cand)
        changed = keep
        shared = new_shared
        for x in changed:
            meta["scores"][x] = new[x]["mean_score"]
            meta["per_episode"][x] = new[x]["per_episode"]
            meta["last_export"][x] = tag
        meta["units"] = self.units
        self.save(f"v{len(self.versions()):03d}", shared, cand, meta)
        rec["outcome"] = (f"extracted into {nid} for {who}" if nid else "refactored") + \
                         (" with a shared-code refactor" if rec["refactored"] else "") + \
                         f"; kept for {changed}" + (f"; {worse} got worse and left the node" if worse else "")
        self.record(tag, rec)
        return shared, cand, meta

    def final(self):
        full = []
        for v in self.versions():
            _, tree, meta = self.load(v)
            if set(tree["order"]) >= set(self.order) and all(meta["scores"].get(c) is not None for c in self.order):
                full.append((statistics.mean(meta["scores"][c] for c in self.order), v.name))
        top = [n for _, n in sorted(full, reverse=True)[:3]]
        sel = {}
        for name in top:
            shared, tree, _ = self.load(self.dir / "state" / name)
            hm, _ = self.harness_map(shared, tree, self.order, f"sel_{name}")
            key = f"{self.a.run}/{name}/selection"
            self.box.submit(key, hm, {"cell": ",".join(self.order), "split": "selection", "n": self.a.n_selection,
                                      "export": False, "smoke": False})
            while (r := self.box.result(key)) is None:
                time.sleep(30)
            sel[name] = r["summary"]["mean_score"] if "summary" in r else -1
        best = max(sel, key=sel.get) if sel else None
        (self.dir / "chosen.json").write_text(json.dumps({"best": best, "selection": sel, "candidates": full}, indent=1))
        print("selection:", sel, "best", best, flush=True)


TASK_ESSENCE = """# Describe what this game is, as a problem

`evidence/{tid}/` has a few traces of a simple agent playing a game (its instructions are the first message).
Write `essence.txt`: one or two sentences on the underlying problem and what it takes to solve it, not the
surface (for example "find a hidden item by proposing candidates and narrowing them down from structured
feedback", or "move through a small world to reach and act on a described object"). Games that are the same
problem should read alike even if their symbols, formats or stories differ. No game or benchmark names.
"""

TASK_LEAF = """# Improve one game's harness (its leaf in a shared tree)

A frozen open model plays several games through one shared code base, `shared.py`; each game's behaviour is set
by CONFIG. Games are the leaves of a tree; internal nodes hold what several games share, and each leaf inherits
from its parents (`tree.md`). `inherited_config.json` (read-only) is what game `{tid}` inherits; your change goes
into its own overrides, `leaf_config.json`, which affect this game only.

You may also change `shared.py`, which is shared by every game: `modules.md` lists its modules and which
games use each. Reuse or extend an existing module before writing a new one, so that games with the same
need end up with the same mechanism (later, common parts are moved up the tree). A new module goes behind a
CONFIG key whose default keeps current behaviour; a change to an existing module must keep the behaviour of
the games already using it. Every game is re-scored when shared.py changes, and the change is undone if any
game gets worse. Keep the `CONFIG = {{}}` line.

Evidence: `evidence/{tid}/` has its scores ({n} problems) and traces. Tabulate how episodes end before reading
traces; read the last steps of failing traces. Code may decide actions; a module that cannot read an
observation should hand the decision back to the model. Write mechanisms for the problem, not its surface:
keep symbols, lengths and formats as CONFIG parameters. No game or benchmark names or answers anywhere.
`check_harness.py` and `mock_env.py` check that the harness runs. Write `proposal.md` (what you saw, what you
changed, why). The change is kept only if this game gains on the same problems, beyond noise.
"""

TASK_EXTRACT = """# Review every game's harness and move what they share up the tree

Game `{tid}` has just been added. You see the whole picture: `leaves.json` (every game's own overrides, its
resolved configuration, a description of its problem and its distance to the new game, nearest first),
`internal_nodes.json` (what is already shared), `shared.py` with `modules.md` (the modules every game runs and
which games use each), `tree.md`, and a few traces per game in `evidence/`.

Two kinds of sharing, either or both:
1. Settings: move what at least two games set alike, in a way that is about the problem, into an internal
   node (an existing one if it is the same idea, or a new one). Games far apart can share a mechanism.
2. Code: where two modules do nearly the same thing for different games, merge them into one module with
   CONFIG parameters (an alphabet, a length, a format), edit `shared.py`, and switch the games over to it.

Write `extract.json`:
{{"node": "<existing node id, 'new', or null>", "config": {{<keys and values the node will hold>}},
  "tasks": ["task1", "task4"], "description": "<one line: what these games share, as a problem>",
  "leaf_updates": {{"task4": {{"<key>": <new value, or null to remove it>}}}}}}
The keys put in the node are removed from those games' leaves. Every game whose configuration changes, and
every game if `shared.py` changes, is re-scored: the change is kept only if no game gets worse and their mean
does not fall. If nothing is genuinely shared, write {{"node": null}} and change nothing.
"""


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
    run = Run(a)
    run.final() if a.final else run.loop()


if __name__ == "__main__":
    main()
