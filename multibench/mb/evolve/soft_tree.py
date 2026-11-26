"""Our method before deployment: a soft general-to-specific tree of harness configurations over one shared
code base (notes/method_decomposition.md, settings of notes/exp_b_design.md).

State (RUNS/<run>/state/vNNN/, one folder per accepted version):
  shared.py    the shared code base (seeds/soft_tree_shared.py at v000); CONFIG is filled in per task
  tree.json    nodes {id: {parent, children, config (overrides), centre}}, the distance scale, task vectors

A task's harness = shared.py with CONFIG set to its mixed configuration:
  weights   per level, HSML Eq. 6: w(t, child) ∝ exp(-||(e_t - c_child) / sigma||^2 / 2) over the children
            of one node, distances divided by the median pairwise distance of the development tasks,
            sigma = 1 (fixed, as HSML fixes it); a leaf's weight is the product along its path
  mixing    each leaf's path configuration (root -> leaf overrides) is mixed by leaf weight: numbers are
            averaged, switches are on when their weighted vote is above one half, other values come from
            the leaf with the largest weight

One round (one code-editing session and 7 x 20 = 140 search episodes, as every method gets):
  1. choose a node, top-down: round r edits level r mod depth, the node there with the largest weighted
     failure mass sum_t w(t, n) (1 - score_t)
  2. the code-editing model (Claude Code, Opus 5.5, subscription, in the Docker sandbox) reads the
     traces of the tasks below that node (more episodes from tasks that weigh it more) and may change
     the shared code (new modules only behind CONFIG switches whose defaults keep current behaviour)
     and that node's configuration
  3. every development task is scored with its own assembled harness on the same 20 search problems
  4. keep / split / reject, by the paired noise test (sign test over problems, one-sided p < 0.1):
     keep if the node's tasks gain by weight and no task drops; split if some of the node's tasks gain
     and others drop: the new configuration goes to a new child for the tasks that gained, the node
     keeps the old one; a change to the shared code is kept only if no task drops
  5. centres move to the weighted mean of their tasks' vectors (weight x positive gain), soft k-means
  6. children whose total weight is under 0.05 are removed
Not yet implemented here (ablation-safe, see notes): promoting a child's override to its parent.

    python -m mb.evolve.soft_tree --run ours --remote mailbox:~/evo_b_mailbox --cells A,B,... [--final]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mb.evolve import remote
from mb.evolve.check import problems

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parents[2] / "sandbox"
RUNS = Path.home() / "evo_b_runs"
EMBED_PY = Path.home() / "venvs" / "softtree" / "bin" / "python"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SIGMA = 1.0
ABLATION = ""        # "" (ours) | "hard": one path per task (TreeMAML) | "flat": no levels, soft weights only
MAX_DEPTH = 4
PRUNE = 0.05
P_NOISE = 0.1


# ---------- vectors and weights ----------

def embed(texts):
    """Task vectors from the fixed local text-embedding model (normalised)."""
    code = ("import json,sys;from sentence_transformers import SentenceTransformer as S;"
            f"m=S('{EMBED_MODEL}');t=json.load(sys.stdin);"
            "print(json.dumps(m.encode(t,normalize_embeddings=True).tolist()))")
    r = subprocess.run([str(EMBED_PY), "-c", code], input=json.dumps(texts), capture_output=True, text=True,
                       check=True)
    return json.loads(r.stdout)


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def child_weights(tree, vec, nid):
    kids = tree["nodes"][nid]["children"]
    if not kids:
        return {}
    logits = {k: -(dist(vec, tree["nodes"][k]["centre"]) / tree["scale"] / SIGMA) ** 2 / 2 for k in kids}
    if ABLATION == "hard":                       # TreeMAML: the nearest child only
        best = max(logits, key=logits.get)
        return {k: float(k == best) for k in kids}
    m = max(logits.values())
    z = sum(math.exp(v - m) for v in logits.values())
    return {k: math.exp(v - m) / z for k, v in logits.items()}


def leaf_weights(tree, vec, nid="root", w=1.0, out=None):
    out = {} if out is None else out
    kids = child_weights(tree, vec, nid)
    if not kids:
        out[nid] = out.get(nid, 0) + w
    for k, wk in kids.items():
        leaf_weights(tree, vec, k, w * wk, out)
    return out


def node_weight(tree, vec, nid):
    """Weight of a task on any node: sum of its leaf weights below that node."""
    below = set(descendants(tree, nid)) | {nid}
    return sum(w for l, w in leaf_weights(tree, vec).items() if l in below)


def descendants(tree, nid):
    out = []
    for k in tree["nodes"][nid]["children"]:
        out += [k] + descendants(tree, k)
    return out


def path(tree, nid):
    p = []
    while nid is not None:
        p.append(nid)
        nid = tree["nodes"][nid]["parent"]
    return p[::-1]


def path_config(tree, leaf):
    cfg = {}
    for n in path(tree, leaf):
        cfg.update(tree["nodes"][n]["config"])
    return cfg


def mixed_config(tree, vec):
    lw = leaf_weights(tree, vec)
    cfgs = {l: path_config(tree, l) for l in lw}
    keys = set().union(*cfgs.values()) if cfgs else set()
    top = max(lw, key=lw.get)
    out = {}
    for k in keys:
        vals = {l: c[k] for l, c in cfgs.items() if k in c}
        wsum = sum(lw[l] for l in vals)
        sample = next(iter(vals.values()))
        if isinstance(sample, bool):
            out[k] = sum(lw[l] for l, v in vals.items() if v) / wsum > 0.5
        elif isinstance(sample, (int, float)) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                     for v in vals.values()):
            avg = sum(lw[l] * v for l, v in vals.items()) / wsum
            out[k] = round(avg) if all(isinstance(v, int) for v in vals.values()) else avg
        else:
            out[k] = cfgs[top][k] if k in cfgs[top] else vals[max(vals, key=lambda l: lw[l])]
    return out


def assemble(shared_src, config):
    """The task's harness: the shared code with CONFIG filled in (one file, as the broker runs it)."""
    line = "CONFIG = " + json.dumps(config, sort_keys=True) + "\n"
    assert "\nCONFIG = {}" in shared_src, "shared code must keep the `CONFIG = {}` line"
    return shared_src.replace("\nCONFIG = {}", "\n" + line.rstrip("\n"), 1)


# ---------- the initial tree ----------

def build_tree(cells, vecs):
    """Agglomerative clustering (average linkage) of the development tasks, cut so that the tree has at
    most MAX_DEPTH levels; leaves hold one task each. Centres are the mean vectors of their tasks."""
    def mean(vs):
        return [sum(x) / len(vs) for x in zip(*vs)]
    clusters = [{"members": [c], "children": []} for c in cells]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = statistics.mean(dist(vecs[a], vecs[b]) for a in clusters[i]["members"]
                                    for b in clusters[j]["members"])
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        new = {"members": clusters[i]["members"] + clusters[j]["members"], "children": [clusters[i], clusters[j]],
               "height": d}
        merges.append(new)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [new]
    root = clusters[0]

    def flatten(c, depth):
        """Collapse binary merges into nodes with several children so the depth stays <= MAX_DEPTH."""
        if not c["children"] or depth >= MAX_DEPTH - 1:
            return {"members": c["members"], "children": []}
        kids = c["children"]
        if depth < MAX_DEPTH - 2:                      # absorb merges that are nearly as tall as this one
            changed = True
            while changed:
                changed = False
                for k in list(kids):
                    if k["children"] and k.get("height", 0) >= 0.85 * c.get("height", 0):
                        kids.remove(k)
                        kids += k["children"]
                        changed = True
        return {"members": c["members"], "children": [flatten(k, depth + 1) for k in kids]}

    flat = flatten(root, 0)
    if ABLATION == "flat":                        # no levels: every task a leaf right under the root
        flat = {"members": list(cells), "children": [{"members": [c], "children": []} for c in cells]}
    nodes, scale = {}, statistics.median(dist(vecs[a], vecs[b]) for a in cells for b in cells if a < b)

    def add(c, parent, name):
        nodes[name] = {"parent": parent, "children": [], "config": {}, "centre": mean([vecs[m] for m in c["members"]]),
                       "members": c["members"]}
        if parent is not None:
            nodes[parent]["children"].append(name)
        if not c["children"] and len(c["members"]) > 1:        # depth cap reached: one leaf per task
            for k, m in enumerate(c["members"]):
                add({"members": [m], "children": []}, name, f"{name}.{k + 1}")
        for k, ch in enumerate(c["children"]):
            add(ch, name, f"{name}.{k + 1}" if name != "root" else f"n{k + 1}")

    add(flat, None, "root")
    return {"nodes": nodes, "scale": scale or 1.0, "vecs": vecs}


def describe(cell):
    """What the method may know about a development task before playing: its instructions as the
    environment gives them (the first episode's first observation, from the c000 traces)."""
    return cell   # replaced by the instructions text at run time (see task_texts)


# ---------- noise test ----------

def sign_test_p(wins, losses):
    """One-sided sign test p-value for `wins` > `losses` (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def paired(new, old):
    """(wins, losses) of new over old on the same problems."""
    w = l = 0
    for i, e in new.items():
        if i in old and e["score"] is not None and old[i]["score"] is not None:
            w += e["score"] > old[i]["score"]
            l += e["score"] < old[i]["score"]
    return w, l


# ---------- one run ----------

class Run:
    def __init__(self, a):
        self.a, self.dir = a, RUNS / a.run
        self.cells = a.cells.split(",")
        self.box = remote.make(a.remote)
        (self.dir / "state").mkdir(parents=True, exist_ok=True)
        (self.dir / "private").mkdir(exist_ok=True)

    def versions(self):
        return sorted(p for p in (self.dir / "state").iterdir() if (p / "tree.json").exists())

    def load(self, v):
        return (v / "shared.py").read_text(), json.loads((v / "tree.json").read_text())

    def save(self, name, shared, tree, meta):
        d = self.dir / "state" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "shared.py").write_text(shared)
        (d / "tree.json").write_text(json.dumps(tree, indent=1))
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return d

    def harnesses(self, shared, tree, folder):
        folder.mkdir(parents=True, exist_ok=True)
        out = {}
        for c in self.cells:
            src = assemble(shared, mixed_config(tree, tree["vecs"][c]))
            bad = problems(src)
            if bad:
                return None, bad
            p = folder / f"{c.replace(':', '_')}.py"
            p.write_text(src)
            out[c] = p
        return out, []

    def score(self, key, harness_map, split="search", n=None, export=True):
        n = n or self.a.n
        params = {"cell": ",".join(self.cells), "split": split, "n": n, "export": export, "smoke": False}
        self.box.submit(key, harness_map, params)
        while (r := self.box.result(key)) is None:
            time.sleep(30)
        return r

    def per_task(self, key, dest):
        """Per-task scores.json from the exported folder (cells/<slug>/scores.json)."""
        self.box.fetch_export(key, dest)
        return {c: json.loads((dest / "cells" / c.replace(":", "_") / "scores.json").read_text())
                for c in self.cells}

    def task_texts(self, dest):
        """Instructions of each task as the environment shows them (first user message of a trace)."""
        out = {}
        for c in self.cells:
            traces = sorted((dest / "cells" / c.replace(":", "_") / "episodes").glob("*/trace.md"))
            t = traces[0].read_text() if traces else c
            # the first message the model saw; our header lines (episode id, score) are dropped
            body = t.split("**user:**", 1)[-1] if "**user:**" in t else t
            out[c] = body.strip().strip("`").strip()[:4000]
        return out

    # ----- the loop -----
    def loop(self):
        a = self.a
        vs = self.versions()
        if not vs:
            shared = (HERE / "seeds" / "soft_tree_shared.py").read_text()
            flat = {"nodes": {"root": {"parent": None, "children": [], "config": {}, "centre": [0.0]}},
                    "scale": 1.0, "vecs": {c: [0.0] for c in self.cells}}
            hm, bad = self.harnesses(shared, flat, self.dir / "tmp" / "v000")
            key = f"{a.run}/v000/search"
            r = self.score(key, hm)
            exp = self.dir / "exports" / "v000"
            scores = self.per_task(key, exp)
            texts = self.task_texts(exp)
            vecs = dict(zip(self.cells, embed([texts[c] for c in self.cells])))
            tree = build_tree(self.cells, vecs)
            self.save("v000", shared, tree, {"round": 0, "node": None, "export": "v000",
                                             "scores": {c: s["mean_score"] for c, s in scores.items()},
                                             "per_episode": {c: s["per_episode"] for c, s in scores.items()}})
            print("v000:", {c: s["mean_score"] for c, s in scores.items()}, "tree:",
                  {k: v["children"] for k, v in tree["nodes"].items()}, flush=True)
        for rnd in range(1, a.rounds + 1):
            if (self.dir / "rounds" / f"r{rnd:02d}.json").exists():
                continue
            # early stop, as every method (notes/exp_b_design.md): 3 rounds in a row with no change kept
            # (a "kept" or "split" passed the paired noise test; anything else is no gain)
            last = [json.loads((self.dir / "rounds" / f"r{k:02d}.json").read_text())["outcome"]
                    for k in range(max(1, rnd - 3), rnd)]
            if len(last) == 3 and not any(o.startswith(("kept", "split")) for o in last):
                print(f"early stop before r{rnd:02d}: 3 rounds in a row without a gain beyond noise", flush=True)
                break
            cur = self.versions()[-1]
            shared, tree = self.load(cur)
            meta = json.loads((cur / "meta.json").read_text())
            node = self.choose_node(tree, meta, rnd)
            ws = self.dir / "workspaces" / f"r{rnd:02d}"
            done_p = self.dir / "private" / f"r{rnd:02d}_proposed.json"
            t0 = time.time()
            if done_p.exists() and json.loads(done_p.read_text()).get("node") == node and ws.exists():
                rc = json.loads(done_p.read_text())["rc"]    # resume: proposer finished, score only
                print(f"r{rnd:02d}: resuming after the proposer (scoring only)", flush=True)
            else:
                ws = self.workspace(rnd, node, shared, tree, cur)
                log = self.dir / "private" / f"r{rnd:02d}_proposer.jsonl"
                sys.path.insert(0, str(SANDBOX))
                import run_proposer
                rc = run_proposer.run(ws, log)
                (self.dir / "private").mkdir(exist_ok=True)
                done_p.write_text(json.dumps({"node": node, "rc": rc}, default=str))
            new_shared = (ws / "shared.py").read_text()
            try:
                new_cfg = json.loads((ws / "node_config.json").read_text())
            except Exception as e:                       # a broken config file is a rejected change
                new_cfg, rc = tree["nodes"][node]["config"], f"bad node_config.json: {e!r}"[:200]
            cand = json.loads(json.dumps(tree))
            cand["nodes"][node]["config"] = new_cfg
            hm, bad = self.harnesses(new_shared, cand, self.dir / "tmp" / f"r{rnd:02d}")
            rec = {"round": rnd, "node": node, "proposer_rc": rc, "proposer_seconds": round(time.time() - t0)}
            if bad or (new_shared == shared and new_cfg == tree["nodes"][node]["config"]):
                rec["outcome"] = "rejected: " + ("; ".join(bad)[:300] if bad else "no change")
                self.record(rnd, rec)
                continue
            key = f"{a.run}/r{rnd:02d}/search"
            r = self.score(key, hm)
            if "rejected" in r:
                rec["outcome"] = "rejected on the cluster: " + "; ".join(r["rejected"])[:300]
                self.record(rnd, rec)
                continue
            exp = self.dir / "exports" / f"r{rnd:02d}"
            new = self.per_task(key, exp)
            rec.update(self.decide(rnd, node, shared, tree, meta, new_shared, cand, new))
            self.record(rnd, rec)
            print(f"r{rnd:02d} node {node}: {rec['outcome']}", flush=True)

    def record(self, rnd, rec):
        (self.dir / "rounds").mkdir(exist_ok=True)
        (self.dir / "rounds" / f"r{rnd:02d}.json").write_text(json.dumps(rec, indent=1))

    def choose_node(self, tree, meta, rnd):
        by_depth = {}
        for n in tree["nodes"]:
            by_depth.setdefault(len(path(tree, n)) - 1, []).append(n)
        depth = (rnd - 1) % len(by_depth)

        def mass(n):
            return sum(node_weight(tree, tree["vecs"][c], n) * (1 - (meta["scores"][c] or 0)) for c in self.cells)
        return max(by_depth[depth], key=mass)

    def workspace(self, rnd, node, shared, tree, cur):
        ws = self.dir / "workspaces" / f"r{rnd:02d}"
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        for f in ("harness_lib.py", "mock_env.py"):
            shutil.copy(HERE / "ws_template" / f, ws / f)
        shutil.copy(HERE / "check.py", ws / "check_harness.py")
        (ws / "shared.py").write_text(shared)
        (ws / "node_config.json").write_text(json.dumps(tree["nodes"][node]["config"], indent=1))
        # the inherited configuration along this node's path, read-only
        inherited = {}
        for n in path(tree, node)[:-1]:
            inherited.update(tree["nodes"][n]["config"])
        (ws / "inherited_config.json").write_text(json.dumps(inherited, indent=1))
        # evidence: the latest scored export of every task, more episodes for tasks that weigh the node more
        last = self.dir / "exports" / json.loads((cur / "meta.json").read_text())["export"]
        ev = ws / "evidence"
        weights = {c: node_weight(tree, tree["vecs"][c], node) for c in self.cells}
        for k, c in enumerate(sorted(self.cells, key=lambda c: -weights[c])):
            w = weights[c]
            if w < 0.05:
                continue
            src = last / "cells" / c.replace(":", "_")
            dst = ev / f"task{k + 1}"
            dst.mkdir(parents=True)
            shutil.copy(src / "scores.json", dst / "scores.json")
            eps = sorted((src / "episodes").iterdir())
            for e in eps[:max(2, round(w * len(eps)))]:
                shutil.copytree(e, dst / "episodes" / e.name)
            (dst / "weight.txt").write_text(f"{w:.2f}\n")
        (ws / "TASK.md").write_text(TASK.format(node=node, n=self.a.n, depth=len(path(tree, node)) - 1))
        return ws

    def decide(self, rnd, node, shared, tree, meta, new_shared, cand, new):
        old = meta["per_episode"]
        tests = {}
        for c in self.cells:
            w, l = paired(new[c]["per_episode"], old[c])
            tests[c] = {"wins": w, "losses": l, "gain_p": sign_test_p(w, l), "drop_p": sign_test_p(l, w)}
        weights = {c: node_weight(tree, tree["vecs"][c], node) for c in self.cells}
        mine = [c for c in self.cells if weights[c] >= 0.2]
        gained = [c for c in mine if tests[c]["gain_p"] < P_NOISE]
        dropped = [c for c in self.cells if tests[c]["drop_p"] < P_NOISE]
        dw = sum(weights[c] * (tests[c]["wins"] - tests[c]["losses"]) for c in mine)
        scores = {c: new[c]["mean_score"] for c in self.cells}
        per = {c: new[c]["per_episode"] for c in self.cells}
        out = {"tests": tests, "weights": weights, "scores": scores}
        code_changed = new_shared != shared
        pooled = sign_test_p(sum(tests[c]["wins"] for c in mine), sum(tests[c]["losses"] for c in mine))
        if not dropped and (gained or (dw > 0 and pooled < P_NOISE)):
            tree2 = cand
            self.update_centres(tree2, tests, weights)
            self.save(f"v{rnd:03d}", new_shared, tree2, {"round": rnd, "node": node, "export": f"r{rnd:02d}",
                                                         "scores": scores, "per_episode": per})
            out["outcome"] = f"kept (gained: {gained})"
        elif gained and dropped and not code_changed and len(path(tree, node)) < MAX_DEPTH:
            # split: the new configuration goes to a new child for the tasks that gained
            tree2 = json.loads(json.dumps(tree))
            kid = f"{node}.s{rnd}"
            tree2["nodes"][kid] = {"parent": node, "children": [], "config": cand["nodes"][node]["config"],
                                   "centre": [sum(x) / len(gained) for x in zip(*[tree["vecs"][c] for c in gained])],
                                   "members": gained}
            tree2["nodes"][node]["children"].append(kid)
            self.prune(tree2)
            self.save(f"v{rnd:03d}", shared, tree2, {"round": rnd, "node": node, "scores": meta["scores"],
                                                    "export": meta["export"], "per_episode": old,
                                                    "split_pending_rescore": True})
            out["outcome"] = f"split: new child {kid} for {gained}; {dropped} dropped"
        else:
            out["outcome"] = f"rejected (gained {gained}, dropped {dropped})"
        return out

    def update_centres(self, tree, tests, weights):
        for n, nd in tree["nodes"].items():
            ws = {c: node_weight(tree, tree["vecs"][c], n) * max(0, tests[c]["wins"] - tests[c]["losses"] + 1)
                  for c in self.cells}
            tot = sum(ws.values())
            if tot > 0 and n != "root":
                nd["centre"] = [sum(ws[c] * tree["vecs"][c][i] for c in self.cells) / tot
                                for i in range(len(nd["centre"]))]
        self.prune(tree)

    def prune(self, tree):
        for n in list(tree["nodes"]):
            if n == "root" or n not in tree["nodes"]:
                continue
            tot = sum(node_weight(tree, tree["vecs"][c], n) for c in self.cells)
            if tot < PRUNE and not tree["nodes"][n]["children"]:
                p = tree["nodes"][n]["parent"]
                tree["nodes"][p]["children"].remove(n)
                del tree["nodes"][n]

    # ----- after the loop -----
    def final(self):
        """Top 3 versions by mean search score -> selection split (20 per task) -> the chosen version."""
        vs = [v for v in self.versions() if not json.loads((v / "meta.json").read_text()).get("split_pending_rescore")]
        ms = {v.name: statistics.mean(s for s in json.loads((v / "meta.json").read_text())["scores"].values()
                                      if s is not None) for v in vs}
        top = sorted(ms, key=ms.get, reverse=True)[:3]
        sel = {}
        for name in top:
            shared, tree = self.load(self.dir / "state" / name)
            hm, _ = self.harnesses(shared, tree, self.dir / "tmp" / f"sel_{name}")
            r = self.score(f"{self.a.run}/{name}/selection", hm, split="selection", n=self.a.n_selection, export=False)
            sel[name] = r["summary"]["mean_score"] if "summary" in r else -1
        best = max(sel, key=sel.get)
        (self.dir / "chosen.json").write_text(json.dumps({"best": best, "selection": sel, "search": ms}, indent=1))
        print("selection:", sel, "best", best, flush=True)


TASK = """# Improve one node of a shared harness tree

A frozen open model plays several text games through a harness: the code around the model (prompts, what
history it sees, how its reply becomes an action). All games share one code base, `shared.py`. Its behaviour
is set by `CONFIG`; every game gets its own CONFIG, mixed from a tree of configurations: the root holds
what every game uses, lower nodes hold what fewer games share.

You are improving node `{node}` (depth {depth} in the tree). The games below it are in `evidence/` (task1, task2,
... ordered by how much they belong to this node; `weight.txt` gives that weight). For each: `scores.json` has
its mean score and every episode's result ({n} episodes, one per problem), and `episodes/<id>/trace.md` shows
everything the model saw and did.

You may change two things:
1. `node_config.json`: this node's CONFIG overrides (JSON). They apply to every game below the node, on top of
   `inherited_config.json` (read-only). Use the keys `shared.py` reads (see its docstring and DEFAULT_CONFIG).
2. `shared.py`: add a new module or improve one, **always behind a CONFIG key whose default in
   DEFAULT_CONFIG keeps the current behaviour exactly**, so games that do not switch it on are unaffected.
   Keep the `CONFIG = {{}}` line. Then switch it on in `node_config.json` if this node's games need it.

Rules: never write game names, benchmark names or answers anywhere; the change must make sense for any game
of this kind. One mechanism per change, as small as possible. `check_harness.py` applies the static rules
(allowed imports: re, json, math, itertools, collections, random, string, functools, heapq, typing,
dataclasses, harness_lib); `mock_env.py` runs a harness against a fake game to check it runs.

How to read the evidence: tabulate how episodes end (win, out of turns, invalid move, error) before reading
traces; read the last steps of failing traces; a change worth making moves several episodes, not one.
Write a short `proposal.md`: what you saw, what you changed, and why it should help the games below this node.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--remote", default="mailbox:~/evo_b_mailbox")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-selection", type=int, default=20)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--ablation", default="", choices=["", "hard", "flat"])
    a = ap.parse_args()
    global ABLATION
    ABLATION = a.ablation
    run = Run(a)
    run.final() if a.final else run.loop()


if __name__ == "__main__":
    main()
