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
SIGMA = 0.3          # calibrated on the development tasks' real vectors (user, 2026-11-26): own leaf 0.55-1.0
ABLATION = ""        # "" (ours) | "hard": one path per task (TreeMAML) | "flat": no levels, soft weights only
MAX_DEPTH = 100         # no depth cap: the data decides how many levels (user, 2026-11-26)
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
    # a Python literal (True/False/None), not JSON (true/false/null), which Python cannot read
    line = "CONFIG = " + repr(dict(sorted(config.items()))) + "\n"
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
        if not c["children"] or depth >= MAX_DEPTH - 2:   # its members become one-task leaves one level down
            return {"members": c["members"], "children": []}
        kids = c["children"]
        if depth < MAX_DEPTH - 3:                      # absorb merges that are nearly as tall as this one
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
            # the first trace that shows a model call (an episode killed before any call has no instructions)
            t = next((x.read_text() for x in traces if "**user:**" in x.read_text()), c)
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
            ess = self.essence(texts, exp)               # Opus: one line per game, what the problem is
            vecs = dict(zip(self.cells, embed([ess.get(c) or texts[c] for c in self.cells])))
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
            if len(last) == 3 and not any(o.startswith(("kept", "split", "factored", "regrouped")) for o in last):
                print(f"early stop before r{rnd:02d}: 3 rounds in a row without a gain beyond noise", flush=True)
                break
            cur = self.versions()[-1]
            shared, tree = self.load(cur)
            meta = json.loads((cur / "meta.json").read_text())
            if meta.get("pending_factor"):                       # the round after a split: separate it
                rec = self.factor_round(rnd, shared, tree, meta, cur)
            else:
                rec = None
                done_regroup = any(json.loads(f.read_text()).get("kind") == "regroup"
                                   for f in (self.dir / "rounds").glob("r*.json")) if (self.dir / "rounds").exists() else False
                if rnd >= 3 and not done_regroup and ABLATION != "flat":   # once, from round 3 on
                    rec = self.regroup_round(rnd, shared, tree, meta, cur)
                if rec is None:
                    rec = self.edit_round(rnd, shared, tree, meta, cur)
            self.record(rnd, rec)
            print(f"r{rnd:02d}: {rec['outcome']}", flush=True)

    def essence(self, texts, exp):
        """One session before the tree is built: the model reads each game's instructions and a few traces and
        writes what the problem is, in terms of the problem, not the game; those lines are embedded."""
        out_f = self.dir / "state" / "essence.json"
        if out_f.exists():
            return json.loads(out_f.read_text())
        ws = self.dir / "workspaces" / "r00e"
        shutil.rmtree(ws, ignore_errors=True)
        (ws / "games").mkdir(parents=True)
        for c in self.cells:
            d = ws / "games" / self.tid(c)
            d.mkdir()
            (d / "instructions.md").write_text(texts[c])
            eps = sorted((exp / "cells" / c.replace(":", "_") / "episodes").iterdir())[:3]
            for e in eps:
                if (e / "trace.md").exists():
                    shutil.copy(e / "trace.md", d / f"{e.name}_trace.md")
        (ws / "TASK.md").write_text(TASK_ESSENCE)
        self.propose(0, ws, "e")
        try:
            got = json.loads((ws / "essence.json").read_text())
        except Exception:
            got = {}
        ids = {self.tid(c): c for c in self.cells}
        ess = {ids[k]: v.strip()[:500] for k, v in got.items() if k in ids and isinstance(v, str) and v.strip()}
        out_f.write_text(json.dumps(ess, indent=1))
        return ess

    # ----- round kinds -----
    def propose(self, rnd, ws, tag=""):
        """One code-editing session in `ws` (resumable: a finished session is not run again)."""
        done_p = self.dir / "private" / f"r{rnd:02d}{tag}_proposed.json"
        if done_p.exists() and ws.exists():
            print(f"r{rnd:02d}{tag}: resuming after the proposer", flush=True)
            return json.loads(done_p.read_text())["rc"]
        sys.path.insert(0, str(SANDBOX))
        import run_proposer
        rc = run_proposer.run(ws, self.dir / "private" / f"r{rnd:02d}{tag}_proposer.jsonl")
        done_p.write_text(json.dumps({"rc": rc}, default=str))
        return rc

    def evaluate(self, rnd, shared, cand, tag=""):
        hm, bad = self.harnesses(shared, cand, self.dir / "tmp" / f"r{rnd:02d}{tag}")
        if bad:
            return None, "; ".join(bad)[:300]
        r = self.score(f"{self.a.run}/r{rnd:02d}{tag}/search", hm)
        if "rejected" in r:
            return None, "rejected on the cluster: " + "; ".join(r["rejected"])[:300]
        exp = self.dir / "exports" / f"r{rnd:02d}{tag}"
        return self.per_task(f"{self.a.run}/r{rnd:02d}{tag}/search", exp), None

    def edit_round(self, rnd, shared, tree, meta, cur):
        node = self.choose_node(tree, meta, rnd)
        ws = self.dir / "workspaces" / f"r{rnd:02d}"
        resumed = (self.dir / "private" / f"r{rnd:02d}_proposed.json").exists() and ws.exists()
        if not resumed:
            ws = self.workspace(rnd, node, shared, tree, cur)
        t0 = time.time()
        rc = self.propose(rnd, ws)
        rec = {"round": rnd, "kind": "edit", "node": node, "proposer_rc": rc, "proposer_seconds": round(time.time() - t0)}
        new_shared = (ws / "shared.py").read_text()
        try:
            new_cfg = json.loads((ws / "node_config.json").read_text())
        except Exception as e:                           # a broken config file is a rejected change
            rec["outcome"] = f"rejected: bad node_config.json: {e!r}"[:200]
            return rec
        # where the model says the change belongs (neural), within this node's path (checked by code)
        try:
            place = json.loads((ws / "placement.json").read_text())
        except Exception:
            place = {}
        target = place.get("target_node") if place.get("target_node") in path(tree, node) else node
        if ABLATION == "flat" and target != node:
            target = "root" if target == "root" else node
        rec["placement"] = {"claimed": place, "target": target}
        desc = (ws / "node_description.txt").read_text().strip()[:300] if (ws / "node_description.txt").exists() else ""
        cand = json.loads(json.dumps(tree))
        old_cfg = tree["nodes"][node]["config"]
        diff = {k: v for k, v in new_cfg.items() if old_cfg.get(k) != v}
        if target != node:                               # move the change up to the claimed ancestor
            cand["nodes"][target]["config"].update(diff)
        else:
            cand["nodes"][node]["config"] = new_cfg
        if desc:
            cand["nodes"][target]["desc"] = desc
        if new_shared == shared and not diff:
            rec["outcome"] = "rejected: no change"
            return rec
        new, err = self.evaluate(rnd, new_shared, cand)
        if new is None:
            rec["outcome"] = "rejected: " + err
            return rec
        rec.update(self.decide(rnd, target, shared, tree, meta, new_shared, cand, new,
                               specific=place.get("level") == "specific"))
        return rec

    def factor_round(self, rnd, shared, tree, meta, cur):
        """After a split: the model separates the change into a general part (at the parent, tested on all its
        games, kept only if none gets worse) and a specific part (the child)."""
        pf = meta["pending_factor"]
        node, child = pf["node"], pf["child"]
        ws = self.dir / "workspaces" / f"r{rnd:02d}f"
        if not ((self.dir / "private" / f"r{rnd:02d}f_proposed.json").exists() and ws.exists()):
            ws = self.workspace(rnd, node, shared, tree, cur, sub="f")
            (ws / "before").mkdir()
            (ws / "before" / "parent_config.json").write_text(json.dumps(pf["parent_before"], indent=1))
            (ws / "parent_config.json").write_text(json.dumps(tree["nodes"][node]["config"], indent=1))
            (ws / "child_config.json").write_text(json.dumps(tree["nodes"][child]["config"], indent=1))
            (ws / "node_config.json").unlink(missing_ok=True)
            (ws / "TASK.md").write_text(TASK_FACTOR.format(node=node, child=child,
                                                           gained=", ".join(self.tid(c) for c in pf["gained"]),
                                                           others=", ".join(self.tid(c) for c in pf["others"]) or "-"))
        rc = self.propose(rnd, ws, "f")
        rec = {"round": rnd, "kind": "factor", "node": node, "child": child, "proposer_rc": rc}
        try:
            pcfg = json.loads((ws / "parent_config.json").read_text())
            ccfg = json.loads((ws / "child_config.json").read_text())
        except Exception as e:
            rec["outcome"] = f"rejected: bad config file: {e!r}"[:200]
            self.clear_pending(cur, meta)
            return rec
        new_shared = (ws / "shared.py").read_text()
        cand = json.loads(json.dumps(tree))
        cand["nodes"][node]["config"] = pcfg
        if child in cand["nodes"]:
            cand["nodes"][child]["config"] = ccfg
        if new_shared == shared and pcfg == tree["nodes"][node]["config"] and ccfg == tree["nodes"][child]["config"]:
            rec["outcome"] = "rejected: no change (nothing general)"
            self.clear_pending(cur, meta)
            return rec
        new, err = self.evaluate(rnd, new_shared, cand, "f")
        if new is None:
            rec["outcome"] = "rejected: " + err
            self.clear_pending(cur, meta)
            return rec
        tests = self.tests(new, meta["per_episode"])
        dropped = [c for c in self.cells if tests[c]["drop_p"] < P_NOISE]
        if dropped:
            rec["outcome"] = f"rejected: the general part hurt {dropped}"
            self.clear_pending(cur, meta)
        else:
            gained = [c for c in self.cells if tests[c]["gain_p"] < P_NOISE]
            self.update_centres(cand, tests, {c: node_weight(cand, cand["vecs"][c], node) for c in self.cells})
            self.save(f"v{rnd:03d}", new_shared, cand, {"round": rnd, "node": node, "export": f"r{rnd:02d}f",
                                                         "scores": {c: new[c]["mean_score"] for c in self.cells},
                                                         "per_episode": {c: new[c]["per_episode"] for c in self.cells}})
            rec["outcome"] = f"factored: general part kept at {node} (gained {gained}, none hurt)"
        rec["tests"] = tests
        return rec

    def clear_pending(self, cur, meta):
        meta = dict(meta)
        meta.pop("pending_factor", None)
        (cur / "meta.json").write_text(json.dumps(meta, indent=1))

    def regroup_round(self, rnd, shared, tree, meta, cur):
        """The model reviews the grouping; moves are kept only if the regrouped tree hurts no game.
        Returns None when it proposes no move (the round then becomes an edit round)."""
        ws = self.dir / "workspaces" / f"r{rnd:02d}g"
        if not ((self.dir / "private" / f"r{rnd:02d}g_proposed.json").exists() and ws.exists()):
            ws = self.workspace(rnd, "root", shared, tree, cur, sub="g")
            for f in ("node_config.json", "inherited_config.json"):
                (ws / f).unlink(missing_ok=True)
            (ws / "TASK.md").write_text(TASK_REGROUP)
        rc = self.propose(rnd, ws, "g")
        try:
            prop = json.loads((ws / "regroup.json").read_text())
        except Exception:
            prop = {}
        cand = json.loads(json.dumps(tree))
        for n, d in (prop.get("descriptions") or {}).items():
            if n in cand["nodes"] and isinstance(d, str):
                cand["nodes"][n]["desc"] = d[:300]
        ids = {self.tid(c): c for c in self.cells}
        moves = [m for m in prop.get("moves") or [] if m.get("task") in ids and m.get("to") in cand["nodes"]]
        if not moves:
            if cand != tree:                                 # descriptions only: no behaviour change
                self.save(cur.name, shared, cand, meta)
            return None
        for m in moves:                                      # pull the target node's centre towards the game
            c, to = ids[m["task"]], m["to"]
            mem = set(cand["nodes"][to].get("members", [])) | {c}
            cand["nodes"][to]["members"] = sorted(mem)
            for n in path(cand, to)[1:]:
                nm = set(cand["nodes"][n].get("members", [])) | {c}
                cand["nodes"][n]["members"] = sorted(nm)
                cand["nodes"][n]["centre"] = [sum(cand["vecs"][x][i] for x in nm) / len(nm)
                                              for i in range(len(cand["vecs"][c]))]
        new, err = self.evaluate(rnd, shared, cand, "g")
        rec = {"round": rnd, "kind": "regroup", "proposer_rc": rc, "moves": moves}
        if new is None:
            rec["outcome"] = "rejected: " + err
            return rec
        tests = self.tests(new, meta["per_episode"])
        dropped = [c for c in self.cells if tests[c]["drop_p"] < P_NOISE]
        rec["tests"] = tests
        if dropped:
            rec["outcome"] = f"rejected: regrouping hurt {dropped}"
        else:
            self.save(f"v{rnd:03d}", shared, cand, {"round": rnd, "node": "root", "export": f"r{rnd:02d}g",
                                                     "scores": {c: new[c]["mean_score"] for c in self.cells},
                                                     "per_episode": {c: new[c]["per_episode"] for c in self.cells}})
            rec["outcome"] = f"regrouped: {[(m['task'], m['to']) for m in moves]}"
        return rec

    def record(self, rnd, rec):
        (self.dir / "rounds").mkdir(exist_ok=True)
        (self.dir / "rounds" / f"r{rnd:02d}.json").write_text(json.dumps(rec, indent=1))

    def choose_node(self, tree, meta, rnd):
        """The node where failure is largest, by data: mean weighted failure of its games x sqrt(their weight),
        so severity and reach both count; a node edited in the previous round is halved, so the same node is
        not chosen again and again. The model may still move its change up the path (placement.json)."""
        last = None
        if rnd > 1 and (self.dir / "rounds" / f"r{rnd - 1:02d}.json").exists():
            last = json.loads((self.dir / "rounds" / f"r{rnd - 1:02d}.json").read_text()).get("node")

        def score(n):
            ws_ = {c: node_weight(tree, tree["vecs"][c], n) for c in self.cells}
            tot = sum(ws_.values())
            if tot < 1e-9:
                return -1
            fail = sum(w * (1 - (meta["scores"][c] or 0)) for c, w in ws_.items()) / tot
            return fail * math.sqrt(tot) * (0.5 if n == last else 1.0)
        return max(tree["nodes"], key=score)

    def workspace(self, rnd, node, shared, tree, cur, sub=""):
        ws = self.dir / "workspaces" / f"r{rnd:02d}{sub}"
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
        for c in sorted(self.cells, key=lambda c: -weights[c]):
            w = weights[c]
            if w < 0.05:
                continue
            src = last / "cells" / c.replace(":", "_")
            dst = ev / self.tid(c)
            dst.mkdir(parents=True)
            shutil.copy(src / "scores.json", dst / "scores.json")
            eps = sorted((src / "episodes").iterdir())
            for e in eps[:max(2, round(w * len(eps)))]:
                shutil.copytree(e, dst / "episodes" / e.name)
            (dst / "weight.txt").write_text(f"{w:.2f}\n")
        (ws / "tree.md").write_text(self.tree_md(tree, cur))
        (ws / "TASK.md").write_text(TASK.format(node=node, n=self.a.n, depth=len(path(tree, node)) - 1))
        return ws

    # ----- shared by the three kinds of round -----
    def tid(self, c):
        """A game's id in every workspace (stable across rounds; never its name)."""
        return f"task{self.cells.index(c) + 1}"

    def tree_md(self, tree, cur):
        """The tree as the code-editing model sees it: descriptions, games below each node, how they end."""
        meta = json.loads((cur / "meta.json").read_text())
        last = self.dir / "exports" / meta["export"]
        ends = {}
        for c in self.cells:
            f = last / "cells" / c.replace(":", "_") / "scores.json"
            per = json.loads(f.read_text()).get("per_episode", {}) if f.exists() else {}
            cnt = {}
            for e in per.values():
                k = e.get("end") or "?"
                k = "win" if e.get("success") else k
                cnt[k] = cnt.get(k, 0) + 1
            ends[c] = cnt
        lines = ["# The tree (node: description; games below it with their weight on the node)", ""]
        for n in sorted(tree["nodes"], key=lambda n: (len(path(tree, n)), n)):
            nd = tree["nodes"][n]
            ws_ = {c: node_weight(tree, tree["vecs"][c], n) for c in self.cells}
            below = ", ".join(f"{self.tid(c)} ({w:.2f})" for c, w in sorted(ws_.items(), key=lambda x: -x[1]) if w >= 0.05)
            lines.append(f"- `{n}` (depth {len(path(tree, n)) - 1}, parent {nd['parent']}): "
                         f"{nd.get('desc') or '(no description yet)'}; config {json.dumps(nd['config'])}; games: {below}")
        lines += ["", "## How each game's episodes end (latest scored version)", ""]
        for c in self.cells:
            lines.append(f"- {self.tid(c)}: mean score {meta['scores'].get(c)}; ends {json.dumps(ends[c])}")
        return "\n".join(lines) + "\n"


    def tests(self, new, old):
        out = {}
        for c in self.cells:
            w, l = paired(new[c]["per_episode"], old[c])
            out[c] = {"wins": w, "losses": l, "gain_p": sign_test_p(w, l), "drop_p": sign_test_p(l, w)}
        return out

    def decide(self, rnd, node, shared, tree, meta, new_shared, cand, new, specific=False):
        """Data decides where the change stays. Kept at `node` only if no game gets worse and the games that
        gained carry most of the node's weight (or the node is a leaf); if only some of its games gained,
        the change moves down to a new child for those games (split), and the next round separates its
        general part. `specific` is the model's own claim; it is recorded, the test decides."""
        old = meta["per_episode"]
        tests = self.tests(new, old)
        weights = {c: node_weight(tree, tree["vecs"][c], node) for c in self.cells}
        mine = [c for c in self.cells if weights[c] >= 0.2]
        gained = [c for c in mine if tests[c]["gain_p"] < P_NOISE]
        dropped = [c for c in self.cells if tests[c]["drop_p"] < P_NOISE]
        outside_dropped = [c for c in dropped if c not in mine]
        dw = sum(weights[c] * (tests[c]["wins"] - tests[c]["losses"]) for c in mine)
        pooled = sign_test_p(sum(tests[c]["wins"] for c in mine), sum(tests[c]["losses"] for c in mine))
        scores = {c: new[c]["mean_score"] for c in self.cells}
        per = {c: new[c]["per_episode"] for c in self.cells}
        out = {"tests": tests, "weights": weights, "scores": scores, "claimed_specific": specific}
        is_leaf = not tree["nodes"][node]["children"]
        share = sum(weights[c] for c in gained) / max(sum(weights[c] for c in mine), 1e-9)
        broad = is_leaf or share >= 0.5 or (not gained and dw > 0 and pooled < P_NOISE)
        if not dropped and (gained or (dw > 0 and pooled < P_NOISE)) and broad:
            tree2 = cand
            self.update_centres(tree2, tests, weights)
            self.save(f"v{rnd:03d}", new_shared, tree2, {"round": rnd, "node": node, "export": f"r{rnd:02d}",
                                                         "scores": scores, "per_episode": per})
            out["outcome"] = f"kept at {node} (gained: {gained})"
        elif gained and not outside_dropped and (home := self.existing_home(tree, node, gained, mine)):
            # an existing node below already holds the games that gained (and few others): the change moves
            # there; games outside it get their old configuration back
            tree2 = json.loads(json.dumps(tree))
            diff = {k: v for k, v in cand["nodes"][node]["config"].items() if tree["nodes"][node]["config"].get(k) != v}
            tree2["nodes"][home]["config"].update(diff)
            if cand["nodes"][node].get("desc"):
                tree2["nodes"][home]["desc"] = cand["nodes"][node]["desc"]
            self.save(f"v{rnd:03d}", new_shared, tree2, {
                "round": rnd, "node": home, "scores": meta["scores"], "export": meta["export"], "per_episode": old,
                "split_pending_rescore": True})
            out["outcome"] = f"split: moved down to existing node {home} for {gained}; dropped {dropped}"
        elif gained and not outside_dropped and len(path(tree, node)) < MAX_DEPTH:
            # split: the change goes to a new child for the games that gained; the node keeps its old config.
            # New shared code stays (its modules are off by default; no game outside the node got worse).
            tree2 = json.loads(json.dumps(tree))
            kid = f"{node}.s{rnd}"
            tree2["nodes"][kid] = {"parent": node, "children": [], "config": cand["nodes"][node]["config"],
                                   "centre": [sum(x) / len(gained) for x in zip(*[tree["vecs"][c] for c in gained])],
                                   "members": gained, "desc": cand["nodes"][node].get("desc", "")}
            tree2["nodes"][node]["children"].append(kid)
            self.prune(tree2)
            others = [c for c in mine if c not in gained]
            self.save(f"v{rnd:03d}", new_shared, tree2, {
                "round": rnd, "node": node, "scores": meta["scores"], "export": meta["export"], "per_episode": old,
                "split_pending_rescore": True,
                "pending_factor": {"node": node, "child": kid, "gained": gained, "others": others,
                                   "parent_before": tree["nodes"][node]["config"]}})
            out["outcome"] = f"split: new child {kid} for {gained}; others {others}; dropped {dropped}"
        else:
            out["outcome"] = f"rejected (gained {gained}, dropped {dropped})"
        return out

    def existing_home(self, tree, node, gained, mine):
        """The lowest node below `node` that clearly belongs to the games that gained: each of them weighs at
        least 0.25 on it and every other game of `node` weighs at most half the smallest of those; None if
        there is none. (Soft weights spread, so the test is relative.)"""
        best = None
        for d in descendants(tree, node):
            wg = [node_weight(tree, tree["vecs"][c], d) for c in gained]
            wo = [node_weight(tree, tree["vecs"][c], d) for c in mine if c not in gained]
            if min(wg) >= 0.25 and (not wo or max(wo) <= 0.5 * min(wg)):
                if best is None or len(path(tree, d)) > len(path(tree, best)):
                    best = d
        return best

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
of this kind. One mechanism per change, as small as possible.

Write mechanisms for the problem, not for its surface. Games that are the same problem can differ in
symbols and formats (guessing letters vs digits, a word of 5 vs 4 characters, colour feedback vs counts).
Keep such details out of the logic: make them CONFIG parameters (alphabet, length, feedback format), or let
the model turn the observation into the structure your code expects. When a module cannot read an
observation, it must hand the decision back to the model instead of failing or guessing. A module written
this way can be reused by a new game with a config change or a few lines, not a rewrite. `check_harness.py` applies the static rules
(allowed imports: re, json, math, itertools, collections, random, string, functools, heapq, typing,
dataclasses, harness_lib); `mock_env.py` runs a harness against a fake game to check it runs.

How to read the evidence: tabulate how episodes end (win, out of turns, invalid move, error) before reading
traces; read the last steps of failing traces; a change worth making moves several episodes, not one.
Write a short `proposal.md`: what you saw, what you changed, and why it should help the games below this node.

## How the tree works (read before deciding where your change belongs)

- The root holds what every game uses: a change there must help the games or at least hurt none of them.
- A branch holds what the games below it share; a change there only needs to help those games.
- A leaf may be specific to one game or a small family, including code that decides actions itself.
- `tree.md` shows the current tree: each node's one-line description, the games below it (task ids as in
  `evidence/`) and how they end. The grouping is not fixed: it comes from the data and from your judgement.
- Your change is tested, problem by problem, on every game below the node it is placed at. If some of them
  do not gain, it does not stay there: it is moved down to a new child holding only the games that gained,
  and you may later be asked to separate its general part from its specific part.

Two more files to write:
- `placement.json`: {{"level": "general" or "specific", "target_node": "<this node or one of its ancestors>",
  "why": "<one sentence>"}}. Choose the highest node whose games should all benefit; say "specific" if only
  some of this node's games should. Moving a change up means it is tested on more games.
- `node_description.txt`: one line saying what the games under `target_node` have in common, in terms of the
  problem (for example "guess a hidden item from structured feedback"), never game or benchmark names.
"""

TASK_ESSENCE = """# Describe what each game is, as a problem

`games/` has one folder per game (task1, task2, ...): its instructions as the player first sees them, and a
few traces of a simple agent playing it. Write `essence.json`: {"task1": "<one or two sentences>", ...}.

Describe the underlying problem and what it takes to solve it, not the surface: for example "find a hidden
item by proposing candidates and narrowing them down from structured feedback", or "move through a small
world to reach and act on a described object". Games that are the same problem should read alike even if
their symbols, formats or stories differ. Never write game or benchmark names. Do not edit any other file.
"""

TASK_FACTOR = """# Separate a change into its general and its specific part

An earlier change to node `{node}` helped some games below it ({gained}) and did not help, or hurt, others
({others}). It now lives in a new child node `{child}` that only the games it helped use.

Your job: split it. Put what is useful to every game below `{node}` into `parent_config.json` (and, if needed,
into `shared.py` behind CONFIG keys whose defaults keep current behaviour), and put what is specific to the
helped games into `child_config.json`. A general mechanism with game-specific details as CONFIG parameters
(an alphabet, a length, a feedback format) is the ideal shape. If nothing is general, leave
`parent_config.json` as it is.

Files: `shared.py`, `parent_config.json` (node `{node}`), `child_config.json` (node `{child}`),
`inherited_config.json` (read-only), `before/` (the configurations before the change), `evidence/` (traces;
task ids as in `tree.md`), `tree.md`. The general part is then tested on every game below `{node}`: it is kept
only if no game gets worse. Rules as before: no game or benchmark names or answers; `check_harness.py` and
`mock_env.py` to check that it runs. Write `proposal.md` (what is general, what is specific, why).
"""

TASK_REGROUP = """# Review how games are grouped in the tree

`tree.md` shows the tree: each node's description, the games below it (task ids), how their episodes end, and
the first instructions each game gives. `evidence/` has a few traces per game.

Games that fail for the same underlying reason should share a branch, even if they look different (for
example two games that both guess a hidden item from structured feedback, one over letters and one over
digits). Games grouped together that fail for different reasons may be better apart.

Write `regroup.json`: {{"moves": [{{"task": "task3", "to": "<existing node id>", "why": "<one sentence>"}}],
"descriptions": {{"<node id>": "<one line, in terms of the problem, no game names>"}}}}. An empty "moves" list is
a fine answer. Every move is tested: the regrouped tree is scored on every game and kept only if no game
gets worse. Do not edit any other file.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--remote", default="mailbox:~/evo_b_mailbox")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--rounds", type=int, default=5)
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
