"""Our method at test time (2026-11-26, user): four versions on the shared test stream.

  A  frozen       the tree learned on the search split; a new task probes, inherits softly, plays. No edit.
  B  adapt        A + the code-editing model edits the new task's own leaf; versions compete on the next
                  problems by Thompson sampling, the model judging wins from the visible traces
  C  continual    B + after each task the model extracts what it shares with earlier leaves into internal
                  nodes (no search split, no regression check: earlier test tasks are finished; only later
                  tasks see the change)
  D  from zero    C starting from an empty tree (no pre-deployment), like Harness²'s empty library

Every test problem is played exactly once, in the shared fixed order; all are scored for the report from the
worker's private results, which no decision reads. Opus sessions are tagged "adapt" (essence, judging, leaf
edits: they serve the current task) or "maintain" (extraction: it serves later tasks).

Per task (problems p1..pN):
  1. probe   p1-p3 with the root configuration
  2. essence the model describes the problem from the probe traces; embedded; soft parents; leaf v0
  3. play    p4..p(3+W) with v0                                                    (A: v0 for the rest)
  4. edit    the model edits the leaf from all visible traces so far -> v1          (B, C, D; up to EDITS times)
  5. compete next J problems: Thompson sampling over the leaf versions in batches of BATCH; the model judges
             each batch's episodes (win / loss) from the visible traces; Beta posteriors update
  6. rest    the version with the highest posterior mean
  7. extract (C, D) after the task: as before deployment, into an internal node; no check

    python -m mb.evolve.deploy2 --version B --tree ~/evo_b_runs/ours2/state/v0NN --run dep_B --order order.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mb.evolve import remote
from mb.evolve.check import problems as static_problems
from mb.evolve.soft_tree import HERE, SANDBOX, assemble, dist, embed
from mb.evolve import soft_tree2 as S

RUNS = Path.home() / "evo_b_runs"
W, J, BATCH, EDITS = 5, 8, 4, 2


class Deploy:
    def __init__(self, a):
        self.a = a
        self.dir = RUNS / a.run
        for d in ("state", "private", "steps", "workspaces", "tmp"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.box = remote.make(a.remote)
        if (self.dir / "state" / "tree.json").exists():
            self.shared = (self.dir / "state" / "shared.py").read_text()
            self.tree = json.loads((self.dir / "state" / "tree.json").read_text())
        elif a.version == "D":
            self.shared = (HERE / "seeds" / "soft_tree_shared.py").read_text()
            self.tree = {"nodes": {"root": {"parents": {}, "config": {}, "desc": "every game", "centre": []}},
                         "vecs": {}, "order": [], "essence": {}}
        else:
            src = Path(a.tree).expanduser()
            self.shared = (src / "shared.py").read_text()
            self.tree = json.loads((src / "tree.json").read_text())
        self.save_state()
        self.ledger = self.dir / "private" / "ledger.jsonl"

    def save_state(self):
        (self.dir / "state" / "shared.py").write_text(self.shared)
        (self.dir / "state" / "tree.json").write_text(json.dumps(self.tree, indent=1))

    def log(self, **kw):
        with self.ledger.open("a") as f:
            f.write(json.dumps({"time": time.time(), **kw}) + "\n")

    # ---- playing (visible traces only for decisions) ----
    def harness_file(self, cfg, cell, tag):
        src = assemble(self.shared, cfg)
        bad = static_problems(src)
        if bad:
            raise RuntimeError(f"{tag}: harness fails the static check: {bad}")
        p = self.dir / "tmp" / tag / f"{cell.replace(':', '_')}.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
        return p

    def play(self, items, tag):
        """items: [(index, cell, instance, cfg, arm)]; played at once; returns {index: visible trace text}."""
        def one(it):
            i, cell, inst, cfg, arm = it
            key = f"{self.a.run}/{tag}/p{i:03d}"
            t0 = time.time()
            vis = remote.play(self.box, key, self.harness_file(cfg, cell, f"{tag}_p{i:03d}"), cell, "test", [inst],
                              visible_only=True)
            traces = sorted(vis.glob("episodes/*/trace.md"))
            self.log(kind="episode", step=tag, index=i, cell=cell, instance=inst, arm=arm, key=key,
                     wall=round(time.time() - t0, 1))
            return i, traces[0].read_text() if traces else "(no trace: the harness could not run)"
        with ThreadPoolExecutor(min(8, len(items)) or 1) as ex:
            return dict(ex.map(one, items))

    # ---- code-model sessions ----
    def session(self, ws, tag, purpose):
        done = self.dir / "private" / f"{tag}_done.json"
        if done.exists() and ws.exists():
            return json.loads(done.read_text())["rc"]
        sys.path.insert(0, str(SANDBOX))
        import run_proposer
        log = self.dir / "private" / f"{tag}_proposer.jsonl"
        t0 = time.time()
        rc = run_proposer.run(ws, log)
        usage = {}
        for line in log.read_text(errors="ignore").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "result":
                usage = {**(e.get("usage") or {}), "cost_usd_list": e.get("total_cost_usd")}
        self.log(kind="opus", purpose=purpose, step=tag, seconds=round(time.time() - t0, 1), usage=usage)
        done.write_text(json.dumps({"rc": rc}, default=str))
        return rc

    def ws(self, tag):
        w = self.dir / "workspaces" / tag
        shutil.rmtree(w, ignore_errors=True)
        w.mkdir(parents=True)
        return w

    def essence(self, task, traces, tag):
        w = self.ws(tag)
        for i, t in traces.items():
            (w / "evidence" / "task" / f"p{i:03d}").mkdir(parents=True)
            (w / "evidence" / "task" / f"p{i:03d}" / "trace.md").write_text(t)
        (w / "TASK.md").write_text(S.TASK_ESSENCE.format(tid="task"))
        self.session(w, tag, "adapt")
        f = w / "essence.txt"
        return f.read_text().strip()[:600] if f.exists() and f.read_text().strip() else task

    def judge(self, traces, tag):
        """The model reads visible traces and says which episodes the agent won."""
        w = self.ws(tag)
        for i, t in traces.items():
            (w / "episodes").mkdir(exist_ok=True)
            (w / "episodes" / f"p{i:03d}.md").write_text(t)
        (w / "TASK.md").write_text(TASK_JUDGE)
        self.session(w, tag, "adapt")
        try:
            got = json.loads((w / "judgements.json").read_text())
        except Exception:
            got = {}
        return {i: 1 if str(got.get(f"p{i:03d}", "")).lower().startswith("win") else 0 for i in traces}

    def edit(self, lf, inherited, traces, tag):
        w = self.ws(tag)
        for f in ("harness_lib.py", "mock_env.py"):
            shutil.copy(HERE / "ws_template" / f, w / f)
        shutil.copy(HERE / "check.py", w / "check_harness.py")
        (w / "shared.py").write_text(self.shared)
        (w / "inherited_config.json").write_text(json.dumps(inherited, indent=1))
        (w / "leaf_config.json").write_text(json.dumps(self.tree["nodes"][lf]["config"], indent=1))
        for i, t in traces.items():
            d = w / "evidence" / "task" / "episodes" / f"p{i:03d}"
            d.mkdir(parents=True)
            (d / "trace.md").write_text(t)
        (w / "TASK.md").write_text(S.TASK_LEAF.format(tid="task", n=len(traces)).replace(
            "The change is kept only if this game gains on the same problems, beyond noise.",
            "This is a new game met at test time: every problem is played once. Your version will compete with "
            "the current one on the next problems; the one that wins more is used for the rest."))
        self.session(w, tag, "adapt")
        try:
            cfg = json.loads((w / "leaf_config.json").read_text())
        except Exception:
            cfg = None
        new_shared = (w / "shared.py").read_text()
        if cfg is None or static_problems(assemble(new_shared, {**inherited, **cfg})):
            return None, None
        return cfg, new_shared

    # ---- one task ----
    def run_task(self, name, probs):
        st_f = self.dir / "steps" / f"{name}.json"
        if st_f.exists():
            return json.loads(st_f.read_text())
        v = self.a.version
        root_cfg = S.resolve(self.tree, "root")
        seen = {}
        seen.update(self.play([(i, c, inst, root_cfg, "root") for i, (c, inst) in enumerate(probs[:3])], f"{name}_probe"))
        ess = self.essence(name, seen, f"{name}_essence")
        vec = embed([ess])[0]
        cell = f"test:{name}"
        self.tree["vecs"][cell], self.tree["essence"][cell] = vec, ess
        if not self.tree["nodes"]["root"].get("centre"):
            self.tree["nodes"]["root"]["centre"] = vec
        lf = S.leaf(cell)
        self.tree["nodes"][lf] = {"parents": S.parent_weights(self.tree, vec), "config": {}, "desc": ess[:200], "centre": vec}
        inherited = S.mix({p: S.resolve(self.tree, p) for p in self.tree["nodes"][lf]["parents"]},
                          self.tree["nodes"][lf]["parents"])
        arms = {"v0": {"cfg": {**inherited}, "own": {}, "shared": self.shared, "a": 1, "b": 1}}
        i = 3
        block = probs[i:i + (W if v != "A" else len(probs))]
        seen.update(self.play([(i + k, c, inst, arms["v0"]["cfg"], "v0") for k, (c, inst) in enumerate(block)],
                              f"{name}_v0"))
        i += len(block)
        rng = random.Random(f"{self.a.run}/{name}")
        for e in range(1, EDITS + 1):
            if v == "A" or i >= len(probs):
                break
            own, new_shared = self.edit(lf, inherited, seen, f"{name}_edit{e}")
            if own is not None:
                if new_shared != self.shared:       # a new module behind a switch: the code base grows
                    self.shared = new_shared
                arms[f"v{e}"] = {"cfg": {**inherited, **own}, "own": own, "shared": self.shared, "a": 1, "b": 1}
            end = min(len(probs), i + J)
            while i < end:                          # compete in batches; the model judges each batch
                chosen = []
                for k in range(min(BATCH, end - i)):
                    draw = {n: rng.betavariate(x["a"], x["b"]) for n, x in arms.items()}
                    chosen.append(max(draw, key=draw.get))
                items = [(i + k, probs[i + k][0], probs[i + k][1], arms[n]["cfg"], n) for k, n in enumerate(chosen)]
                got = self.play(items, f"{name}_e{e}_b{i:03d}")
                seen.update(got)
                wins = self.judge(got, f"{name}_judge{i:03d}")
                for (idx, _, _, _, n) in items:
                    arms[n]["a"] += wins[idx]
                    arms[n]["b"] += 1 - wins[idx]
                i += len(items)
        best = max(arms, key=lambda n: arms[n]["a"] / (arms[n]["a"] + arms[n]["b"]))
        if i < len(probs):
            seen.update(self.play([(i + k, c, inst, arms[best]["cfg"], best) for k, (c, inst) in enumerate(probs[i:])],
                                  f"{name}_rest"))
        self.tree["nodes"][lf]["config"] = arms[best]["own"]
        self.tree["order"].append(cell)
        if v in ("C", "D"):
            self.extract(name, cell, seen)
        self.save_state()
        res = {"task": name, "essence": ess, "parents": self.tree["nodes"][lf]["parents"], "best": best,
               "arms": {n: {"a": x["a"], "b": x["b"], "own": x["own"]} for n, x in arms.items()}, "n": len(probs)}
        st_f.write_text(json.dumps(res, indent=1))
        print(f"{name}: parents {res['parents']}, best {best}, arms {[(n, x['a'], x['b']) for n, x in arms.items()]}",
              flush=True)
        return res

    def extract(self, name, cell, seen):
        """C, D: what this task's leaf shares with earlier leaves goes into an internal node; no check."""
        others = sorted([x for x in self.tree["order"] if x != cell and S.leaf(x) in self.tree["nodes"]],
                        key=lambda x: dist(self.tree["vecs"][x], self.tree["vecs"][cell]))[:3]
        if not others:
            return
        w = self.ws(f"{name}_extract")
        tid = {cell: "task1", **{x: f"task{k + 2}" for k, x in enumerate(others)}}
        leaves = {tid[x]: {"own": self.tree["nodes"][S.leaf(x)]["config"], "resolved": S.resolve(self.tree, S.leaf(x)),
                           "essence": self.tree["essence"].get(x, "")} for x in [cell] + others}
        (w / "leaves.json").write_text(json.dumps(leaves, indent=1))
        (w / "internal_nodes.json").write_text(json.dumps(
            {n: {"config": self.tree["nodes"][n]["config"], "desc": self.tree["nodes"][n].get("desc", "")}
             for n in S.internal(self.tree)}, indent=1))
        (w / "tree.md").write_text("(see leaves.json and internal_nodes.json)\n")
        for i, t in list(seen.items())[:4]:
            (w / "evidence" / "task1").mkdir(parents=True, exist_ok=True)
            (w / "evidence" / "task1" / f"p{i:03d}.md").write_text(t)
        (w / "TASK.md").write_text(S.TASK_EXTRACT.format(tid="task1").replace(
            "is re-scored: the extraction is kept only if none gets worse.",
            "applies to later games only (this is test time; there is no check)."))
        self.session(w, f"{name}_extract", "maintain")
        try:
            prop = json.loads((w / "extract.json").read_text())
        except Exception:
            return
        ids = {v: k for k, v in tid.items()}
        who = [ids[t] for t in prop.get("tasks", []) if t in ids]
        cfg = prop.get("config")
        if not prop.get("node") or not isinstance(cfg, dict) or len(who) < 2:
            return
        same = [n for n in S.internal(self.tree) if all(self.tree["nodes"][n]["config"].get(k) == v for k, v in cfg.items())]
        node = prop["node"] if prop["node"] in S.internal(self.tree) else (same[0] if same else f"n{len(S.internal(self.tree)) + 1}")
        self.tree["nodes"].setdefault(node, {"parents": {"root": 1.0}, "config": {}, "desc": "", "centre": []})
        self.tree["nodes"][node]["config"].update(cfg)
        self.tree["nodes"][node]["desc"] = str(prop.get("description", ""))[:300]
        for x in who:
            for k in cfg:
                self.tree["nodes"][S.leaf(x)]["config"].pop(k, None)
        self.tree["nodes"][node]["centre"] = [sum(self.tree["vecs"][x][i] for x in who) / len(who)
                                              for i in range(len(self.tree["vecs"][cell]))]
        for x in self.tree["order"]:
            if S.leaf(x) in self.tree["nodes"]:
                self.tree["nodes"][S.leaf(x)]["parents"] = S.parent_weights(self.tree, self.tree["vecs"][x])
        S.recentre(self.tree)

    def run(self, order):
        tasks = [t for t in order if not self.a.tasks or t in self.a.tasks.split(",")]
        for t in tasks:
            probs = [tuple(p) for p in order[t]][: self.a.max_problems or None]
            self.run_task(t, probs)

    # ---- report (our side only; reads the private scores) ----
    def report(self):
        rows = [json.loads(l) for l in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        sc, cost = {}, {"adapt": {}, "maintain": {}}
        for r in rows:
            if r["kind"] == "episode":
                p = self.box.repo / "play_results" / r["key"] / "private.json"
                if p.exists():
                    s = json.loads(p.read_text()).get("mean_score")
                    sc.setdefault(r["step"].split("_")[0], []).append((r["index"], s))
            elif r["kind"] == "opus":
                c = cost[r["purpose"]]
                c["sessions"] = c.get("sessions", 0) + 1
                for k, v in (r.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        c[k] = c.get(k, 0) + v
        out = {"scores": {t: {"n": len(v), "mean": sum(s for _, s in v if s is not None) / max(1, len(v)),
                              "curve": [s for _, s in sorted(v)]} for t, v in sc.items()}, "opus": cost}
        (self.dir / "report.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({t: (x["n"], round(x["mean"], 3)) for t, x in out["scores"].items()}), json.dumps(cost))


TASK_JUDGE = """# Did the agent win?

`episodes/` has the traces of a few episodes of a game, as the agent saw them (instructions, its replies, the
environment's answers). For each episode, decide from what the environment said whether the agent achieved
the goal. Write `judgements.json`: {"p004": "win", "p005": "loss", ...} (one entry per file). Do not edit any
other file.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--tree", default="", help="state folder of the pre-deployment run (A, B, C)")
    ap.add_argument("--order", default=str(Path.home() / "expb_inputs" / "order.json"))
    ap.add_argument("--remote", default="mailbox:~/evo_b_mailbox")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--max-problems", type=int, default=0, help="pilot: only the first k problems of each task")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    d = Deploy(a)
    if a.report:
        d.report()
        return
    d.run(json.loads(Path(a.order).expanduser().read_text()))
    d.report()


if __name__ == "__main__":
    main()
