"""Our method at test time, on the graph soft_tree3 learned before deployment (decided with the user, 2026-11-27).

Each test task is a stream of problems in the shared fixed order (~/expb_inputs/order.json); every problem is
played exactly once and scored once. Per task:

  1. probe        the first PROBE problems with the base agent (scored like the rest)
  2. describe     the model describes the game and its required capabilities from the probe traces; embedded
  3. compose      the same retrieval and composing as before deployment -> v0; replayed on the probe traces and
                  repaired until it runs
  4. groups       the remaining problems are split into GROUPS groups. Group 1 is played by v0. Before each later
                  group (versions B, C, D):
                    edit   the model edits the current harness in a workspace like the one before deployment:
                           every earlier version with its code, proposal and traces, and shared/ with the
                           retrieval (what it saw before deployment), but no scores: none exist at test time
                    repair the new harness is replayed on recorded traces and repaired until it runs
                    split  the group's problems alternate between the current version and the new one
                    judge  the model reads both sides' code, descriptions and traces (labels X/Y, randomised) and
                           picks the version that plays better; it becomes the current version
  5. review       (C, D) the task joins the graph with its final harness, and the review distils shared code;
                  only later tasks see the change

  A  compose only (no edit)      B  + edits judged on traces      C  + review after each task
  D  C starting from an empty graph (nothing learned before deployment)

Scores live only in the worker's private results; the method never reads them. `report` (our evaluation code)
does. Every Opus session is timed and its tokens are in private/<tag>_proposer.jsonl.

    python -m mb.evolve.deploy3 --version B --graph ~/evo_b_runs/ours3/state/v0NN --run dep3_B
    python -m mb.evolve.deploy3 --run dep3_B --report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mb.evolve import remote, soft_tree2 as st2, soft_tree3 as S
from mb.evolve.soft_tree import embed

ORDER = Path.home() / "expb_inputs" / "order.json"
DEV_CELLS = ("wordle:basic5,wordle:basic7,hangman:basic,babyai:GoToLocal,babyai:Open,sciworld:test-conductivity,"
             "dc:all").split(",")                     # soft_tree3's cells: task ids stay the same as before deployment
PROBE, GROUPS = 3, 5
TEMPLATE = (S.HERE / "ws_template" / "TASK_template.md").read_text()


def _section(title, until=None):
    a = TEMPLATE.index(title)
    b = TEMPLATE.index(until, a) if until else len(TEMPLATE)
    return TEMPLATE[a:b].rstrip() + "\n"


TEST_EDIT = """# Improve a game-playing agent's harness while it plays a new game

You are improving the harness of an agent that plays a text game. The harness is everything around its model
(`{model}`, a frozen open model with its own thinking mode switched off): the prompts, how observations and history
are shown to the model, the control flow, any extra model calls, and any code that tracks state or helps decide.
In each episode the harness gets the game's instructions, sends actions, and gets one text observation back per
action.

This game is being played for real, one problem after another, and there are no scores: only what the game
itself says in the traces (its messages, and whether it says an episode was won or lost) tells you how an
episode went.

## Your workspace

- `harness/harness.py` is the harness you improve, a copy of version `{parent}`, the one in use. Edit it in place.
- `archive/summary.md` lists every version so far, its parent, and how many problems it played.
- `archive/<id>/` holds one version. `harness/harness.py` is its code and `proposal.md` says what its author
  changed and why. `episodes/<episode>/trace.md` shows every message the model saw, every reply, every action and
  every observation of one episode that version played.
- `harness_lib.py` holds helpers you may import. `mock_env.py` runs `harness/harness.py` against a fake game and a fake
  model, to check that it runs end to end. `check_harness.py` applies the static rules below.

""" + _section("## The interface", "## How to read the evidence") + """
## How to read the evidence

Each episode is one problem, and the model samples at temperature 0.7, so one episode tells you little. Look for
patterns across episodes:

1. For each episode, note how it ended: a win or a loss as the game reports it, running out of turns, an invalid
   move that ends the game, a harness error. Count them.
2. Read the last few steps of the episodes that went wrong, not only the first ones. Find the step where the model
   went wrong and what it saw at that point.
3. Compare versions in the archive: what did earlier changes fix, and what did they break?

## What to do

1. Work through the evidence above. Name the failure pattern you find and count how many episodes show it.
2. Change the harness to fix one pattern that you saw in several episodes. Make the smallest change that could fix
   it: one mechanism, and as few lines as you can. Do not try to play the game yourself.
   Before you finish, check that it passes `python check_harness.py harness/harness.py` and runs with
   `python mock_env.py`.
3. Write `proposal.md` in the workspace root: the failure pattern with the episodes and steps where you saw it, your
   change, and why you expect it to help.

""" + _section("## Rules")

TEST_EDIT_NOTE = """
## In this setting

- Also keep `harness.md` (in the workspace root) up to date with the code: how the harness works, its main
  functions and what each does, its parameters. You may edit it besides `harness/harness.py` and `proposal.md`.
- Keep functions small and named for what they do, with symbols, lengths and formats as named parameters. No game
  or benchmark names anywhere.
- `replay_test.py` replays this game's recorded episodes to a harness: `python replay_test.py harness/harness.py`
  must pass, as well as `check_harness.py` and `mock_env.py`.
- The next problems are split between the version in use and yours; a judge reads both sides' traces and keeps
  the one that plays better.
"""

TEST_JUDGE = """# Which harness plays this game better?

Two versions of a harness for the same text game, `X/` and `Y/`, each just played different problems of the game
(the problems alternate between them, and problems differ in difficulty). There are no scores.

- `X/harness.py`, `X/harness.md`, `X/proposal.md` (if any): version X. The same for `Y/`.
- `episodes_X/<episode>/trace.md`, `episodes_Y/<episode>/trace.md`: every message the model saw, every reply,
  every action and every observation of each episode the version played.

Decide which version plays this game better. Judge by what happened in the episodes: how each ended as the game
reports it (won, lost, out of turns, an invalid move, a crash), and how close the play came. Do not judge by how
much the code changed or how good its description sounds; use the code and descriptions only to understand the
episodes. Write `choice.json` in the workspace root:

    {"better": "X" or "Y", "why": "<the evidence, citing episodes>"}

Do not change any other file.
"""
TEST_JUDGE_TIE = TEST_JUDGE.replace(
    "Write `choice.json`",
    "If they play equally well, pick the one that sends the model less text per episode (shorter prompts, fewer\n"
    "calls): it costs less to run. Write `choice.json`")
GATE = False            # group 1 is split between the base agent and the composed v0, judged like an edit


class Deploy(S.Run):
    def __init__(self, a):
        self.a, self.dir = a, S.RUNS / a.run
        for d in ("state", "private", "rounds", "exports", "workspaces", "tmp"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.box = remote.make(a.remote)
        order = json.loads(ORDER.read_text())
        self.tasks = [t for t in order if not a.tasks or t in a.tasks.split(",")]
        self.problems = {t: [tuple(p) for p in order[t]][:a.limit or None] for t in self.tasks}
        self.cells = DEV_CELLS + [self.cid(t) for t in order]
        self.units = 0

    @staticmethod
    def cid(task):
        return f"test:{task}"

    # ----- state: the graph carried across tasks (C, D) -----
    def start_graph(self):
        vs = self.versions()
        if vs:
            return self.load(vs[-1])
        if self.a.version == "D":
            g = {"nodes": {}, "leaves": {}, "vecs": {}, "descriptions": {}, "order": [], "affinity": {}}
            meta = {"scores": {}, "per_episode": {}, "last_export": {}, "start": {}, "used": {}, "leaf_cand": {}, "units": 0}
        else:
            src = Path(self.a.graph).expanduser()
            g, meta = json.loads((src / "graph.json").read_text()), json.loads((src / "meta.json").read_text())
            (self.dir / "private" / "graph_source.txt").write_text(str(src))
        self.save_state(g, meta)
        return g, meta

    # ----- playing problems (scores stay on the cluster; we keep the visible traces) -----
    def play(self, tag, code, items):
        """Play `items` [(cell, instance)] with this harness, each once; returns the local folder of the
        visible traces (exports/<tag>/episodes/<episode>/trace.md)."""
        dest = self.dir / "exports" / tag / "episodes"
        if self.done(f"{tag}_played"):
            return dest
        hp = self.dir / "tmp" / f"{tag}.py"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(code)
        by_cell = {}
        for cell, inst in items:
            by_cell.setdefault(cell, []).append(inst)
        t0 = time.time()

        def one(cell):
            key = f"{self.a.run}/{tag}/{cell.replace(':', '_')}"
            return key, cell, remote.play(self.box, key, str(hp), cell, "test", by_cell[cell], visible_only=True)

        with ThreadPoolExecutor(len(by_cell)) as ex:
            got = list(ex.map(one, by_cell))
        self.timed("play", tag, t0)
        dest.mkdir(parents=True, exist_ok=True)
        for key, cell, vis in got:
            for tr in sorted(Path(vis).glob("episodes/*/trace.md")):
                (dest / tr.parent.name).mkdir(exist_ok=True)
                shutil.copy(tr, dest / tr.parent.name / "trace.md")
            if (Path(vis) / "rejected.json").exists():           # the harness crashed on its first problem
                (dest.parent / f"rejected_{cell.replace(':', '_')}.json").write_text((Path(vis) / "rejected.json").read_text())
        with (self.dir / "private" / "plays.jsonl").open("a") as f:
            for key, cell, _ in got:
                f.write(json.dumps({"tag": tag, "key": key, "cell": cell, "instances": by_cell[cell]}) + "\n")
        self.record(f"{tag}_played", {"outcome": f"{len(items)} problems", "items": items})
        return dest

    # ----- versions of one task's harness -----
    def vdir(self, c, vid):
        return self.dir / "archive" / c.replace(":", "_") / vid

    def v_add(self, c, vid, parent, code, doc, proposal):
        d = self.vdir(c, vid)
        (d / "harness").mkdir(parents=True, exist_ok=True)
        (d / "harness" / "harness.py").write_text(code)
        (d / "harness.md").write_text(doc or "")
        (d / "proposal.md").write_text(proposal or "")
        (d / "meta.json").write_text(json.dumps({"id": vid, "parent": parent}))

    def v_traces(self, c, vid, episodes):
        d = self.vdir(c, vid) / "episodes"
        d.mkdir(parents=True, exist_ok=True)
        for e in Path(episodes).iterdir():
            shutil.copytree(e, d / e.name, dirs_exist_ok=True)

    def all_traces(self, c):
        out = self.dir / "tmp" / f"traces_{c.replace(':', '_')}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        for e in (self.dir / "archive" / c.replace(":", "_")).glob("*/episodes/*"):
            shutil.copytree(e, out / e.name, dirs_exist_ok=True)
        return out

    def test_edit(self, tag, c, g, meta, cur):
        ws = self.dir / "workspaces" / tag
        new = f"v{len([p for p in (self.dir / 'archive' / c.replace(':', '_')).iterdir() if p.name.startswith('v')])}"
        if self.fresh(tag):
            ws = self.ws(tag)
            (ws / "harness").mkdir()
            (ws / "harness" / "harness.py").write_text((self.vdir(c, cur) / "harness" / "harness.py").read_text())
            (ws / "harness.md").write_text((self.vdir(c, cur) / "harness.md").read_text())
            arc = ws / "archive"
            rows = ["| id | parent | problems played |", "|---|---|---|"]
            for d in sorted((self.dir / "archive" / c.replace(":", "_")).iterdir()):
                m = json.loads((d / "meta.json").read_text())
                shutil.copytree(d, arc / d.name, ignore=shutil.ignore_patterns("meta.json"))
                (arc / d.name / "meta.json").write_text(json.dumps({"id": m["id"], "parent": m["parent"]}))
                n = len(list((d / "episodes").iterdir())) if (d / "episodes").exists() else 0
                rows.append(f"| {m['id']} | {m['parent'] or '-'} | {n} |")
            (arc / "summary.md").write_text("\n".join(rows) + "\n")
            shared = self.share_with_edit(ws, c, g, meta)
            (ws / "TASK.md").write_text(TEST_EDIT.format(model=S.MODEL, parent=cur, chat_cap=S.CHAT_CAP)
                                        + TEST_EDIT_NOTE + (S.SHARED_NOTE if shared else "")
                                        + (S.EDIT_STRUCTURE_NOTE if S.STRUCT_HINT else ""))
        self.propose(ws, tag)
        hp = ws / "harness" / "harness.py"
        code = hp.read_text() if hp.exists() else ""
        doc = (ws / "harness.md").read_text().strip() if (ws / "harness.md").exists() else ""
        prop = (ws / "proposal.md").read_text() if (ws / "proposal.md").exists() else ""
        if code == (self.vdir(c, cur) / "harness" / "harness.py").read_text():
            return None, "no change"
        code, doc, fixes = self.until_it_runs(tag, c, self.all_traces(c), code, doc)
        if code is None:
            return None, f"still crashing after {fixes} repairs"
        self.v_add(c, new, cur, code, doc, prop)
        return new, f"{fixes} repairs"

    def judge(self, tag, c, cur, new, ep_cur, ep_new):
        ws = self.dir / "workspaces" / tag
        flip = random.Random(tag).random() < 0.5            # which side is X: no position bias toward the new one
        side = {"X": new if flip else cur, "Y": cur if flip else new}
        eps = {cur: ep_cur, new: ep_new}
        if self.fresh(tag):
            ws = self.ws(tag)
            for lab, vid in side.items():
                (ws / lab).mkdir()
                for f in ("harness.md", "proposal.md"):
                    shutil.copy(self.vdir(c, vid) / f, ws / lab / f)
                shutil.copy(self.vdir(c, vid) / "harness" / "harness.py", ws / lab / "harness.py")
                shutil.copytree(eps[vid], ws / f"episodes_{lab}")
            (ws / "TASK.md").write_text(TEST_JUDGE_TIE if S.TIE_COST else TEST_JUDGE)
        self.propose(ws, tag)
        try:
            ch = json.loads((ws / "choice.json").read_text())
            pick = side[str(ch.get("better", "")).strip().upper()]
            why = str(ch.get("why", ""))[:500]
        except Exception as e:
            pick, why = cur, f"no valid choice.json ({e!r}); the version in use stays"
        return pick, why, side

    def history_md(self, x, meta):
        """At test time: before deployment the graph's own games keep their summary; a test game's versions come
        with the judge's choices instead of scores (none exist)."""
        if not x.startswith("test:"):
            return super().history_md(x, meta)
        k = self.tasks.index(x[5:]) + 1 if x[5:] in self.tasks else None
        d = self.dir / "archive" / x.replace(":", "_")
        rows, notes = ["| version | parent | problems played |", "|---|---|---|"], []
        for v in sorted(d.iterdir()) if d.exists() else []:
            m = json.loads((v / "meta.json").read_text())
            n = len(list((v / "episodes").iterdir())) if (v / "episodes").exists() else 0
            rows.append(f"| {m['id']} | {m.get('parent') or '-'} | {n} |")
            prop = (v / "proposal.md").read_text().strip() if (v / "proposal.md").exists() else ""
            if prop:
                notes.append(f"### {m['id']}\n\n{prop[:2500]}")
        judged = []
        for j in range(2, GROUPS + 1):
            r = self.done(f"T{k}_g{j}_done") if k else None
            if r and r.get("why"):
                judged.append(f"- group {j}: {r['outcome']}. Why: {r['why']}")
        return ("# Versions\n\n" + "\n".join(rows) + "\n\n# The judge's choices (no scores at test time)\n\n"
                + ("\n".join(judged) or "(none)") + "\n\n# What each version changed and why\n\n" + "\n\n".join(notes) + "\n")

    # ----- one task -----
    def run_task(self, k, task, g, meta):
        c, items = self.cid(task), self.problems[task]
        tp = f"T{k}"
        probe, rest = items[:PROBE], items[PROBE:]
        size = len(rest) / GROUPS
        groups = [rest[round(i * size):round((i + 1) * size)] for i in range(GROUPS)]
        # 1-2 probe and description
        self.play(f"{tp}_probe", S.BASE_AGENT.read_text(), probe)
        if c not in g["vecs"]:
            desc, caps = self.describe(c, f"{tp}_probe")
            vecs = embed([desc] + caps)
            g["vecs"][c], g["descriptions"][c] = vecs[0], desc
            if S.PROFILE:
                g.setdefault("profiles", {})[c] = self.profile(f"{tp}_probe")
            g.setdefault("capabilities", {})[c] = [{"capability": t, "vec": v} for t, v in zip(caps, vecs[1:])]
        # 3 compose
        if not self.vdir(c, "v0").exists():
            tasks, workflows, modules = self.retrieve(g, c)
            code, used, doc = S.BASE_AGENT.read_text(), {}, S.BASE_AGENT_DOC
            if tasks or workflows or modules:
                code, used, doc = self.compose(f"{tp}_compose", c, g, meta, tasks, workflows, modules)
            how = ("composed from earlier games' code: " + json.dumps(sorted(used))) if (tasks or workflows or modules) \
                else "the base agent"
            # the probe's episodes were played by the base agent: they are in the archive as version "base"
            self.v_add(c, "base", None, S.BASE_AGENT.read_text(), S.BASE_AGENT_DOC, "The base agent (played the probe).\n")
            self.v_traces(c, "base", self.dir / "exports" / f"{tp}_probe" / "episodes")
            self.v_add(c, "v0", None, code, doc, S.START_NOTE.format(how=how))
            self.record(f"{tp}_compose_used", {"outcome": f"used {sorted(used)}", "tasks": list(tasks)})
        cur = (self.done(f"{tp}_current") or {}).get("current", "v0")
        # 4 groups
        for j, grp in enumerate(groups, 1):
            gt = f"{tp}_g{j}"
            if self.done(f"{gt}_done"):
                cur = self.done(f"{gt}_done")["current"]
                continue
            new = None
            if j > 1 and self.a.version != "A":
                new, how = self.test_edit(f"{gt}_edit", c, g, meta, cur)
                self.record(f"{gt}_edit", {"outcome": f"{new or 'no new version'} ({how})", "parent": cur})
            elif j == 1 and GATE and cur == "v0" and \
                    (self.vdir(c, "v0") / "harness" / "harness.py").read_text() != S.BASE_AGENT.read_text():
                new = "base"            # adaptive transfer (user, 2026-11-27): keep v0 only if it beats the base agent
            code_cur = (self.vdir(c, cur) / "harness" / "harness.py").read_text()
            if new is None:
                ep = self.play(f"{gt}_{cur}", code_cur, grp)
                self.v_traces(c, cur, ep)
                rec = {"outcome": f"played by {cur}", "current": cur}
            else:
                a_items, b_items = grp[0::2], grp[1::2]
                code_new = (self.vdir(c, new) / "harness" / "harness.py").read_text()
                with ThreadPoolExecutor(2) as ex:
                    f1 = ex.submit(self.play, f"{gt}_{cur}", code_cur, a_items)
                    f2 = ex.submit(self.play, f"{gt}_{new}", code_new, b_items)
                    ep_cur, ep_new = f1.result(), f2.result()
                self.v_traces(c, cur, ep_cur)
                self.v_traces(c, new, ep_new)
                pick, why, side = self.judge(f"{gt}_judge", c, cur, new, ep_cur, ep_new)
                rec = {"outcome": f"{cur} vs {new}: {pick} kept", "current": pick, "why": why, "labels": side}
                cur = pick
            self.record(f"{gt}_done", rec)
            self.record(f"{tp}_current", {"outcome": cur, "current": cur})
        # 5 the task joins the graph; review (C, D)
        if self.a.version in ("C", "D") and c not in g["order"]:
            g["order"].append(c)
            g["leaves"][c] = {"code": (self.vdir(c, cur) / "harness" / "harness.py").read_text(),
                              "doc": (self.vdir(c, cur) / "harness.md").read_text()}
            meta["scores"][c] = None                        # no score at test time
            self.save_state(g, meta)
            if len(g["order"]) >= 2 and S.ABLATION not in ("flat", "retrieve"):
                g, meta = self.review(f"{tp}_review", c, g, meta)
        self.record(f"{tp}_done", {"outcome": f"final {cur}"})
        return g, meta

    def loop(self):
        g, meta = self.start_graph()
        if self.a.version in ("C", "D"):                   # sequential: each task may use what earlier ones added
            for k, t in enumerate(self.tasks, 1):
                if not self.done(f"T{k}_done"):
                    g, meta = self.run_task(k, t, g, meta)
        else:                                                # A, B: tasks are independent; run them together
            with ThreadPoolExecutor(len(self.tasks)) as ex:
                for f in [ex.submit(self.run_task, k, t, json.loads(json.dumps(g)), json.loads(json.dumps(meta)))
                          for k, t in enumerate(self.tasks, 1) if not self.done(f"T{k}_done")]:
                    f.result()
        print("all test tasks done", flush=True)

    # ----- our evaluation (reads the private scores; the method above never does) -----
    def report(self):
        box = self.box
        box.pull()
        per = {}
        for l in (self.dir / "private" / "plays.jsonl").read_text().splitlines():
            p = json.loads(l)
            priv = box.repo / "play_results" / p["key"] / "private.json"
            res = json.loads(priv.read_text()) if priv.exists() else {}
            eps = res.get("per_episode") or {}
            bench, sub = p["cell"].split(":", 1)
            k = int(p["tag"][1:].split("_")[0])
            task = self.tasks[k - 1]
            for i in p["instances"]:
                hid = hashlib.md5(f"{bench}/{sub}/{i}".encode()).hexdigest()[:10]
                sc = eps.get(hid, {}).get("score")
                if "rejected" in res:
                    sc = 0.0                                 # the harness crashed: these problems score 0
                per.setdefault(task, {})[f"{p['cell']}/{i}"] = {"score": sc, "tag": p["tag"]}
        out = {}
        for t, d in per.items():
            sc = [v["score"] for v in d.values() if isinstance(v["score"], (int, float))]
            out[t] = {"problems": len(d), "scored": len(sc), "mean": round(sum(sc) / len(sc), 4) if sc else None}
        out["_cost"] = self.cost()
        (self.dir / "results.json").write_text(json.dumps({"tasks": out, "per_problem": per}, indent=1))
        print(json.dumps(out, indent=1))

    def cost(self):
        """Opus tokens and list-price dollars per test task, by session kind."""
        tot = {}
        for f in (self.dir / "private").glob("T*_proposer.jsonl"):
            tag = f.name[:-len("_proposer.jsonl")]
            k = tag.split("_")[0]
            kind = ("describe" if "description" in tag else "compose" if "compose" in tag else
                    "judge" if "judge" in tag else "review" if "review" in tag else
                    "repair" if "_fix" in tag else "edit" if "edit" in tag else "other")
            for line in f.read_text().splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "result":
                    u = d.get("usage", {})
                    row = tot.setdefault(k, {}).setdefault(kind, {"sessions": 0, "in": 0, "out": 0, "usd": 0.0})
                    row["sessions"] += 1
                    row["in"] += u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                    row["out"] += u.get("output_tokens", 0)
                    row["usd"] = round(row["usd"] + d.get("total_cost_usd", 0), 4)
        return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--version", choices=list("ABCD"), default="B")
    ap.add_argument("--graph", help="the state folder (graph.json, meta.json) chosen before deployment")
    ap.add_argument("--ablation", default="", choices=["", "hard", "flat", "retrieve"])
    ap.add_argument("--remote", default=f"mailbox:{Path.home() / 'evo_b_mailbox'}")
    ap.add_argument("--tasks", default="", help="comma-separated test tasks (default: all, in the shared order)")
    ap.add_argument("--limit", type=int, default=0, help="a smoke test: only the first problems of each task")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--compose-free", action="store_true", help="as soft_tree3 --compose-free")
    ap.add_argument("--profile", action="store_true", help="as soft_tree3 --profile (use with a graph built with it)")
    ap.add_argument("--tie-cost", action="store_true", help="the judge picks the cheaper version when both play equally well")
    ap.add_argument("--gate", action="store_true",
                    help="group 1 is split between the base agent and the composed v0 and judged: the composed "
                         "harness is used only if it plays better (adaptive transfer, user 2026-11-27)")
    ap.add_argument("--struct-hint", action="store_true",
                    help="test-time edits get the same checklist of harness parts as the dev edits of a "
                         "--struct-hint run (what was visible in dev is visible at test time)")
    a = ap.parse_args()
    a.n = 20                                       # soft_tree3 helpers read it (noise estimates); unused here
    S.ABLATION = st2.ABLATION = a.ablation
    S.STRUCT_HINT, S.COMPOSE_FREE, S.TIE_COST, S.PROFILE = a.struct_hint, a.compose_free, a.tie_cost, a.profile
    global GATE
    GATE = a.gate
    d = Deploy(a)
    d.report() if a.report else d.loop()


if __name__ == "__main__":
    main()
