"""Harness²'s method in our setting: ports of the official core/recursion.py, core/selection.py and
core/schedule.py (run_task, run_domain), driven over our task streams.

One Harness² "task" is one step: k reference rollouts under the common H0, then the recursion mode
(parallel: k INV + k DOM proposals from the shared evidence, each rolled once, then k independent
compositions, each rolled once; sequential: k rounds of INV/DOM/compose), then blind selection among
the k finals (no selector session at k=1), the experience reflector, and the library update. 4k
rollouts and 3k (+1 selector if k>1) (+1 reflector) sessions per step, as in the official ledger.

Where the rollouts of a step are played (the only structural change, see `assignment`):
  * development stream: every rollout of a step plays the same problem (the paper's repeated
    execution on the current task);
  * test stream: one episode per problem. A step's rollouts are played on consecutive problems of
    the fixed order (k=1: H0 on p, INV on p+1, DOM on p+2, the composed harness on p+3), each counted
    as that problem's one episode. A step that the stream ends inside plays what it can.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import evidence as E
from . import guard, official
from . import surface as S
from .seams import SessionError

STAGE_ATTEMPTS = 2                 # official recursion.STAGE_ATTEMPTS
PHASE_WIDTH = 2                    # official config.PHASE_WIDTH
_CHOICE = re.compile(r"```choice\s*\n?\s*([A-Za-z0-9_-]+)\s*\n?\s*```", re.S)
_WHY = re.compile(r"<why>(.*?)</why>", re.S)
_EXP = re.compile(r"<experience>(.*?)</experience>", re.S)
_EDITS = re.compile(r"<edits>(.*?)</edits>", re.S)
_NOTES = re.compile(r"<notes>(.*?)</notes>", re.S)
TEST_NOTE = ("\n\n---\n(Each rollout in this step played a different instance of this task, one episode "
             "each; instances differ in their hidden facts. The instructions above are those of the first.)\n")


@dataclass
class Stage:
    label: str
    dim: str
    tree: S.Tree
    kept_base: bool = False
    fell_back: bool = False
    error: str = ""
    attempts: int = 0
    notes: str = ""
    execution_failed: bool = False
    skipped: bool = False


def labels_in_order(mode: str, k: int) -> list[str]:
    """Rollout labels in the order the official schedule plays them."""
    bases = [f"base{i}" for i in range(1, k + 1)]
    if mode == "agg":
        return bases + [f"{d}{i}" for i in range(1, k + 1) for d in ("inv", "dom")] + [f"agg{i}" for i in range(1, k + 1)]
    return bases + [x for t in range(1, k + 1) for x in (f"inv{t}", f"dom{t}", f"c{t}")]


def assignment(mode, k, problems, same_problem: bool) -> dict:
    """label -> (cell, split, instance). Development: all on problems[0]; test: one label per problem."""
    labels = labels_in_order(mode, k)
    if same_problem:
        return {lbl: problems[0] for lbl in labels}
    return {lbl: p for lbl, p in zip(labels, problems)}


class Harness2:
    def __init__(self, run_dir: Path, session, play, answers: guard.Answers | None, *, k=1, mode="agg",
                 width=PHASE_WIDTH, sequential_play=False):
        self.run = Path(run_dir)
        self.session, self.play, self.answers = session, play, answers
        self.k, self.mode, self.width = k, mode, width
        self.sequential_play = sequential_play          # test streams: one problem at a time, in order
        self.bp = official.PROMPTS

    # ---------------------------------------------------------------- rollouts
    def roll(self, step_dir: Path, step_id: str, label: str, tree: S.Tree, problem) -> E.Rollout:
        cell, split, inst = problem
        harness = S.materialise(tree, step_dir / "harnesses" / label)
        [ep] = self.play(harness, cell, split, [inst], key=f"{step_id}/{label}")
        if self.answers is not None:
            self.answers.mark_seen(cell, split, inst)
        return E.Rollout(label=label, key=f"{step_id}/{label}", episode=ep)

    def roll_many(self, step_dir, step_id, items, assign) -> dict:
        items = [(lbl, t) for lbl, t in items if lbl in assign]
        if self.sequential_play or len(items) < 2:
            return {lbl: self.roll(step_dir, step_id, lbl, t, assign[lbl]) for lbl, t in items}
        with ThreadPoolExecutor(min(self.width, len(items))) as ex:
            got = list(ex.map(lambda it: self.roll(step_dir, step_id, it[0], it[1], assign[it[0]]), items))
        return {lbl: r for (lbl, _), r in zip(items, got)}

    # ---------------------------------------------------------------- one improvement stage
    def accept(self, new: S.Tree, fallback: S.Tree, label: str) -> tuple[bool, str]:
        """official recursion._accept_tree: repair triggers, then validate; plus the shared-store rule."""
        S.autotrigger(new)
        changed = S.diff_summary(fallback, new)
        kept = not (changed["added"] or changed["removed"] or changed["modified"])
        errs = S.validate(new)
        try:
            official.official_context.assert_tree_has_no_grader_verdicts(new)
        except ValueError as exc:
            errs.append(str(exc))
        errs += guard.violations(new, self.answers)
        return kept, " | ".join(errs)

    def stage(self, *, step_dir, step_id, label, dim, job, fallback, task_md, rollouts, harness, library_dir,
              pool=(), decisions=None) -> Stage:
        role = "editor" if dim in ("INV", "DOM") else "composer"
        try:
            ctx = E.build(step_dir / "context" / label, task_md, rollouts, harness=harness,
                          library_dir=library_dir, pool=pool, decisions=decisions)
            mission = official.mission(ctx, job=job)
        except Exception as exc:
            return Stage(label, dim, dict(fallback), fell_back=True, error=f"could not build evidence: {exc}"[:400])
        err = ""
        for attempt in range(1, STAGE_ATTEMPTS + 1):
            extra = (f"\n\n# YOUR PREVIOUS ATTEMPT WAS REJECTED\n\n{err}\n\nFix it in `context/harness/` and "
                     f"reply again." if err else "")
            try:
                reply = self.session(role, f"{step_id}/{label}.a{attempt}", ctx.root, mission + extra,
                                     harvest="context/harness")
            except SessionError as exc:
                return Stage(label, dim, dict(fallback), fell_back=True, error=str(exc)[:400], attempts=attempt,
                             execution_failed=True)
            if reply.workspace is None:
                err = ("the harness directory did not come back from the session; edit the files in "
                       "`context/harness/` in place")
                continue
            rejected: list[str] = []
            new = S.read_tree(reply.workspace, rejected)
            if not new:
                err = "`context/harness/` came back empty -- edit the files, do not delete them all"
                continue
            if rejected:
                err = ("these files are off the editable surface and were not applied: " +
                       ", ".join(sorted(rejected)[:6]) + ". Component names must be lowercase letters, digits, "
                       "`_` or `-` (e.g. `skills/narrow_down/SKILL.md`, `scripts/parse_feedback.py`).")
                S.write_tree(new, ctx.root / "harness")
                continue
            kept, err = self.accept(new, fallback, label)
            if err:
                S.write_tree(new, ctx.root / "harness")
                continue
            m = _NOTES.search(reply.text)
            return Stage(label, dim, new, kept_base=kept, attempts=attempt,
                         notes=m.group(1).strip() if m else reply.text.strip())
        return Stage(label, dim, dict(fallback), fell_back=True, error=err[:400], attempts=STAGE_ATTEMPTS)

    def stages(self, specs, **kw) -> list[Stage]:
        if len(specs) < 2:
            return [self.stage(**s, **kw) for s in specs]
        with ThreadPoolExecutor(min(self.width, len(specs))) as ex:
            return list(ex.map(lambda s: self.stage(**s, **kw), specs))

    # ---------------------------------------------------------------- recursion modes
    def run_parallel(self, step_dir, step_id, base, base_tree, lib_dir, task_md, assign):
        k, rollouts, trees, stages = self.k, {r.label: r for r in base}, {}, []
        common = dict(step_dir=step_dir, step_id=step_id, fallback=base_tree, task_md=task_md,
                      rollouts=base, harness=base_tree, library_dir=lib_dir)
        specs = [dict(label=f"{d.lower()}{i}", dim=d, job=official.propose(self.bp, d, official.SURFACE_NAME, k=k, index=i))
                 for i in range(1, k + 1) for d in ("INV", "DOM") if f"{d.lower()}{i}" in assign]
        got = self.stages(specs, **common)
        stages += got
        trees.update({s.label: s.tree for s in got})
        rollouts.update(self.roll_many(step_dir, step_id, [(s.label, s.tree) for s in got], assign))
        pool = [(s.label, s.tree, rollouts.get(s.label)) for s in got]
        decisions = {s.label: s.notes or s.error for s in got}
        specs = [dict(label=f"agg{i}", dim="AGG", job=official.aggregate(self.bp, official.SURFACE_NAME, k=k, index=i))
                 for i in range(1, k + 1) if f"agg{i}" in assign]
        aggs = self.stages(specs, pool=pool, decisions=decisions, **common)
        stages += aggs
        trees.update({s.label: s.tree for s in aggs})
        rollouts.update(self.roll_many(step_dir, step_id, [(s.label, s.tree) for s in aggs], assign))
        return stages, trees, rollouts, [s.label for s in aggs]

    def run_sequential(self, step_dir, step_id, base, base_tree, lib_dir, task_md, assign):
        k, rollouts, trees, stages = self.k, {r.label: r for r in base}, {}, []
        current, history, finals = dict(base_tree), [], []
        for t in range(1, k + 1):
            specs = [dict(label=f"{d.lower()}{t}", dim=d,
                          job=official.propose(self.bp, d, official.SURFACE_NAME, k=k, index=t,
                                               round_t=(t if k > 1 else None)))
                     for d in ("INV", "DOM") if f"{d.lower()}{t}" in assign]
            common = dict(step_dir=step_dir, step_id=step_id, fallback=current, task_md=task_md, rollouts=base,
                          harness=current, library_dir=lib_dir)
            got = self.stages(specs, pool=list(history), decisions={s.label: s.notes or s.error for s in stages},
                              **common)
            stages += got
            trees.update({s.label: s.tree for s in got})
            rollouts.update(self.roll_many(step_dir, step_id, [(s.label, s.tree) for s in got], assign))
            round_pool = list(history) + [(s.label, s.tree, rollouts.get(s.label)) for s in got]
            ck = f"c{t}"
            if ck not in assign:
                break
            job = (official.aggregate(self.bp, official.SURFACE_NAME, k=1, index=1) if k == 1
                   else official.compose_round(self.bp, official.SURFACE_NAME, t=t, k=k))
            comp = self.stage(label=ck, dim="COMPOSE", job=job, pool=round_pool,
                              decisions={s.label: s.notes or s.error for s in stages}, **common)
            stages.append(comp)
            trees[ck] = comp.tree
            rollouts.update(self.roll_many(step_dir, step_id, [(ck, comp.tree)], assign))
            history = round_pool + [(ck, comp.tree, rollouts[ck])]
            current = comp.tree
            finals.append(ck)
        return stages, trees, rollouts, finals

    # ---------------------------------------------------------------- selection and reflection
    def select(self, step_dir, step_id, finals, rollouts, task_md) -> dict:
        """official selection: trivial at k=1, else the blind selector over the finals' deliverables."""
        pool = {lbl: rollouts[lbl] for lbl in finals if lbl in rollouts and rollouts[lbl].delivered}
        sel = {"chosen_label": "", "why": "", "ok": False, "trivial": False, "dropped": sorted(set(finals) - set(pool))}
        if not pool:
            sel["error"] = "no candidate produced an artifact"
            return sel
        if len(finals) == 1:
            lbl = next(iter(pool))
            return {**sel, "chosen_label": lbl, "ok": True, "trivial": True,
                    "why": "trivial pool: single final at k=1; delivered without a selector session"}
        seed = int(hashlib.sha1(f"{step_id}|{self.k}|{self.mode}".encode()).hexdigest()[:12], 16)
        order = sorted(pool)
        random.Random(seed).shuffle(order)
        id_map = {chr(ord("A") + i): lbl for i, lbl in enumerate(order)}
        root = step_dir / "selection" / f"{self.mode}{self.k}"
        shutil.rmtree(root, ignore_errors=True)
        (root / "candidates").mkdir(parents=True)
        (root / "task.md").write_text(task_md)
        for anon, lbl in id_map.items():
            text = E.deliverable(pool[lbl].episode)
            (root / "candidates" / f"{anon}.md").write_text(
                f"<!-- candidate {anon}: {len(text):,} chars, {text.count(chr(10)) + 1:,} lines. COMPLETE -- nothing "
                f"trimmed. Navigate with grep/offsets; evaluating a candidate means opening the part that determines "
                f"whether it works.\n-->\n" + text)
        ids = ", ".join(sorted(id_map))
        mission = "\n\n".join([
            official.select(self.bp, len(pool), pool_kind="finals"),
            "# YOUR EVIDENCE\n\nIt is on disk. Read it -- do not answer from this message alone.\n\n"
            "context/task.md                 the instructions\n"
            f"context/candidates/<id>.md      one file per candidate: {ids}\n"
            f"\nOpen every candidate before choosing. Choose exactly one of: {ids}."])
        official.official_context.assert_no_grader_verdicts(root)
        try:
            reply = self.session("selector", f"{step_id}/select.{self.mode}{self.k}", root, mission)
            m = _CHOICE.search(reply.text)
            if not m:
                raise ValueError("no ```choice block in the reply")
            anon = m.group(1).strip()
            if anon not in id_map:
                raise ValueError(f"chose {anon!r}, which is not one of {ids}")
            w = _WHY.search(reply.text)
            sel.update(chosen_label=id_map[anon], why=(w.group(1).strip() if w else "")[:2000], ok=True,
                       id_map=id_map)
        except (SessionError, ValueError) as exc:     # official _fallback over a finals pool: the first final
            sel.update(chosen_label=sorted(pool)[0], ok=True, fallback=True, error=str(exc)[:400], id_map=id_map)
        return sel

    def reflect(self, step_dir, step_id, sel, trees, rollouts, decisions, task_md) -> dict:
        """official selection.reflect_adoption + _distil."""
        chosen = sel["chosen_label"]
        root = step_dir / "reflection" / f"{self.mode}{self.k}"
        ctx = E.build(root, task_md, list(rollouts.values()), harness=trees[chosen],
                      pool=[(lbl, trees[lbl], r) for lbl, r in rollouts.items() if not lbl.startswith("base")],
                      decisions=decisions)
        S.write_tree(S.H0, root / "base_harness")
        (root / "adoption.json").write_text(json.dumps({"chosen_label": chosen, "candidate_ids": sel.get("id_map", {}),
                                                        "reason": sel["why"]}, indent=2) + "\n")
        mission = "\n\n".join([
            official.experience(self.bp),
            "# WHAT HAPPENED\n\n"
            f"Harness {chosen} was adopted. The selection reasoning was:\n\n{sel['why']}\n\n"
            "context/adoption.json maps any anonymous IDs in that reasoning to rollout labels. "
            "Compare context/harness/ with context/base_harness/ and the candidates in "
            "context/pool/. Read the decisions and execution contrasts before writing.\n\n" + ctx.layout])
        official.official_context.assert_no_grader_verdicts(root)
        try:
            reply = self.session("reflector", f"{step_id}/experience.{self.mode}{self.k}", root, mission)
        except SessionError:
            return {}
        edits = _EDITS.search(reply.text)
        m = _EXP.search(reply.text)
        if not m:
            return {}
        out = {}
        for line in m.group(1).strip().splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                if key.strip().lower() in ("cue", "lesson", "evidence", "confidence"):
                    out[key.strip().lower()] = val.strip()
        if not E.has_experience(out):
            return {}
        if edits:
            out["edits"] = edits.group(1).strip()
        return out

    # ---------------------------------------------------------------- one task (official schedule.run_task)
    def run_task(self, step_id: str, task_id: str, problems: list, lib: E.Library, *, test: bool) -> dict:
        step_dir = self.run / "steps" / step_id
        summary = step_dir / "summary.json"
        assign = assignment(self.mode, self.k, problems, same_problem=not test)
        if summary.is_file():                     # resume (official schedule._resume/_restore_libraries)
            if self.answers is not None:
                for c, s, i in assign.values():
                    self.answers.mark_seen(c, s, i)
            res = json.loads(summary.read_text())
            if res["library_added"]:
                lib.restore(task_id, step_dir / "library_entry")
            return res
        step_dir.mkdir(parents=True, exist_ok=True)
        base_labels = [f"base{i}" for i in range(1, self.k + 1)]
        base = list(self.roll_many(step_dir, step_id, [(b, S.H0) for b in base_labels], assign).values())
        task_md = E.instructions_of(base[0].episode) + (TEST_NOTE if test else "")
        lib_dir = lib.context_dir()
        run = self.run_parallel if self.mode == "agg" else self.run_sequential
        stages, trees, rollouts, finals = run(step_dir, step_id, base, S.H0, lib_dir, task_md, assign)
        trees.update({b.label: S.H0 for b in base})
        played_finals = [f for f in finals if f in rollouts]
        res = {"step": step_id, "task_id": task_id, "mode": self.mode, "k": self.k,
               "assignment": {lbl: list(p) for lbl, p in assign.items()},
               "stages": [{kk: v for kk, v in s.__dict__.items() if kk != "tree"} for s in stages],
               "finals": played_finals, "library_added": False}
        complete = len(played_finals) == self.k
        if complete:
            sel = self.select(step_dir, step_id, played_finals, rollouts, task_md)
            res["selection"] = sel
            if sel["ok"]:
                decisions = {s.label: s.notes or s.error for s in stages}
                exp = self.reflect(step_dir, step_id, sel, trees, rollouts, decisions, task_md)
                res["experience"] = exp
                if E.has_experience(exp):
                    tree = trees[sel["chosen_label"]]
                    text = E.entry_text(task_id, sel["chosen_label"], tree, sel["why"], exp)
                    bad = guard.violations({**tree, "EXPERIENCE.md": text}, self.answers)
                    try:
                        official.official_context.assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": text})
                    except ValueError as exc:
                        bad.append(str(exc))
                    if bad:
                        res["library_refused"] = bad
                    else:
                        d = lib.add(task_id, tree, why=sel["why"], label=sel["chosen_label"], experience=exp)
                        shutil.rmtree(step_dir / "library_entry", ignore_errors=True)
                        shutil.copytree(d, step_dir / "library_entry")
                        res["library_added"] = True
        else:
            res["note"] = "the stream ended inside this step; no selection, reflection or library update"
        private = {lbl: {"cell": r.episode.get("cell"), "instance": r.episode.get("instance_id"),
                         "score": r.episode.get("score"), "success": r.episode.get("success")}
                   for lbl, r in rollouts.items()}
        E.dump(self.run / "private" / "steps" / f"{step_id}.json", private)
        E.dump(summary, res)
        return res

    # ---------------------------------------------------------------- streams
    def dev_stream(self, problems: list, lib: E.Library) -> list[dict]:
        """official schedule.run_domain over the development problems, threading the library forward."""
        out = []
        for i, p in enumerate(problems, 1):
            lib.retain([f"task-{j:04d}" for j in range(1, i)])
            r = self.run_task(f"dev-{i:04d}", f"task-{i:04d}", [tuple(p)], lib, test=False)
            out.append(r)
            print(f"  [dev] {i}/{len(problems)} added={r['library_added']}", flush=True)
        return out

    def test_stream(self, name: str, problems: list, lib: E.Library) -> list[dict]:
        """One episode per problem in the given order; a step's rollouts take consecutive problems."""
        size = len(labels_in_order(self.mode, self.k))
        out = []
        blocks = [problems[i:i + size] for i in range(0, len(problems), size)]
        for j, block in enumerate(blocks, 1):
            lib.retain([e for e in lib.entries() if not e.startswith("task-t") or int(e[6:]) < j])
            r = self.run_task(f"test-{slug(name)}-{j:03d}", f"task-t{j:04d}", [tuple(p) for p in block], lib, test=True)
            out.append(r)
            print(f"  [test {name}] block {j}/{len(blocks)} added={r['library_added']}", flush=True)
        return out


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
