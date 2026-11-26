"""Evidence trees and the library: ports of the official core/context.py and core/library.py.

What a session can read is built here and nowhere else. A rollout is shown through what the playing
model saw and did (every message, reply, action and observation; mb/evolve/broker.py's trace
without its score line) plus its "deliverable": the sequence of actions with the environment's reply
to each. The episode's score, success flag, instance key and our action-source labels are never
written into an evidence tree; neither is anything from a split the method has not played.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import official
from . import surface as S
from .runtime import RUNTIME

OBS_TRUNC = 50_000                     # official BenchSpec.obs_trunc default


@dataclass
class Rollout:
    """What the method knows about one finished rollout (official RunView). Never carries a score."""
    label: str
    key: str
    episode: dict                      # the broker record; only the fields read below are used

    @property
    def ok(self):
        return self.episode.get("end") in ("end", "crash") or bool(self.episode.get("steps_log"))

    @property
    def delivered(self):
        return bool(self.episode.get("steps_log"))


def _trunc(s, n=OBS_TRUNC):
    s = str(s or "")
    return s if len(s) <= n else s[:n] + f"\n[... renderer truncated {len(s) - n:,} chars ...]"


def _clean(text):
    return re.sub(r'File "[^"]*/([^/"]+)"', r'File "\1"', text or "")


def trajectory(label: str, e: dict) -> str:
    """broker._render, minus the score/success header and the instance id."""
    lines = [f"# rollout {label}", "",
             f"actions {len(e.get('steps_log', []))} · episode ended {bool(e.get('done'))} · harness "
             f"{e.get('end')} · error {_clean(e.get('error')) or '-'}", ""]
    prev, by_step = [], {}
    for c in e.get("calls", []):
        by_step.setdefault(c["step"], []).append(c)
    steps = e.get("steps_log", [])
    for k in range(len(steps) + 1):
        for c in by_step.get(k, []):
            msgs = c["messages"]
            same = 0
            while same < min(len(prev), len(msgs)) and prev[same] == msgs[same]:
                same += 1
            lines.append(f"## [MODEL CALL] (before action {k + 1})")
            if same:
                lines.append(f"_[first {same} messages identical to the previous call]_")
            for m in msgs[same:]:
                lines += [f"**{m.get('role')}:**", "```", _trunc(m.get("content", "")), "```"]
            lines += ["**reply:**" if "reply" in c else f"**[ERROR]** {c.get('error')}", "```",
                      _trunc(c.get("reply", "")), "```", ""]
            prev = msgs + ([{"role": "assistant", "content": c["reply"]}] if "reply" in c else [])
        if k < len(steps):
            s = steps[k]
            lines += [f"## [ACTION] {k + 1}", "```", s["action"], "```", "## [OBSERVATION]", "```",
                      _trunc(s["observation"]), "```", ""]
    if e.get("harness_stderr"):
        lines += ["## harness stderr (last 2000 chars)", "```", _clean(e["harness_stderr"]), "```"]
    return "\n".join(lines)


def deliverable(e: dict) -> str:
    steps = e.get("steps_log", [])
    if not steps:
        return "(no action was sent)"
    out = ["The actions sent to the environment, each with the environment's reply:", ""]
    for i, s in enumerate(steps, 1):
        out += [f"{i}. action:", "```", s["action"], "```", "   reply:", "```", _trunc(s["observation"]), "```"]
    out.append("(the episode ended)" if e.get("done") else
               f"(the episode did not end: the harness stopped -- {e.get('end')}"
               f"{': ' + _clean(e.get('error'))[:300] if e.get('error') else ''})")
    return "\n".join(out)


def instructions_of(e: dict) -> str:
    """The task as the agent received it: the first user message of an H0 rollout (= env.instructions)."""
    for c in e.get("calls", []):
        for m in c["messages"]:
            if m.get("role") == "user":
                return str(m.get("content", ""))
    return "(the instructions are not available)"


def _copy(src: Path, dst: Path) -> int:
    n = 0
    for p in sorted(Path(src).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            t = dst / p.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
            n += 1
    return n


def build(dest: Path, task_md: str, rollouts: Sequence[Rollout] = (), *, harness: S.Tree | None = None,
          library_dir: Path | None = None, pool: Sequence[tuple[str, S.Tree, Rollout | None]] = (),
          decisions: dict | None = None):
    """official context.build: task.md, decisions/, rollouts/, deliverables/, harness/, pool/, library/;
    plus runtime/harness_runtime.py (the fixed loop, ours). Returns the official ContextTree."""
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    (dest / "task.md").write_text(task_md)
    (dest / "runtime").mkdir()
    (dest / "runtime" / "harness_runtime.py").write_text(RUNTIME.lstrip())
    for label, notes in (decisions or {}).items():
        (dest / "decisions").mkdir(exist_ok=True)
        (dest / "decisions" / f"{label}.md").write_text(notes or "No decision notes returned.")
    got = []
    if rollouts:
        (dest / "rollouts").mkdir()
        (dest / "deliverables").mkdir()
    for r in rollouts:
        (dest / "rollouts" / f"{r.label}.md").write_text(trajectory(r.label, r.episode))
        (dest / "deliverables" / f"{r.label}.md").write_text(deliverable(r.episode))
        got.append(r.label)
    n_h = 0
    if harness is not None:
        S.write_tree(harness, dest / "harness")
        n_h = sum(1 for p in (dest / "harness").rglob("*") if p.is_file())
    got_p = []
    for label, tree, roll in pool:
        d = dest / "pool" / label
        S.write_tree(tree, d / "harness")
        (d / "harness.diff").write_text(S.unified_diff(harness or {}, tree, "current", label))
        if roll is not None:
            (d / "rollout.md").write_text(trajectory(label, roll.episode))
            (d / "produced.md").write_text(deliverable(roll.episode))
        got_p.append(label)
    lib = False
    if library_dir and Path(library_dir).is_dir() and any(Path(library_dir).iterdir()):
        _copy(Path(library_dir), dest / "library")
        lib = True
    official.official_context.assert_no_grader_verdicts(dest)
    return official.ContextTree(root=dest, task_id="", rollouts=got, deliverables=got, harness_files=n_h,
                                library=lib, pool=got_p, decisions=len(decisions or {}))


# ------------------------------------------------------------------------------------------- library

def has_experience(exp) -> bool:
    return isinstance(exp, dict) and all(isinstance(exp.get(k), str) and exp[k].strip()
                                         for k in ("cue", "lesson", "evidence"))


def _inventory(tree: S.Tree) -> str:
    st = {"skills": len(S.names(tree, "skill")), "sub_agents": len(S.names(tree, "subagent")),
          "scripts": len(S.names(tree, "script")), "plugins": len(S.names(tree, "plugin"))}
    bits = [f"{v} {k}" for k, v in st.items() if v] or ["always-on edits only"]
    return f"{', '.join(bits)}; {len(S.always_on_text(tree)):,} always-on chars"


def entry_text(task_id, label, tree, why, exp) -> str:
    """EXPERIENCE.md exactly as official Library.add writes it."""
    lines = [f"# {task_id}", "", f"- adopted: `{label}`", f"- components: {_inventory(tree)}", ""]
    if why.strip():
        lines += ["## Why this was chosen", why.strip(), ""]
    if exp.get("edits"):
        lines += ["## What each edit did", str(exp["edits"]).strip(), ""]
    if exp.get("lesson"):
        lines += ["## Transferable lesson"] + [f"- **{f}**: {str(exp[f]).strip()}"
                                              for f in ("cue", "lesson", "evidence", "confidence") if exp.get(f)] + [""]
    return "\n".join(lines)


class Library:
    """official core/library.Library with one domain (our whole task stream shares one library).

    Entries are keyed by anonymous task ids (task-0001, ...): our problem ids name the game and, for
    some games, are the answer itself."""

    def __init__(self, root: Path):
        self.dir = Path(root)

    @property
    def index(self):
        return self.dir / "INDEX.md"

    def entries(self):
        d = self.dir / "harnesses"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def context_dir(self):
        return self.dir if self.index.is_file() and any(self.dir.rglob("EXPERIENCE.md")) else None

    def retain(self, keep: Sequence[str]):
        drop = [e for e in self.entries() if e not in set(keep)]
        for e in drop:
            shutil.rmtree(self.dir / "harnesses" / e, ignore_errors=True)
        if drop:
            self.rebuild_index()

    def add(self, task_id, tree, *, why, label, experience):
        d = self.dir / "harnesses" / task_id
        shutil.rmtree(d, ignore_errors=True)
        S.write_tree(tree, d / "harness")
        (d / "EXPERIENCE.md").write_text(entry_text(task_id, label, tree, why, experience))
        self.rebuild_index()
        return d

    def restore(self, task_id, saved: Path):
        """Put back an entry a finished step added (a resumed stream first drops later entries)."""
        d = self.dir / "harnesses" / task_id
        if not d.is_dir():
            shutil.copytree(saved, d)
            self.rebuild_index()

    def rebuild_index(self):
        rows = ["# Library — area", "",
                "Each entry is an earlier task in THIS domain whose harness was chosen, stored with its "
                "FULL component tree and why it was picked. Browse this index, then open the entries that "
                "look relevant to the matter in front of you.", ""]
        names = self.entries()
        if not names:
            rows.append("_(empty — this is the first task in this domain)_")
        for name in names:
            ed = self.dir / "harnesses" / name
            cue = ""
            for line in (ed / "EXPERIENCE.md").read_text().splitlines():
                if line.startswith("- **cue**:"):
                    cue = line.split(":", 1)[1].strip()
            rows += [f"## `harnesses/{name}/`",
                     f"- harness: `harnesses/{name}/harness/` — {_inventory(S.read_tree(ed / 'harness'))}",
                     f"- why:     `harnesses/{name}/EXPERIENCE.md`"]
            if cue:
                rows.append(f"- cue:     {cue}")
            rows.append("")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index.write_text("\n".join(rows))

    def snapshot(self, dest: Path):
        shutil.rmtree(dest, ignore_errors=True)
        if self.dir.is_dir():
            shutil.copytree(self.dir, dest)
        return Library(dest)


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)
