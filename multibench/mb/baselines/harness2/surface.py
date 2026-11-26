"""Harness²'s eight-component harness tree, mapped onto our one-file harness.

The editor edits a directory tree exactly as in Harness² (`context/harness/`, files edited in place,
an omitted file is inherited, a deleted file removes the component). Before any play the tree is
assembled into one `harness.py` (our interface: `run_episode(env, chat)`, one Python file, played
only through mb/evolve/broker.py's child sandbox). The mapping, component by component:

| Harness² (OpenCode surface)          | here (same path)                    | in harness.py                         |
|--------------------------------------|-------------------------------------|---------------------------------------|
| C1 `systemprompt.md`                 | `systemprompt.md`                   | SYSTEM_PROMPT, always in the system prompt |
| C2 `guardrails.md`                   | `guardrails.md`                     | GUARDRAILS, always on                 |
| C3 `LongTermMEMORY.md`               | `LongTermMEMORY.md`                 | MEMORY, always on                     |
| C4 `tool_descriptions/tool_guidance.md` | same                             | TOOL_GUIDANCE, always on              |
| C5 `sub_agents/<n>/AGENT.md`         | same                                | SUBAGENTS: own context, one model call, returns text (`TOOL: subagent`) |
| C6 `skills/<n>/SKILL.md`             | same                                | SKILLS: name+description listed, body loaded on demand (`TOOL: skill`) |
| C7 `scripts/<n>.py`                  | same, Python `main(text) -> str`    | SCRIPTS: run on demand (`TOOL: script`) or from a plugin |
| C8 `plugin/<n>.ts` (TypeScript hook in the tool loop) | `plugin/<n>.py`    | PLUGINS: `on_observation` / `on_action` hooks inside the loop |

Not on our surface: skill-local helper scripts (`skills/<n>/scripts/*`) and README stubs.
The fixed loop is runtime.RUNTIME. Python components obey mb/evolve/check.py (the whole assembled
file is checked), so they cannot reach files, the network or the environment's internals.
"""
from __future__ import annotations

import ast
import difflib
import re
import shutil
import textwrap
from pathlib import Path

from mb.evolve.check import problems as static_problems

from .runtime import RUNTIME

Tree = dict[str, str]

ALWAYS_ON = ("systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md")
ALWAYS_ON_MAX = 24_000                       # Harness² OpenCode surface cap (substrate/opencode.py)
AUTOTRIGGER_FILE = "guardrails.md"           # as in Harness² (OpenCode surface)
TRIGGER_LINE = "- When {desc}, load the `{name}` skill.\n"
_NAME = r"[a-z0-9][a-z0-9_\-]*"
RX = {
    "skill": re.compile(rf"skills/({_NAME})/SKILL\.md"),
    "subagent": re.compile(rf"sub_agents/({_NAME})/AGENT\.md"),
    "script": re.compile(r"scripts/([a-z0-9_][a-z0-9_]*)\.py"),
    "plugin": re.compile(r"plugin/([a-z0-9_][a-z0-9_]*)\.py"),
}
HOOKS = ("on_observation", "on_action")

# H0: TextArena's standard agent (seeds/textarena_standard.py), the generic seed of "one harness for
# all tasks"; the other three always-on files start empty.
STANDARD_GAME_PROMPT = ("You are a competitive game player. Make sure you read the game instructions "
                        "carefully, and always follow the required format.")
H0: Tree = {"systemprompt.md": STANDARD_GAME_PROMPT, "guardrails.md": "", "LongTermMEMORY.md": "",
            "tool_descriptions/tool_guidance.md": ""}


def is_allowed(rel: str) -> bool:
    return rel in ALWAYS_ON or any(rx.fullmatch(rel) for rx in RX.values())


def read_tree(d: Path | str, rejected: list[str] | None = None) -> Tree:
    d = Path(d)
    out: Tree = {}
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(d).as_posix()
        if is_allowed(rel):
            out[rel] = p.read_text(encoding="utf-8", errors="replace")
        elif rejected is not None:
            rejected.append(rel)
    return out


def write_tree(tree: Tree, d: Path | str) -> Path:
    d = Path(d)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    for rel in ALWAYS_ON:                      # always-on files exist even when empty
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(tree.get(rel, ""), encoding="utf-8")
    for rel, body in tree.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(body, encoding="utf-8")
    return d


def normalise(tree: Tree) -> Tree:
    out = {rel: tree.get(rel, "") for rel in ALWAYS_ON}
    out.update(tree)
    return out


def diff_summary(before: Tree, after: Tree) -> dict:
    b, a = normalise(before), normalise(after)
    return {"added": sorted(set(a) - set(b)), "removed": sorted(set(b) - set(a)),
            "modified": sorted(r for r in set(a) & set(b) if a[r] != b[r])}


def unified_diff(before: Tree, after: Tree, a_name="current", b_name="candidate") -> str:
    b, a = normalise(before), normalise(after)
    return "".join(line for name in sorted(b.keys() | a.keys())
                   for line in difflib.unified_diff(b.get(name, "").splitlines(keepends=True),
                                                    a.get(name, "").splitlines(keepends=True),
                                                    fromfile=f"{a_name}/{name}", tofile=f"{b_name}/{name}"))


def always_on_text(tree: Tree) -> str:
    return "\n\n".join(tree.get(rel, "") for rel in ALWAYS_ON if tree.get(rel, "").strip())


def frontmatter(body: str) -> dict[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", body or "", re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip().lower()] = v.strip()
    return out


def strip_frontmatter(body: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n?", "", body or "", count=1, flags=re.S)


def names(tree: Tree, kind: str) -> list[str]:
    return sorted({m.group(1) for rel in tree if (m := RX[kind].fullmatch(rel))})


def autotrigger(tree: Tree) -> list[str]:
    """Harness² tree.autotrigger: give every skill a trigger line in always-on text."""
    fixed = []
    for name in names(tree, "skill"):
        if name in always_on_text(tree):
            continue
        fm = frontmatter(tree.get(f"skills/{name}/SKILL.md", ""))
        desc = (fm.get("description") or f"the {name} workflow applies").rstrip(".")
        line = TRIGGER_LINE.format(name=name, desc=desc)
        if len(always_on_text(tree)) + len(line) + 2 > ALWAYS_ON_MAX:
            continue
        cur = tree.get(AUTOTRIGGER_FILE, "")
        tree[AUTOTRIGGER_FILE] = (cur.rstrip() + "\n" + line) if cur.strip() else line
        fixed.append(name)
    return fixed


def _defs(src: str) -> set[str]:
    try:
        return {n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return set()


def validate(tree: Tree) -> list[str]:
    """Every problem, in the actionable prose Harness² feeds back as retry feedback ([] if valid)."""
    errs = []
    body = always_on_text(tree)
    if len(body) > ALWAYS_ON_MAX:
        errs.append(f"always-on content is {len(body)} chars, over the {ALWAYS_ON_MAX} cap; every token "
                    f"here competes with the agent's own output budget")
    for name in names(tree, "skill"):
        fm = frontmatter(tree[f"skills/{name}/SKILL.md"])
        if fm.get("name") != name or not fm.get("description"):
            errs.append(f"skills/{name}/SKILL.md needs frontmatter with `name: {name}` and a `description:`")
        if name not in body:
            errs.append(f"skill `{name}` is never named in systemprompt.md or guardrails.md (add a trigger line)")
    for name in names(tree, "subagent"):
        if not frontmatter(tree[f"sub_agents/{name}/AGENT.md"]).get("description"):
            errs.append(f"sub_agents/{name}/AGENT.md needs frontmatter with a `description:`")
    for name in names(tree, "script"):
        src = tree[f"scripts/{name}.py"]
        if "main" not in _defs(src):
            errs.append(f"scripts/{name}.py must define a top-level `def main(text)` returning a string")
    for name in names(tree, "plugin"):
        if not set(HOOKS) & _defs(tree[f"plugin/{name}.py"]):
            errs.append(f"plugin/{name}.py must define at least one of {', '.join(HOOKS)} at top level")
    scripts = set(names(tree, "script"))
    for rel, src in tree.items():               # Harness² rule 3: every script named must exist
        for ref in re.findall(r"(?:TOOL:\s*script\s+|scripts/)([a-z0-9_]+)", src):
            if ref not in scripts:
                errs.append(f"{rel} names script `{ref}`, but scripts/{ref}.py is not in the harness")
    if not errs:
        errs += [f"assembled harness.py: {p}" for p in static_problems(assemble(tree))]
    return sorted(set(errs))


def _wrap(name: str, src: str, ret: str) -> str:
    return f"def {name}():\n" + textwrap.indent(src.rstrip() + "\n", "    ") + f"    return {ret}\n"


def assemble(tree: Tree) -> str:
    """The one-file harness for a tree: components as data/functions, then the fixed runtime."""
    t = normalise(tree)
    skills = {n: {"description": frontmatter(t[f"skills/{n}/SKILL.md"]).get("description", ""),
                  "body": strip_frontmatter(t[f"skills/{n}/SKILL.md"]).strip()} for n in names(t, "skill")}
    subs = {n: {"description": frontmatter(t[f"sub_agents/{n}/AGENT.md"]).get("description", ""),
                "prompt": strip_frontmatter(t[f"sub_agents/{n}/AGENT.md"]).strip()} for n in names(t, "subagent")}
    out = ['"""Harness² harness, generated from a component tree by mb/baselines/harness2/surface.py."""',
           "import re", "",
           "# C1 systemprompt.md", f"SYSTEM_PROMPT = {t['systemprompt.md'].strip()!r}",
           "# C2 guardrails.md", f"GUARDRAILS = {t['guardrails.md'].strip()!r}",
           "# C3 LongTermMEMORY.md", f"MEMORY = {t['LongTermMEMORY.md'].strip()!r}",
           "# C4 tool_descriptions/tool_guidance.md",
           f"TOOL_GUIDANCE = {t['tool_descriptions/tool_guidance.md'].strip()!r}",
           "# C5 sub_agents/", f"SUBAGENTS = {subs!r}",
           "# C6 skills/", f"SKILLS = {skills!r}", "", "# C7 scripts/"]
    docs = {}
    for n in names(t, "script"):
        src = t[f"scripts/{n}.py"]
        try:
            doc = ast.get_docstring(ast.parse(src)) or ""
        except SyntaxError:
            doc = ""
        docs[n] = doc.strip().splitlines()[0] if doc.strip() else ""
        out.append(_wrap(f"script_component_{n}", src, "main"))
    out.append("SCRIPTS = {" + ", ".join(f"{n!r}: script_component_{n}()" for n in names(t, "script")) + "}")
    out.append(f"SCRIPT_DOCS = {docs!r}")
    out += ["", "# C8 plugin/"]
    for n in names(t, "plugin"):
        src = t[f"plugin/{n}.py"]
        hooks = [h for h in HOOKS if h in _defs(src)]
        out.append(_wrap(f"plugin_component_{n}", src, "{" + ", ".join(f"{h!r}: {h}" for h in hooks) + "}"))
    out.append("PLUGINS = [" + ", ".join(f"plugin_component_{n}()" for n in names(t, "plugin")) + "]")
    return "\n".join(out) + "\n" + RUNTIME


def materialise(tree: Tree, d: Path | str) -> Path:
    """Write the tree and its assembled harness: d/tree/..., d/harness.py (what play() runs)."""
    d = Path(d)
    write_tree(tree, d / "tree")
    (d / "harness.py").write_text(assemble(tree), encoding="utf-8")
    return d / "harness.py"
