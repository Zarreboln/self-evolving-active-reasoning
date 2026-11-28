"""Offline check (user, 2026-11-28): can a program assemble a working harness from the graph, with no model?

For each development task in turn: assemble a harness from the graph's nodes and replay the task's recorded probe
episodes on it (the same static check and replay that composed harnesses must pass). Two ways to choose the parts:
  config     the task's own config, as the review wrote it (skeleton, module per slot): tests whether the parts
             plug together at all
  retrieved  the skeleton and, per slot, the module that retrieval by structure ranks first, with the task's own
             memberships removed from the graph: tests the automatic choice as well

Assembly: every node's code is copied in with its top-level names prefixed by the node id (no clashes); the
skeleton's `run_*` function is called with the slots as keyword arguments (slot `instruction` goes to `system`);
a module whose entry takes `env` first is passed as is, any other entry is called with its parameters (the most
similar member's values, else the defaults) and its result is passed.

    python -m mb.evolve.assemble_study ~/evo_b_runs/ours3_mh/state/vNNN
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from mb.evolve import soft_tree3 as S
from mb.evolve import structure as T

RUNS = Path.home() / "evo_b_runs"


def top_names(tree):
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return out


class Prefix(ast.NodeTransformer):
    def __init__(self, names, pre):
        self.names, self.pre = names, pre

    def visit_Name(self, n):
        if n.id in self.names:
            n.id = self.pre + n.id
        return n

    def visit_FunctionDef(self, n):
        if n.name in self.names:
            n.name = self.pre + n.name
        self.generic_visit(n)
        return n


def node_source(nid, code):
    """(imports, body, renamed top-level names) of a node's code with its names prefixed."""
    tree = ast.parse(code)
    names = top_names(tree)
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
            and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
    mod = ast.Module(body=body, type_ignores=[])
    Prefix(names, f"{nid}_").visit(mod)
    return imports, ast.unparse(mod), names


def signature(code, fname):
    for n in ast.parse(code).body:
        if isinstance(n, ast.FunctionDef) and n.name == fname:
            a = n.args
            pos = [x.arg for x in a.args]
            required = pos[:len(pos) - len(a.defaults)]
            return pos + [x.arg for x in a.kwonlyargs], required
    return None, None


def assemble(g, skeleton, slot_modules, params):
    """Harness source: the skeleton's run function called with each slot's module."""
    nodes = g["nodes"]
    used = [skeleton] + [m for m in dict.fromkeys(slot_modules.values()) if m]
    imports, bodies = [], []
    for nid in used:
        im, body, _ = node_source(nid, nodes[nid]["code"])
        imports += im
        bodies.append(f"# ---- node {nid}\n{body}")
    sk = nodes[skeleton]
    run = next((n.name for n in ast.parse(sk["code"]).body if isinstance(n, ast.FunctionDef) and n.name.startswith("run_")), None)
    if not run:
        return None, f"skeleton {skeleton} has no run_* function"
    kw, _ = signature(sk["code"], run)
    lines, notes = [], []
    for slot, m in slot_modules.items():
        if not m:
            continue
        arg = slot if slot in kw else ("system" if slot == "instruction" and "system" in kw else None)
        if not arg:
            notes.append(f"slot {slot}: no matching argument of {run}")
            continue
        entry = nodes[m].get("entry")
        args, required = signature(nodes[m]["code"], entry) if entry else (None, None)
        if not entry or args is None:
            notes.append(f"slot {slot}: module {m} has no entry function")
            continue
        if args and args[0] == "env":
            lines.append(f"    kw[{arg!r}] = {m}_{entry}")
        else:
            p = {k: v for k, v in params.get(m, {}).items() if k in args}
            missing = [r for r in required if r not in p]
            if missing:
                notes.append(f"slot {slot}: {m}.{entry} needs {missing}")
                continue
            lines.append(f"    kw[{arg!r}] = {m}_{entry}(**{p!r})")
    head = "\n".join(dict.fromkeys(imports))
    src = (head + "\n\n" + "\n\n".join(bodies) + "\n\n\ndef run_episode(env, chat):\n    kw = {}\n"
           + ("\n".join(lines) + "\n" if lines else "") + f"    return {skeleton}_{run}(env, chat, **kw)\n")
    return src, "; ".join(notes)


def member_params(g, m, near):
    """Parameter values for module m: those of its member most similar to the task (by structure), else defaults."""
    nd = g["nodes"][m]
    vals = {k: v.get("default") for k, v in (nd.get("params") or {}).items()}
    for x in near:
        if x in nd.get("members", {}) and nd["members"][x].get("params"):
            vals.update(nd["members"][x]["params"])
            break
    return vals


def main():
    state = Path(sys.argv[1]).expanduser()
    g0 = json.loads((state / "graph.json").read_text())
    run = S.Run.__new__(S.Run)
    run.cells = g0["order"]
    run.order = g0["order"]
    S.STRUCT_RETRIEVE = True
    rows = []
    for k, t in enumerate(g0["order"], 1):
        # the graph as a new task t would see it
        g = json.loads(json.dumps(g0))
        for n in list(g["nodes"]):
            g["nodes"][n]["members"].pop(t, None)
            if not g["nodes"][n]["members"]:
                del g["nodes"][n]
        for nd in g["nodes"].values():
            nd["parents"] = [p for p in nd.get("parents", []) if p in g["nodes"]]
            for sv in nd.get("slots", {}).values():
                if sv.get("default") not in g["nodes"]:
                    sv["default"] = None
        idx = g0["order"].index(t) + 1
        traces = sorted(str(p) for p in (RUNS / "ours3_mh" / "exports" / f"t{idx}_probe" / "episodes").glob("*/trace.md"))
        # 1 the task's own config
        cf = g0.get("configs", {}).get(t, {})
        res = {}
        if cf.get("workflow") in g["nodes"]:
            sm = {sl: (m if m in g["nodes"] else None) for sl, m in cf.get("slots", {}).items()}
            par = {m: {**member_params(g, m, []), **cf.get("params", {}).get(m, {})} for m in sm.values() if m}
            src, notes = assemble(g, cf["workflow"], sm, par)
            res["config"] = (src, notes)
        else:
            res["config"] = (None, f"config skeleton {cf.get('workflow')!r} not in the graph without this task")
        # 2 retrieval by structure
        tasks, wfs, mods = run.retrieve(g, t)
        near = list(tasks)
        if wfs:
            sk = next(iter(wfs))
            sm = {}
            for sl, sv in g["nodes"][sk].get("slots", {}).items():
                cands = [m for m in getattr(run, "_by_slot", {}).get(f"{sk}.{sl}", []) if m in g["nodes"]]
                sm[sl] = cands[0] if cands else sv.get("default")
            par = {m: member_params(g, m, near) for m in sm.values() if m}
            src, notes = assemble(g, sk, sm, par)
            res["retrieved"] = (src, notes, sk, sm)
        else:
            res["retrieved"] = (None, "no skeleton retrieved", None, {})
        for way in ("config", "retrieved"):
            src, notes = res[way][0], res[way][1]
            if src is None:
                verdict = "not assembled: " + notes
            else:
                rep = run.crash_report(src, traces)
                verdict = "RUNS" if not rep else "FAILS: " + rep.strip().splitlines()[-1][:160]
                if notes:
                    verdict += f"  [{notes[:160]}]"
            extra = f" skeleton={res[way][2]} slots={res[way][3]}" if way == "retrieved" and res[way][2] else ""
            print(f"{t:28} {way:9} {verdict}{extra}")
            rows.append((t, way, verdict.startswith("RUNS")))
            out = RUNS / "assemble_study" / f"{t.replace(':', '_')}_{way}.py"
            out.parent.mkdir(exist_ok=True)
            if src:
                out.write_text(src)
    for way in ("config", "retrieved"):
        ok = sum(r[2] for r in rows if r[1] == way)
        print(f"{way}: {ok}/{sum(1 for r in rows if r[1] == way)} run on their recorded episodes")


if __name__ == "__main__":
    main()
