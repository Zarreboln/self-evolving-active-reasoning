"""Interface check of a components file (stdlib only: also copied into the Builder's workspace).

The analog of the released validation (schema.HarnessBundle.from_dict, ModularSession._add_program):
every component present, None or a dict literal of callbacks naming top-level functions with the
required arities, tool names valid and not colliding with the fixed capabilities.
"""
from __future__ import annotations

import ast
import re

COMPONENTS = ("INSTRUCTION", "MEMORY", "CONTEXT", "TOOLS", "CONTROLLER", "VERIFICATION", "WORKSPACE")
# callback -> number of arguments (released ModularSession requirements; before_finish -> before_action)
CALLBACKS = {"MEMORY": {"write": 2, "retrieve": 2}, "CONTEXT": {"select_context": 2},
             "CONTROLLER": {"on_event": 2}, "VERIFICATION": {"before_action": 2},
             "WORKSPACE": {"setup": 1}}
OPTIONAL = {"VERIFICATION": {"on_failure": 2}}
PRIMITIVES = {"act", "workbench_write", "workbench_read", "workbench_list", "finish"}
TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")      # released schema._TOOL_NAME
RESERVED = re.compile(r"^(rt_|Rt|RT_)")
MAX_TOOLS = 32                                                # released schema.MAX_TOOLS
MAX_INSTRUCTION_BYTES = 16_384                                # released schema.MAX_INSTRUCTION_BYTES


def _arity(fn: ast.FunctionDef) -> int | None:
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        return None
    return len(a.args)


def component_problems(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    out = []
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigned = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for t in targets:
            if isinstance(t, ast.Name):
                if t.id in assigned and t.id in COMPONENTS:
                    out.append(f"{t.id} is assigned more than once")
                assigned[t.id] = node.value
        for name in ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                     [t.id for t in targets if isinstance(t, ast.Name)]):
            if RESERVED.match(name):
                out.append(f"line {node.lineno}: names starting with rt_/Rt/RT_ belong to the fixed runner ({name})")
            if name == "run_episode":
                out.append(f"line {node.lineno}: run_episode belongs to the fixed runner")
    for name in COMPONENTS:
        if name not in assigned:
            out.append(f"component {name} is missing (use None to disable it)")
    if out:
        return out

    try:
        instr = ast.literal_eval(assigned["INSTRUCTION"])
        if not isinstance(instr, str):
            out.append("INSTRUCTION must be a string")
        elif len(instr.encode()) > MAX_INSTRUCTION_BYTES:
            out.append(f"INSTRUCTION exceeds {MAX_INSTRUCTION_BYTES} bytes")
    except ValueError:
        out.append("INSTRUCTION must be a string literal")

    def callback_problems(comp, node, required, optional):
        if isinstance(node, ast.Constant) and node.value is None:
            return []
        if not isinstance(node, ast.Dict):
            return [f"{comp} must be None or a dict literal of callbacks"]
        p, keys = [], []
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                p.append(f"{comp}: callback names must be string literals")
                continue
            keys.append(k.value)
            want = required.get(k.value, optional.get(k.value))
            if want is None:
                p.append(f"{comp}: unknown callback {k.value!r} (allowed: {sorted({**required, **optional})})")
            elif not (isinstance(v, ast.Name) and v.id in funcs):
                p.append(f"{comp}.{k.value} must name a top-level function of this file")
            elif _arity(funcs[v.id]) != want:
                p.append(f"{comp}.{k.value} must accept {want} arguments")
        for k in required:
            if k not in keys:
                p.append(f"{comp}.{k} is required when {comp} is enabled")
        return p

    for comp, req in CALLBACKS.items():
        out += callback_problems(comp, assigned[comp], req, OPTIONAL.get(comp, {}))

    tools = assigned["TOOLS"]
    if not isinstance(tools, ast.List):
        out.append("TOOLS must be a list literal (empty to disable)")
    else:
        if len(tools.elts) > MAX_TOOLS:
            out.append(f"at most {MAX_TOOLS} tools")
        names = []
        for i, t in enumerate(tools.elts):
            if not isinstance(t, ast.Dict):
                out.append(f"TOOLS[{i}] must be a dict literal")
                continue
            fields = {k.value: v for k, v in zip(t.keys, t.values) if isinstance(k, ast.Constant)}
            if set(fields) != {"name", "description", "parameters", "next_action"}:
                out.append(f"TOOLS[{i}] needs exactly name, description, parameters, next_action")
                continue
            try:
                name = ast.literal_eval(fields["name"])
                desc = ast.literal_eval(fields["description"])
                params = ast.literal_eval(fields["parameters"])
            except ValueError:
                out.append(f"TOOLS[{i}]: name, description and parameters must be literals")
                continue
            if not isinstance(name, str) or not TOOL_NAME.fullmatch(name) or name in PRIMITIVES:
                out.append(f"TOOLS[{i}]: invalid tool name {name!r} (or it collides with a fixed capability)")
            names.append(name)
            if not isinstance(desc, str):
                out.append(f"TOOLS[{i}]: description must be a string")
            if not (isinstance(params, dict) and params.get("type") == "object"):
                out.append(f"TOOLS[{i}]: parameters must be a JSON schema of type object")
            v = fields["next_action"]
            if not (isinstance(v, ast.Name) and v.id in funcs):
                out.append(f"TOOLS[{i}].next_action must name a top-level function of this file")
            elif _arity(funcs[v.id]) != 3:
                out.append(f"TOOLS[{i}].next_action must accept 3 arguments")
        if len(set(names)) != len(names):
            out.append("duplicate tool names")
    return out
