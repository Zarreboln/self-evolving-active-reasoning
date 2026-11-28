"""Offline study of retrieval for new tasks (user, 2026-11-28): leave one development task out, remove its traces
from the graph, rank the graph's nodes for it with each method, and check whether the ranking recovers the nodes the
task actually belongs to (its memberships, and in a --repr-d graph its config).

    python -m mb.evolve.retrieval_study ~/evo_b_runs/<run>/state/vNNN [--tau 0.1]

Methods
  desc   the current retrieval: workflows by their nearest member's description similarity; modules by their best
         match to the task's required capabilities
  B      bottom-up: sum over earlier tasks x of sim(t, x) * w(x, n), where w is the evidence edge weight (how much of
         x's own code the node's code covers), then passed up the parent links
  popular  nodes ranked by how many tasks share them, ignoring the new task (a check: a method must beat it)
  A      per level (HSML): at each level a softmax over siblings of -distance(task, node centre) / tau; a node's weight
         is the sum over its paths of the product along the path
Metrics per held-out task: precision at k (k = its true node count), recall at 5, and whether the top workflow is
one of its own; averaged over tasks.
"""
from __future__ import annotations

import ast
import difflib
import json
import math
import statistics
import sys
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def functions_source(code, names):
    """Source text of the named top-level (or nested) functions in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    lines, out = code.splitlines(), []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            out.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    return "\n".join(out)


def edge_weight(node_code, member_code, member):
    """How much of the member's own code (its evidence functions and excerpts) the node's code matches, 0..1."""
    ev = member.get("evidence", {})
    src = functions_source(member_code, set(ev.get("functions", []))) + "\n" + "\n".join(ev.get("excerpts", []))
    if not src.strip():
        return 0.0
    tok = lambda s: s.replace("(", " ( ").replace(")", " ) ").split()
    return difflib.SequenceMatcher(None, tok(node_code), tok(src), autojunk=False).ratio()


def held_out_graph(g, t):
    """The graph as a new task t would see it: t's memberships and leaf removed; nodes left with no member dropped."""
    nodes = {}
    for n, nd in g["nodes"].items():
        mem = {x: m for x, m in nd.get("members", {}).items() if x != t}
        if mem:
            nodes[n] = dict(nd, members=mem)
    for nd in nodes.values():
        nd["parents"] = [p for p in nd.get("parents", []) if p in nodes]
    return nodes


def truth(g, t):
    return {n for n, nd in g["nodes"].items() if t in nd.get("members", {})}


def rank_desc(g, nodes, t):
    v, out = g["vecs"][t], {}
    caps = [x["vec"] for x in g.get("capabilities", {}).get(t, [])]
    for n, nd in nodes.items():
        if nd.get("kind") == "workflow":
            out[n] = max(dot(v, g["vecs"][x]) for x in nd["members"])
        else:
            dv = nd.get("desc_vec")
            out[n] = max((dot(cv, dv) for cv in caps), default=0.0) if dv else 0.0
    return out


def rank_B(g, nodes, t, weights):
    v = g["vecs"][t]
    others = [x for x in g["order"] if x != t]
    sims = {x: max(dot(v, g["vecs"][x]), 0.0) for x in others}
    z = sum(sims.values()) or 1.0
    base = {n: sum(sims[x] * weights[(n, x)] for x in nd["members"] if x in sims) / z for n, nd in nodes.items()}
    out = dict(base)                                   # pass up: a parent gains what its children carry
    for _ in range(4):
        for n, nd in nodes.items():
            for p in nd.get("parents", []):
                out[p] = max(out[p], base[p] + out[n] * 0.5)
    return out


def rank_A(g, nodes, t, tau):
    v = g["vecs"][t]
    centre = {n: [statistics.mean(g["vecs"][x][i] for x in nd["members"]) for i in range(len(v))] for n, nd in nodes.items()}
    others = [x for x in g["order"] if x != t]
    pair = [dist(g["vecs"][a], g["vecs"][b]) for i, a in enumerate(others) for b in others[i + 1:]]
    scale = statistics.median(pair) if pair else 1.0
    children = {n: [] for n in nodes}
    for n, nd in nodes.items():
        for p in nd.get("parents", []):
            children[p].append(n)
    roots = [n for n, nd in nodes.items() if not nd.get("parents")]

    def soft(group):
        s = {n: -(dist(v, centre[n]) / scale) ** 2 / (2 * tau) for n in group}
        m = max(s.values())
        e = {n: math.exp(x - m) for n, x in s.items()}
        z = sum(e.values())
        return {n: x / z for n, x in e.items()}

    out = {n: 0.0 for n in nodes}
    frontier = soft(roots) if roots else {}
    for _ in range(6):                                  # depth is small; a node may be reached by several paths
        nxt = {}
        for n, w in frontier.items():
            out[n] += w
            if children[n]:
                for ch, cw in soft(children[n]).items():
                    nxt[ch] = nxt.get(ch, 0.0) + w * cw
        frontier = nxt
        if not frontier:
            break
    return out


def score(rank, true):
    order = sorted(rank, key=lambda n: -rank[n])
    k = max(len(true), 1)
    p_at_k = len(set(order[:k]) & true) / k
    r_at_5 = len(set(order[:5]) & true) / len(true) if true else float("nan")
    return p_at_k, r_at_5


def main():
    src = Path(sys.argv[1]).expanduser()
    tau = float(sys.argv[sys.argv.index("--tau") + 1]) if "--tau" in sys.argv else 0.1
    g = json.loads((src / "graph.json").read_text())
    weights = {}
    for n, nd in g["nodes"].items():
        for x, m in nd.get("members", {}).items():
            weights[(n, x)] = edge_weight(nd.get("code", ""), g["leaves"][x]["code"], m)
    rows = {"popular": [], "desc": [], "B": [], "A": []}
    wf_hit = {m: [] for m in rows}
    rows_specific = {m: [] for m in rows}       # only nodes that fewer than half the other tasks share
    for t in g["order"]:
        true = truth(g, t)
        if not true:
            continue
        nodes = held_out_graph(g, t)
        ranks = {"popular": {n: len(nd["members"]) for n, nd in nodes.items()},
                 "desc": rank_desc(g, nodes, t), "B": rank_B(g, nodes, t, weights), "A": rank_A(g, nodes, t, tau)}
        true &= set(nodes)
        wf_true = {n for n in true if nodes[n].get("kind") == "workflow"}
        half = (len(g["order"]) - 1) / 2
        specific = {n for n in nodes if len(nodes[n]["members"]) < half}
        for m, r in ranks.items():
            rows[m].append(score(r, true))
            if true & specific:
                rows_specific[m].append(score({n: r[n] for n in specific}, true & specific))
            wfs = [n for n in sorted(r, key=lambda n: -r[n]) if nodes[n].get("kind") == "workflow"]
            if wf_true and wfs:
                wf_hit[m].append(wfs[0] in wf_true)
        print(f"{t:28} true={sorted(true)}")
        for m, r in ranks.items():
            top = sorted(r, key=lambda n: -r[n])[:5]
            print(f"   {m:5} top5={top}")
    print(f"\n{len(rows['desc'])} held-out tasks; tau={tau}")
    print(f"{'method':8}{'P@k':>8}{'R@5':>8}{'top workflow is its own':>26}")
    for m in rows:
        p = statistics.mean(x for x, _ in rows[m])
        r = statistics.mean(y for _, y in rows[m])
        w = f"{sum(wf_hit[m])}/{len(wf_hit[m])}" if wf_hit[m] else "-"
        print(f"{m:8}{p:8.2f}{r:8.2f}{w:>26}")
    print("\nonly the less shared nodes (members < half of the other tasks):")
    for m in rows_specific:
        if rows_specific[m]:
            p = statistics.mean(x for x, _ in rows_specific[m])
            r = statistics.mean(y for _, y in rows_specific[m])
            print(f"{m:8}{p:8.2f}{r:8.2f}   ({len(rows_specific[m])} tasks)")
    ew = sorted(weights.values())
    print(f"\nedge weights (code overlap): median {statistics.median(ew):.2f}, min {ew[0]:.2f}, max {ew[-1]:.2f}")


if __name__ == "__main__":
    main()
