"""Mac-side driver for the ACE baseline of experiment B.

ACE's own roles run here (Claude Code on the subscription, in the Docker sandbox); episodes are played
on the cluster by the long-lived worker (mb/evolve/worker.py) through the git mailbox.

    # on the cluster, once (mb.benches needs the benchmark packages): problem lists
    python -m mb.baselines.ace.run inputs --out ace_inputs/       # instances.json, order.json

    # on the Mac, with a worker job running on ORCD
    python -m mb.baselines.ace.run offline --run aceB --mailbox ~/mailbox_play --instances instances.json
    python -m mb.baselines.ace.run test    --run aceB --mailbox ~/mailbox_play --order order.json --variant ace
    python -m mb.baselines.ace.run test    --run aceB --mailbox ~/mailbox_play --order order.json --variant ace_online_only
    python -m mb.baselines.ace.run report  --run aceB --mailbox ~/mailbox_play      # our scores, for the paper

Every command resumes from what is on disk under ~/evo_b_runs/<run>/ace/.
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mb.baselines.ace import core, guard

RUNS = Path.home() / "evo_b_runs"
DEV_CELLS = ["wordle:basic5", "wordle:basic7", "hangman:basic", "babyai:GoToLocal", "babyai:Open",
             "sciworld:test-conductivity", "dc:all"]
TEST_TASKS = {"mastermind_basic": ["mastermind:basic"], "alfworld_pick_and_place_simple": ["alfworld:pick_and_place_simple"],
              "twentyq_basic": ["twentyq:basic-people", "twentyq:basic-places", "twentyq:basic-things"],
              "minesweeper_basic": ["minesweeper:basic"], "wordle_hard5": ["wordle:hard5"],
              "hangman_hard": ["hangman:hard"], "sciworld_grow-plant": ["sciworld:grow-plant"]}
N_SEARCH, N_TEST = 20, 50
BUDGET = 8 * 7 * N_SEARCH          # equal episode budget before deployment: 8 rounds x 140
TOKEN_BUDGET = 16_000              # playbook size cap; see core.py, "Deviations"


def interleave(instances):
    """ACE goes through its training samples in order; ours are 7 tasks' search problems, taken task
    by task (problem 1 of every task, then problem 2, ...)."""
    per = [[(c, i) for i in instances[c]["search"][:N_SEARCH]] for c in DEV_CELLS]
    return [p for k in range(max(map(len, per))) for p in (q[k] for q in per if k < len(q))]


def make_llm(root, fake):
    if fake:
        from mb.baselines.ace.fakes import FakeLLM
        return FakeLLM()
    from mb.baselines.ace.llm import ClaudeCodeLLM
    return ClaudeCodeLLM(root / "llm_ledger.jsonl")


def cmd_offline(a):
    root = RUNS / a.run / "ace" / "offline"
    inst = json.loads(Path(a.instances).read_text())
    from mb.baselines.ace.backend import MailboxBackend
    run = core.ACERun(root, make_llm(RUNS / a.run / "ace", False), MailboxBackend(a.mailbox),
                      f"ace/{a.run}/offline", a.token_budget, a.seed)
    final = core.offline(run, interleave(inst), a.budget)
    check_copying(root, final, inst)


def check_copying(root, playbook, inst):
    rep = {"names": guard.NAME_RE.findall(playbook),
           "copying": {c: guard.copying(playbook, inst[c]["search"][:N_SEARCH], inst[c]["selection"])
                       for c in DEV_CELLS}}
    core._atomic(root / "store_check.json", rep)
    bad = rep["names"] or any(rep["copying"].values())
    print("store check:", "FLAGGED " + json.dumps(rep) if bad else "clean", flush=True)
    return not bad


def cmd_test(a):
    order = json.loads(Path(a.order).read_text())
    base = RUNS / a.run / "ace"
    if a.variant == "ace":
        off = base / "offline"
        if not (off / "offline_done.json").exists():
            raise SystemExit("offline phase not finished")
        chk = json.loads((off / "store_check.json").read_text())
        if (chk["names"] or any(chk["copying"].values())) and not a.force:
            raise SystemExit(f"offline playbook flagged by the store check: {chk}")
        init = (off / "final_playbook.txt").read_text()
    else:
        init = None                                   # ACE's empty playbook
    from mb.baselines.ace.backend import MailboxBackend
    backend, llm = MailboxBackend(a.mailbox), make_llm(base, False)
    tasks = [t for t in order if not a.tasks or t in a.tasks.split(",")]

    def one(task):
        run = core.ACERun(base / a.variant / task, llm, backend, f"ace/{a.run}/{a.variant}/{task}",
                          a.token_budget, a.seed, initial_playbook=init)
        core.online(run, [tuple(p) for p in order[task]])
        return task
    with ThreadPoolExecutor(a.workers) as ex:     # tasks are independent streams; problems are sequential
        for t in ex.map(one, tasks):
            print("done", t, flush=True)


def cmd_report(a):
    from mb.baselines.ace.backend import MailboxBackend
    backend = MailboxBackend(a.mailbox)
    out = {}
    for v in ("ace", "ace_online_only"):
        for ev in sorted((RUNS / a.run / "ace" / v).glob("*/events.jsonl")):
            rows = []
            for line in ev.read_text().splitlines():
                e = json.loads(line)
                s = backend.private_scores(e["key"]) or {}
                rows.append({"index": e["index"], "instance": e["instance"], "score": s.get("mean_score")})
            out.setdefault(v, {})[ev.parent.name] = rows
    p = RUNS / a.run / "ace" / "report.json"
    core._atomic(p, out)
    for v, tasks in out.items():
        for t, rows in tasks.items():
            sc = [r["score"] for r in rows if r["score"] is not None]
            print(v, t, len(rows), round(sum(sc) / len(sc), 4) if sc else None)


def cmd_inputs(a):
    """Cluster side: the problem lists (needs mb.benches with the benchmark packages)."""
    from mb.benches import get
    inst = {}
    for c in DEV_CELLS:
        b, s = c.split(":", 1)
        inst[c] = {"search": get(b).instances(s, "search", N_SEARCH), "selection": get(b).instances(s, "selection")}
    order = {}
    for task, cells in TEST_TASKS.items():
        pool = [(c, i) for c in cells for i in get(c.split(":")[0]).instances(c.split(":", 1)[1], "test")]
        rng = random.Random(f"exp_b/test_order/{task}")
        rng.shuffle(pool)
        order[task] = pool[:N_TEST]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instances.json").write_text(json.dumps(inst, indent=1))
    (out / "order.json").write_text(json.dumps(order, indent=1))
    print({t: len(v) for t, v in order.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["offline", "test", "report", "inputs"])
    ap.add_argument("--run", default="aceB")
    ap.add_argument("--mailbox", default="~/mailbox_play", help="local clone of the play mailbox")
    ap.add_argument("--instances", help="instances.json (cmd inputs)")
    ap.add_argument("--order", help="order.json: {task: [[cell, instance], ...]}, the order shared by all methods")
    ap.add_argument("--variant", choices=["ace", "ace_online_only"], default="ace")
    ap.add_argument("--tasks", default="", help="comma-separated subset of the order's tasks")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    ap.add_argument("--seed", default=None, help="one seed for every task (default: each benchmark's official agent)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="ace_inputs")
    a = ap.parse_args()
    {"offline": cmd_offline, "test": cmd_test, "report": cmd_report, "inputs": cmd_inputs}[a.cmd](a)


if __name__ == "__main__":
    main()
