"""Cost ledger of experiment B: code-model (Opus) sessions and tokens per method, policy-model calls and
tokens per method, GPU hours, wall-clock times. Writes ~/evo_b_runs/LEDGER.md.

    python -m mb.evolve.ledger            # on the Mac (reaches ORCD over the existing ssh connection)

Opus runs on the Claude subscription, so its dollar figure is the list-price equivalent Claude Code reports
(total_cost_usd), not money spent. Proposer sessions of discarded candidates are counted from the run logs'
"proposer rc=.. cost≈$.." lines, so wasted sessions stay in the ledger.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

RUNS = Path.home() / "evo_b_runs"
TOK = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def method_of(run: str) -> str:
    if run.startswith("mh_pbta_"):
        return "MH per benchmark (generic start)"
    if run.startswith("mh_pb_"):
        return "MH per benchmark (official start)"
    return {"mh_global": "MH global", "ours": "Ours", "abl_hard": "Ablation: hard tree",
            "abl_flat": "Ablation: flat", "h2": "Harness²", "ms": "Meta-Skills",
            "ours2": "Ours (incremental)", "ours2_hard": "Ours ablation: hard", "ours2_flat": "Ours ablation: flat",
            "ours2_pilot": "Ours pilot (discarded)", "ours3": "Ours v3 (graph of distilled code)",
            "ours3_hard": "Ours v3 ablation: hard", "ours3_flat": "Ours v3 ablation: flat"}.get(run, run)


def last_result(f: Path):
    r = None
    for line in f.open(errors="ignore"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "result":
            r = e
    return r


def opus() -> dict:
    agg = defaultdict(lambda: defaultdict(float))
    # Meta-Harness and the soft tree: one Claude Code stream log per proposer session
    for f in (list(RUNS.glob("mh_*/private/c*/proposer.jsonl")) + list(RUNS.glob("*/private/r*_proposer.jsonl"))
              + list(RUNS.glob("ours[23]*/private/t*_proposer.jsonl"))):
        run = f.relative_to(RUNS).parts[0]
        r = last_result(f)
        a = agg[method_of(run)]
        a["sessions"] += 1
        if r:
            for k in TOK:
                a[k] += (r.get("usage") or {}).get(k, 0) or 0
            a["usd_list_equiv"] += r.get("total_cost_usd", 0) or 0
    # sessions whose logs were discarded with their candidate: from the run logs
    for log in RUNS.glob("mh_*.log"):
        logged = len(re.findall(r"proposer rc=", log.read_text(errors="ignore")))
        kept = len(list((RUNS / log.stem / "private").glob("c*/proposer.jsonl"))) if (RUNS / log.stem).exists() else 0
        if logged > kept:
            costs = [float(x) for x in re.findall(r"proposer rc=\S+ cost≈\$([0-9.]+)", log.read_text(errors="ignore"))]
            a = agg[method_of(log.stem)]
            a["sessions_discarded"] += logged - kept
    # Harness²: CachedSession meta files
    for f in RUNS.glob("harness2/**/sessions/*.meta.json"):
        u = json.loads(f.read_text()).get("usage") or {}
        a = agg["Harness²"]
        a["sessions"] += 1
        for k in TOK:
            a[k] += u.get(k, 0) or 0
        a["usd_list_equiv"] += u.get("cost_usd_equiv", 0) or 0
    # Meta-Skills: private/sessions.jsonl
    for f in RUNS.glob("metaskill_*/private/sessions.jsonl"):
        for line in f.open():
            s = json.loads(line)
            u = s.get("usage") or {}
            a = agg["Meta-Skills"]
            a["sessions"] += 1
            for k, k2 in (("input", "input_tokens"), ("cache_write", "cache_creation_input_tokens"),
                          ("cache_read", "cache_read_input_tokens"), ("output", "output_tokens")):
                a[k2] += u.get(k, 0) or 0
            a["usd_list_equiv"] += s.get("cost_usd_equiv", 0) or 0
    return agg


CLUSTER = r'''
import json, glob, os, collections, subprocess
home = os.path.expanduser("~")
agg = collections.defaultdict(lambda: collections.Counter())
for root, tag in ((home + "/results/mb/evo_b", "eval"), (home + "/results/mb/play", "play")):
    for f in glob.glob(root + "/**/out/**/*.json", recursive=True):
        b = os.path.basename(f)
        if b in ("summary.json", "rejected.json"):
            continue
        try:
            e = json.load(open(f))
        except Exception:
            continue
        if not isinstance(e, dict) or "policy_tokens" not in e:
            continue
        rel = f[len(root) + 1:]
        run = rel.split("/")[0] if tag == "eval" else ("h2" if rel.startswith("harness2") else "ms")
        p = e.get("policy_tokens") or {}
        c = agg[run]
        c["episodes"] += 1
        c["calls"] += p.get("calls", 0) or 0
        c["in_tok"] += p.get("in_tok", 0) or 0
        c["out_tok"] += p.get("out_tok", 0) or 0
        c["episode_seconds"] += e.get("seconds", 0) or 0
out = subprocess.run(["sacct", "-u", os.environ["USER"], "-X", "-n", "-S", "2026-11-26", "-P",
                      "-o", "JobName,ElapsedRaw,AllocTRES"], capture_output=True, text=True).stdout
gpu = collections.Counter()
for line in out.splitlines():
    name, el, tres = (line.split("|") + ["", "", ""])[:3]
    if "gres/gpu" in tres and (name.startswith("mb") or name.startswith("mbe")):
        gpu[name.rstrip("0123456789")] += int(el or 0) / 3600
print(json.dumps({"policy": agg, "gpu_hours": gpu}))
'''


def cluster() -> dict:
    cmd = ("source ~/.mb_site.sh; cd ~/code/multibench && source mb/env.sh >/dev/null 2>&1; python -")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "engaging", cmd], input=CLUSTER, capture_output=True,
                       text=True, timeout=900)
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {}


def walls() -> dict:
    out = {}
    for log in sorted(RUNS.glob("*.log")):
        t = log.read_text(errors="ignore")
        starts = re.findall(r"== \S+ start (\S+Z)", t)
        ends = re.findall(r"== \S+ end (\S+Z)", t)
        if starts:
            last_start = t.rfind(" start ")
            done = ends and t.rfind(" end ") > last_start and "Traceback" not in t[last_start:]
            out[log.stem] = (starts[0], ends[-1] if done else "running")
    return out


def main():
    o, c, w = opus(), cluster(), walls()
    L = ["# Experiment B cost ledger", "",
         "Opus runs on the Claude subscription: dollars are the list-price equivalent, not money spent.", "",
         "## Code model (Opus 5.5)", "",
         "| method | sessions | discarded | input | cache write | cache read | output | list-price $ |",
         "|---|---|---|---|---|---|---|---|"]
    for m, a in sorted(o.items()):
        L.append(f"| {m} | {int(a['sessions'])} | {int(a['sessions_discarded'])} | {int(a['input_tokens']):,} | "
                 f"{int(a['cache_creation_input_tokens']):,} | {int(a['cache_read_input_tokens']):,} | "
                 f"{int(a['output_tokens']):,} | {a['usd_list_equiv']:.2f} |")
    L += ["", "## Policy model (Qwen3.5-9B on ORCD)", "",
          "| run | episodes | calls | input tokens | output tokens | episode-hours |", "|---|---|---|---|---|---|"]
    for run, p in sorted((c.get("policy") or {}).items()):
        L.append(f"| {run} | {p.get('episodes', 0):,} | {p.get('calls', 0):,} | {p.get('in_tok', 0):,} | "
                 f"{p.get('out_tok', 0):,} | {p.get('episode_seconds', 0) / 3600:.1f} |")
    L += ["", "## GPU hours (ORCD, by job family, since 2026-11-26)", ""]
    L += [f"- {k}: {v:.1f}" for k, v in sorted((c.get("gpu_hours") or {}).items())]
    L += ["", "Workers serve every method at once; per-method GPU share ≈ its share of episode-hours above.",
          "Meta-Skills sessions record tokens but no dollar figure. Opus sessions of discarded Harness² and",
          "Meta-Skills work (2026-11-26 rollback) were deleted with that work and are not counted.", "",
          "## Wall clock (UTC)", "", "| run | first start | last end |", "|---|---|---|"]
    L += [f"| {k} | {a} | {b} |" for k, (a, b) in w.items()]
    (RUNS / "LEDGER.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
