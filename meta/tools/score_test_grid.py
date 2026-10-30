"""Score every AR-Bench published arm already on disk, on the TEST split.

run_sharded.sh pins DATA=$HOME/code/AR-Bench/data/$TASK/test.json and make_ar_splits.py defines
test as exactly that file, untouched -- so ~/results/<task>_<method>.json is already a test-split
run. Nothing is re-run here; this reads and scores.

Protocol, verified identical to the one our evolved harnesses were measured under:
  policy      Qwen2.5-7B-Instruct frozen, temperature 0.7 / top_p 0.7, 25 turns
  gn feedback pure function, no second model either side
  dc suspects the same served 7B at 0.7/0.7 (DC_RESPONDER=local, dc/env.py)
  sp referee  the jev shim on :8099, labelled jev-latest, 0.7/0.7, both sides
  sp scoring  sp_semantic_equivalence.py at JEV_MODEL=jev-1.14.0, both sides
  sp metric   (equivalent + 0.5 * partial) / n, both sides

Three evaluators recorded tokens under three different key pairs, and DC's three prompt baselines
recorded neither side until dc_backfill_tokens.py and dc_backfill_output.py recovered both from
`record`. All of them are input+output per episode here, which is what RUN_LOG states the cost
axis is.
"""
import json
import os

R = os.path.expanduser("~/results")
M = {"entailment": 2, "neutral": 1, "contradiction": 0}
METHODS = ["zero_shot", "few_shot", "few_shot_inst", "proactive_cot", "tot"]
EXTRA = {"gn": ["tot_fixed"], "dc": ["tot_eval", "proactive_stop"],
         "sp": ["tot_eval", "proactive_stop", "uot"]}
PAIRS = [("input_token", "output_token"),
         ("input_token_sum", "output_token_sum"),
         ("input_token_backfilled", "output_token_backfilled")]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def tokens(rows):
    per, basis = [], set()
    for r in rows:
        for a, b in PAIRS:
            if r.get(a) is not None:
                per.append((r.get(a) or 0) + (r.get(b) or 0))
                basis.add("in+out" if r.get(b) is not None else "IN ONLY")
                break
        else:
            basis.add("MISSING")
    return (sum(per) / len(per) if per else None), "/".join(sorted(basis))


EQ = (load(R + "/sp_equiv_jev.json") or {"runs": {}})["runs"]


def sp_score(fname):
    js = EQ.get(fname)
    if not js:
        return None, None
    t = {"equivalent": 0, "partial": 0, "wrong": 0}
    for j in js:
        l, r = M.get(j.get("left2right")), M.get(j.get("right2left"))
        if l is None or r is None or 0 in (l, r):
            t["wrong"] += 1
        elif l == 2 and r == 2:
            t["equivalent"] += 1
        elif 2 in (l, r):
            t["partial"] += 1
        else:
            t["wrong"] += 1
    n = len(js)
    return 100.0 * (t["equivalent"] + 0.5 * t["partial"]) / n, t


OURS = {   # finalization 2026-10-27, logs/finalize_run1b.out
    "gn": [("packed_group_echo", 100.0, 7), ("baseline_no_questions", 0.0, 226)],
    "dc": [("lean_verify", 63.0, 25715), ("prior_verify_stop", 57.0, 18308),
           ("baseline_no_questions", 36.0, 847)],
    "sp": [("atomic_audit", 32.5, 16142), ("family_answer", 31.5, 10913),
           ("coarse_first", 30.5, 14183), ("baseline_no_questions", 10.5, 215)],
}

out = {}
for task in ("gn", "dc", "sp"):
    print("=" * 80)
    print("%s   TEST split, n=100, frozen Qwen2.5-7B-Instruct, 25 turns" % task.upper())
    print("=" * 80)
    print("  %-24s %-10s %7s %11s  %s" % ("system", "kind", "acc", "tok/ep", "tokens"))
    rows = []
    for m in METHODS + EXTRA[task]:
        f = "%s_%s.json" % (task, m)
        ep = load(os.path.join(R, f))
        if ep is None:
            # sp/uot ran on the retired h100; the episode file is in the Baidu backup, but its jev
            # judgments survived in sp_equiv_jev.json, so the accuracy is recomputable under this
            # tables metric. RUN_LOGs 5.0 for it is the non-strict tier count, not this metric.
            acc, t = sp_score(f) if task == "sp" else (None, None)
            if acc is None:
                rows.append((m, "AR-Bench", None, None, "no run on disk"))
            else:
                rows.append((m, "AR-Bench", acc, None,
                             "eq=%d part=%d  cost: episode file not on this host" % (t["equivalent"], t["partial"])))
                out["%s/%s" % (task, m)] = {"accuracy": acc, "tokens": None, "n": len(EQ[f])}
            continue
        tk, basis = tokens(ep)
        if task == "sp":
            acc, t = sp_score(f)
            note = "%s  eq=%d part=%d" % (basis, t["equivalent"], t["partial"]) if acc is not None else "no jev judgment"
        else:
            acc = 100.0 * sum(1 for r in ep if r.get("correctness")) / len(ep)
            note = basis
        rows.append((m, "AR-Bench", acc, tk, note))
        if acc is not None:
            out["%s/%s" % (task, m)] = {"accuracy": acc, "tokens": tk, "n": len(ep)}
    for name, acc, tk in OURS[task]:
        rows.append((name, "anchor" if "no_questions" in name else "OURS", acc, tk, "in+out"))
    rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))
    for name, kind, acc, tk, note in rows:
        print("  %-24s %-10s %7s %11s  %s"
              % (name, kind,
                 "--" if acc is None else "%.1f" % acc,
                 "--" if tk is None else format(round(tk), ","), note))
    print()
json.dump(out, open(os.path.expanduser("~/results/test_grid_scored.json"), "w"), indent=1)
print("-> ~/results/test_grid_scored.json")
