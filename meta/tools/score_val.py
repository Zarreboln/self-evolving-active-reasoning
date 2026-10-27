"""Turn a finished run into the val.json Meta-Harness's loop reads.

`load_results` walks `<logs>/<task>/<system>/<model>/val.json`. What goes in each field:

    accuracy              non-strict semantic equivalence for sp, correctness for gn and dc.
                          Measured: the same harness re-run on the same 50 puzzles scored 35/50
                          non-strict both times while every per-episode prediction differed, and
                          strict moved 12 -> 17 over that same pair. The stable one is what the
                          frontier is computed from; the other is recorded beside it.
    memory_context_tokens mean input tokens per episode -- the cost axis.
    strict, partial_share sentinels (sp only). A candidate whose non-strict rises while strict
                          stays flat and partial_share rises has learned to say the safest
                          fragment. partial_share is the share of non-strict passes where the gold
                          entails the prediction but not the reverse: the prediction said LESS.

    POLICY_BASE_URL=... python score_val.py <run.json> --task sp --system few_shot_inst
"""
from __future__ import annotations

import argparse, json, os, statistics, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/code"))
import sp_semantic_equivalence as SEQ  # noqa: E402

MODEL = os.environ.get("POLICY_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def mean_tokens(rows):
    """Input PLUS output, per episode.

    Input alone hides where the cost is. On Guess Number the spread between the published
    baselines is 8.6x by input and 38x by input plus output: the expensive ones pay on the
    generated side, having the model write its state out turn after turn, and decoding is the
    slow half. Total is what a run actually costs.
    """
    tot = []
    for r in rows:
        i = r.get("input_token") or r.get("input_token_sum") or r.get("input_token_backfilled") or 0
        o = r.get("output_token") or r.get("output_token_sum") or 0
        tot.append(i + o)
    return statistics.mean(tot) if any(tot) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", required=True, choices=("gn", "sp", "dc"))
    ap.add_argument("--system", required=True)
    ap.add_argument("--logs-dir", default=os.path.expanduser("~/results/mh/logs"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge", choices=("jev", "policy"), default="jev",
                    help="jev by default: on the selection split's gold texts the policy judge "
                         "calls 4 of 10 not-equivalent-to-themselves, deterministically, while "
                         "jev calls 10 of 10 equivalent")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    rows = json.load(open(a.run))
    out = {"seed": a.seed, "n": len(rows), "judge": a.judge,
           "llm_calls": sum(r.get("llm_calls", 0) for r in rows),
           "total_tokens": round(mean_tokens(rows), 1)}

    if a.task == "sp":
        if a.judge == "jev":
            tok = os.environ.get("TYPESAFE_API_KEY") or \
                open(os.path.expanduser("~/.typesafe_token")).read().strip()
            judge = lambda p: SEQ.ask_jev(p, tok)                      # noqa: E731
        else:
            base = os.environ["POLICY_BASE_URL"]
            key = os.environ.get("POLICY_API_KEY", "x")
            judge = lambda p: SEQ.ask_policy(base, key, a.model, p, 0.0)   # noqa: E731
        ok, why = SEQ.judge_is_usable(judge, rows)
        print("  judge self-check: %s -- %s" % ("PASS" if ok else "FAIL", why))
        if not ok and not a.force:
            raise SystemExit("refusing to score with this judge; --force to override and label it")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            js = list(ex.map(lambda r: SEQ.score_episode(r, judge), rows))
        # Three tiers out of the two directional judgments, not a new call. The first two
        # together are exactly BALAR's non-strict equivalence; splitting them is what gives the
        # axis any range, because with an independent judge the top tier is near-empty: a frozen
        # 7B almost never writes an explanation that entails the gold AND is entailed by it.
        M = {"entailment": 2, "neutral": 1, "contradiction": 0}
        tiers = {"equivalent": 0, "partial": 0, "wrong": 0}
        for j in js:
            l, r = M.get(j.get("left2right")), M.get(j.get("right2left"))
            if l is None or r is None or 0 in (l, r):
                tiers["wrong"] += 1
            elif l == 2 and r == 2:
                tiers["equivalent"] += 1
            elif 2 in (l, r):
                tiers["partial"] += 1
            else:
                tiers["wrong"] += 1
        n = len(js)
        out["tiers"] = tiers
        out["accuracy"] = (tiers["equivalent"] + 0.5 * tiers["partial"]) / n
        out["equivalent"] = tiers["equivalent"] / n
        out["partial"] = tiers["partial"] / n
        out["nonstrict"] = (tiers["equivalent"] + tiers["partial"]) / n   # BALAR's, for the table
        out["judge_failed"] = sum(1 for j in js if j.get("judge_failed"))
    else:
        out["accuracy"] = sum(1 for r in rows if r.get("correctness")) / len(rows)

    d = os.path.join(a.logs_dir, a.task, a.system, a.model.split("/")[-1])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "val.json"), "w") as f:
        json.dump(out, f, indent=1)
    extra = ("  equiv=%.0f%% partial=%.0f%% wrong=%.0f%%  (non-strict=%.0f%%)"
             % (100 * out["equivalent"], 100 * out["partial"],
                100 * out["tiers"]["wrong"] / out["n"], 100 * out["nonstrict"])) \
        if a.task == "sp" else ""
    print("%-10s %-18s accuracy=%.3f  tot-tok/ep=%-9.0f%s"
          % (a.task, a.system, out["accuracy"], out["total_tokens"], extra))


if __name__ == "__main__":
    main()
