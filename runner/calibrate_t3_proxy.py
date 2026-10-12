"""Calibrate T3's CircuitDecoding proxy condition against our exact ground truth.

T3 (arXiv 2510.12264, Definition 2) detects likely entry into the belief trap
region with task-specific heuristics because the true belief trap is defined
on the agent's internal belief, which is not observable. For CircuitDecoding
their stated proxy is: truncate if the hypothesis space fails to shrink for 3
consecutive turns. This is checkable offline from already-logged trajectories
(gate_cd.jsonl) -- no new model calls -- because we log n_before/n_after
(exact feasible-set size) and zero_info (exact ex-ante efficiency < 1e-9) on
every turn.

"Ground truth" here is the per-turn zero_info flag: the agent's actual choice
carried zero expected information given the exact feasible set at that turn.
T3's proxy is coarser than this (it requires 3 consecutive non-shrinking
turns, not a single zero-info turn), so recall should be expected to lag
genuinely zero-info turns that happen in isolation, and the interesting
number is how much.
"""

import json
import sys
from collections import defaultdict


def dedup(path):
    """Same rule as cd_summarize.py: keep only the latest record per
    (condition, key, seed) -- retried episodes are appended, not rewritten."""
    latest = {}
    for line in open(path):
        r = json.loads(line)
        latest[(r["condition"], r["key"], r["seed"])] = r
    return list(latest.values())


def t3_flags(rows, window=3):
    """Turn indices (0-based within this episode's legal-turn sequence) where
    T3's proxy would fire: n_before has been unchanged for `window` consecutive
    turns, i.e. this turn and the (window-1) turns before it all failed to
    shrink the feasible set."""
    legal = [r for r in rows if not r.get("illegal") and not r.get("error")]
    sizes = [r["n_before"] for r in legal]
    flags = [False] * len(legal)
    for i in range(window - 1, len(legal)):
        if len(set(sizes[i - window + 1:i + 1])) == 1:
            flags[i] = True
    return legal, flags


def main(path, window=3):
    recs = dedup(path)
    tp = fp = fn = tn = 0
    per_cond = defaultdict(lambda: [0, 0, 0, 0])  # tp fp fn tn
    for r in recs:
        legal, flags = t3_flags(r["rows"], window)
        for turn, flagged in zip(legal, flags):
            actual = bool(turn.get("zero_info"))
            c = per_cond[r["condition"]]
            if flagged and actual:
                tp += 1; c[0] += 1
            elif flagged and not actual:
                fp += 1; c[1] += 1
            elif not flagged and actual:
                fn += 1; c[2] += 1
            else:
                tn += 1; c[3] += 1

    def prf(tp, fp, fn, tn):
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) else float("nan")
        return p, r, f1

    print(f"T3 proxy (hypothesis space unchanged for {window} consecutive "
          f"turns) vs. exact zero-info ground truth, on {path}\n")
    print(f"{'condition':<20}{'tp':>7}{'fp':>7}{'fn':>7}{'tn':>7}"
          f"{'precision':>11}{'recall':>9}{'f1':>7}")
    print("-" * 75)
    for c, (ctp, cfp, cfn, ctn) in per_cond.items():
        p, r, f1 = prf(ctp, cfp, cfn, ctn)
        print(f"{c:<20}{ctp:>7}{cfp:>7}{cfn:>7}{ctn:>7}"
              f"{p:>11.3f}{r:>9.3f}{f1:>7.3f}")
    p, r, f1 = prf(tp, fp, fn, tn)
    print("-" * 75)
    print(f"{'ALL':<20}{tp:>7}{fp:>7}{fn:>7}{tn:>7}{p:>11.3f}{r:>9.3f}{f1:>7.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gate_cd.jsonl",
         int(sys.argv[2]) if len(sys.argv) > 2 else 3)
