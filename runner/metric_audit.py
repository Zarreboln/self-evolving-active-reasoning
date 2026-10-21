"""What does AR-Bench's Situation Puzzles metric actually measure?

Character-level F1 compares the model's answer to the hidden bottom. It has no length penalty
that bites, so a longer answer overlaps more of the reference almost mechanically. This script
quantifies that on our own frozen-evaluation records: the correlation of F1 with answer length,
with our process metric (key-question recall), the variance each explains, and the gap between
two harnesses once answer length is held fixed.

    python runner/metric_audit.py meta/cd_tthe/logs/eval_sp100 meta/cd_tthe/logs/eval_sp100_b1r0g0 \
        --exclude-harness cand_sp_strict1_b10g3r2_b1r0g0 --baseline bare --evolved cand_sp_strict1_b10g3r2_b2r1g0

Episodes whose status is not ok, or whose answer is empty, are dropped: an interrupted run leaves
zero-word answers at zero F1, and those points sit exactly on the length axis and inflate the
correlation. (They did: 0.90 before dropping them, 0.81 after.)
"""

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import numpy as np


def load(dirs, exclude):
    rows, dropped = [], 0
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "results" / "*.jsonl"))):
            name = Path(f).stem
            if name in exclude:
                continue
            for line in open(f):
                r = json.loads(line)
                g = r.get("gold") or {}
                if r.get("status") != "ok" or g.get("f1_char") is None:
                    dropped += 1
                    continue
                w = len((g.get("prediction") or "").split())
                if w == 0:
                    dropped += 1
                    continue
                rows.append({"h": name, "i": r["index"], "f1": g["f1_char"],
                             "rec": g.get("keypoint_recall"), "w": w,
                             "tok": (r.get("tokens") or {}).get("out", 0)})
    return rows, dropped


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--exclude-harness", nargs="*", default=[])
    p.add_argument("--baseline", default="bare")
    p.add_argument("--evolved", default=None)
    a = p.parse_args()

    rows, dropped = load(a.dirs, set(a.exclude_harness))
    y = np.array([r["f1"] for r in rows])
    w = np.array([r["w"] for r in rows], float)
    rc = np.array([r["rec"] for r in rows], float)
    print(f"{len(rows)} episodes kept, {dropped} dropped (crashed or empty answer)\n")

    print("correlation with character-level F1")
    print(f"  answer length in words   r = {np.corrcoef(y, w)[0, 1]:+.3f}")
    print(f"  key-question recall      r = {np.corrcoef(y, rc)[0, 1]:+.3f}")
    print(f"  output tokens            r = {np.corrcoef(y, np.array([r['tok'] for r in rows], float))[0, 1]:+.3f}")

    X1 = np.column_stack([np.ones(len(y)), w])
    b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    X2 = np.column_stack([np.ones(len(y)), w, rc])
    b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    print("\nvariance explained")
    print(f"  length alone             R2 = {r2(y, X1 @ b1):.3f}")
    print(f"  length + recall          R2 = {r2(y, X2 @ b2):.3f}   (recall coefficient {b2[2]:+.3f})")
    br = np.linalg.lstsq(X1, rc, rcond=None)[0]
    print(f"  recall vs F1, length partialled out:  r = {np.corrcoef(rc - X1 @ br, y - X1 @ b1)[0, 1]:+.3f}")

    print("\nmean F1 by answer length")
    for k in range(8):
        lo, hi = k * 25, (k + 1) * 25
        s = [r["f1"] for r in rows if lo <= r["w"] < hi]
        if len(s) >= 10:
            print(f"  {lo:>3}-{hi:<4} words  n={len(s):>3}  F1={st.mean(s):.3f}")

    if a.evolved:
        print(f"\n{a.baseline} vs {a.evolved}, within the same length band")
        print(f"  {'band':>10} {'n base':>7} {'F1 base':>8} {'n evo':>6} {'F1 evo':>7} {'gap':>7}")
        for lo, hi in [(25, 50), (50, 75), (75, 100), (100, 125), (125, 150)]:
            b = [r["f1"] for r in rows if r["h"] == a.baseline and lo <= r["w"] < hi]
            e = [r["f1"] for r in rows if r["h"] == a.evolved and lo <= r["w"] < hi]
            if len(b) >= 5 and len(e) >= 5:
                print(f"  {f'{lo}-{hi}':>10} {len(b):>7} {st.mean(b):>8.3f} {len(e):>6} "
                      f"{st.mean(e):>7.3f} {st.mean(e) - st.mean(b):>+7.3f}")
        allb = [r["f1"] for r in rows if r["h"] == a.baseline]
        alle = [r["f1"] for r in rows if r["h"] == a.evolved]
        print(f"  {'pooled':>10} {len(allb):>7} {st.mean(allb):>8.3f} {len(alle):>6} "
              f"{st.mean(alle):>7.3f} {st.mean(alle) - st.mean(allb):>+7.3f}   <- the headline gap")


if __name__ == "__main__":
    main()
