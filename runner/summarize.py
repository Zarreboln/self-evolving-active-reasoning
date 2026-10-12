"""Read a runner JSONL and print the gate-step table.

Reference points measured on synthetic agents (stage0/probe_synthetic.py,
2026-10-06, 40 instances, seed 0):

    oracle      eff 1.000   dead 15.5%   ham1  ~0%    solved 100%
    consistent  eff 0.936   dead  0.0%   ham1  ~4%    solved 100%
    anchored    eff 0.265   dead 96.0%   ham1 79.2%   solved   0%

Read the real model against those. The anchored row is the hypothesis under
test: it is the strategy the AR-Bench 1-shot-instruction prompt teaches.
"""

import json
import statistics as st
import sys
from collections import defaultdict


def main(path):
    agg = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = agg[r["condition"]]
        a["solved"].append(bool(r["solved"]))
        if r["solved"]:
            a["turns"].append(r["turns_to_solve"])
        a["out_tokens"].append(r.get("out_tokens", 0))
        for q in r["rows"]:
            if q.get("error"):          # e.g. context-length overflow on a
                a["illegal"].append(True)  # long-running episode; too rare to
                continue                    # separate from "illegal" in the table
            if q.get("illegal"):
                a["illegal"].append(True)
                continue
            a["illegal"].append(False)
            if q["efficiency"] is not None:
                a["eff"].append(q["efficiency"])
            a["dead"].append(q["dead_guess"])
            if q["anchor_dist"] is not None:
                a["ham1"].append(q["anchor_dist"] == 1)
            # the pathology that matters: infeasible AND uninformative
            if q["efficiency"] is not None:
                a["dead_and_dull"].append(
                    q["dead_guess"] and q["efficiency"] < 0.5)

    def m(v, pct=False, nd=3):
        if not v:
            return "n/a"
        x = st.mean(v)
        return f"{100*x:.1f}%" if pct else f"{x:.{nd}f}"

    cols = f"{'condition':<20}{'solved':>9}{'turns':>7}{'eff':>8}{'dead%':>8}" \
           f"{'ham1%':>8}{'d&dull%':>9}{'illegal%':>9}{'tok/ep':>8}"
    print(cols)
    print("-" * len(cols))
    for c, a in agg.items():
        n = len(a["solved"])
        print(f"{c:<20}{sum(a['solved'])}/{n:<7}"
              f"{m(a['turns'], nd=2):>7}{m(a['eff']):>8}{m(a['dead'], True):>8}"
              f"{m(a['ham1'], True):>8}{m(a['dead_and_dull'], True):>9}"
              f"{m(a['illegal'], True):>9}{m(a['out_tokens'], nd=0):>8}")
    print("\nsynthetic reference -- oracle eff 1.000 dead 15.5% ham1 ~0% | "
          "consistent 0.936 / 0.0% / 4% | anchored 0.265 / 96.0% / 79.2%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gate_gn.jsonl")
