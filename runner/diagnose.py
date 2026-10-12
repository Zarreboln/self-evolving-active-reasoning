"""What is the model actually doing when it fails?

The summary table says the taught-incremental prompt halves information
efficiency but does NOT produce the Hamming-1 signature I predicted. This
breaks the failure down far enough to name the real behaviour.
"""

import json
import sys
from collections import Counter, defaultdict


def main(path):
    per = defaultdict(lambda: {
        "ham": Counter(), "n_guess": 0, "n_ep": 0,
        "repeat_prev": 0, "repeat_ever": 0, "distinct": [],
        "first_digit_runs": 0, "stuck_runs": [], "dead_and_dull": 0,
        "dull_only": 0, "dead_only": 0, "clean": 0,
    })
    for line in open(path):
        r = json.loads(line)
        a = per[r["condition"]]
        a["n_ep"] += 1
        seen, prev, run = set(), None, 0
        for q in r["rows"]:
            # error rows (e.g. context-length overflow on a long-running
            # episode) carry no guess -- treat like an illegal turn, skip.
            if q.get("error") or q.get("illegal"):
                continue
            g = q["guess"]
            a["n_guess"] += 1
            if q["anchor_dist"] is not None:
                a["ham"][q["anchor_dist"]] += 1
            if prev == g:
                a["repeat_prev"] += 1
                run += 1
            else:
                if run:
                    a["stuck_runs"].append(run + 1)
                run = 0
            if g in seen:
                a["repeat_ever"] += 1
            seen.add(g)
            prev = g
            if q["n_before"] > 1:
                dead = q["dead_guess"]
                dull = (q["efficiency"] or 0) < 0.5
                a["dead_and_dull" if (dead and dull) else
                  "dull_only" if dull else
                  "dead_only" if dead else "clean"] += 1
        if run:
            a["stuck_runs"].append(run + 1)
        a["distinct"].append(len(seen))

    for c, a in per.items():
        n, ng = a["n_ep"], max(a["n_guess"], 1)
        print(f"\n=== {c}  ({n} episodes, {ng} guesses) ===")
        tot = sum(a["ham"].values()) or 1
        print("  与上一次猜测的汉明距离分布:")
        for d in sorted(a["ham"]):
            bar = "#" * round(40 * a["ham"][d] / tot)
            print(f"    {d}: {100*a['ham'][d]/tot:5.1f}%  {bar}")
        print(f"  与上一次完全相同      : {100*a['repeat_prev']/ng:5.1f}%")
        print(f"  猜过又重猜(任意位置)  : {100*a['repeat_ever']/ng:5.1f}%")
        print(f"  每局不同猜测数(均值)  : {sum(a['distinct'])/max(n,1):5.2f}")
        if a["stuck_runs"]:
            print(f"  连续重复的最长串      : {max(a['stuck_runs'])} "
                  f"(均值 {sum(a['stuck_runs'])/len(a['stuck_runs']):.1f}, "
                  f"{len(a['stuck_runs'])} 次卡住)")
        s = a["dead_and_dull"] + a["dull_only"] + a["dead_only"] + a["clean"]
        s = max(s, 1)
        print(f"  动作分类: 不可行且低信息 {100*a['dead_and_dull']/s:.1f}% | "
              f"仅低信息 {100*a['dull_only']/s:.1f}% | "
              f"仅不可行 {100*a['dead_only']/s:.1f}% | "
              f"干净 {100*a['clean']/s:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
