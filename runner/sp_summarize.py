"""SP gate table: bare vs belief_injected, read against run_arbench_sp.py's
zero_shot numbers (same prompts, should match on 'bare' up to sampling noise)."""
import json
import statistics as st
import sys
from collections import defaultdict


def main(path):
    latest = {}
    for line in open(path):
        r = json.loads(line)
        latest[(r["condition"], r["key"], r["seed"])] = r
    dupes = sum(1 for _ in open(path)) - len(latest)
    if dupes:
        print(f"(deduplicated {dupes} retried/duplicate episode records)")

    agg = defaultdict(lambda: defaultdict(list))
    for r in latest.values():
        a = agg[r["condition"]]
        res = r.get("result") or {}
        a["err"].append(bool(r.get("error")))
        if not res:
            continue
        a["f1c"].append(res["f1_char"])
        a["f1w"].append(res["f1_word"])
        a["cov"].append(len(res["confirmed_keypoints"]) / res["n_keypoints"])
        a["in_tok"].append(r.get("in_tokens", 0))
        a["out_tok"].append(r.get("out_tokens", 0))

    hdr = (f"{'condition':<18}{'n':>5}{'err%':>7}{'f1_char':>9}{'f1_word':>9}"
           f"{'kp_cov':>8}{'in_tok':>9}{'out_tok':>8}")
    print(hdr); print("-" * len(hdr))
    for c, a in agg.items():
        n = len(a["err"])
        m = lambda k, nd=3: ("n/a" if not a[k] else f"{st.mean(a[k]):.{nd}f}")
        print(f"{c:<18}{n:>5}{100*st.mean(a['err']):>6.1f}%{m('f1c'):>9}{m('f1w'):>9}"
              f"{m('cov'):>8}{m('in_tok',0):>9}{m('out_tok',0):>8}")
    print("\nreference (run_arbench_sp.py, official zero_shot path, same prompts):")
    print("  bare  100 episodes  f1_char=0.659 f1_word=0.242 (100-case run, 2026-10-07)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gate_sp.jsonl")
