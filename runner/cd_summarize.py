"""Circuit Decoding gate table, read against the synthetic reference policies."""
import json, statistics as st, sys
from collections import defaultdict

REF = """synthetic reference (300 released instances, 18-query budget):
  oracle          300/300  11.60 q  eff 1.000  zero-info  0.0%
  greedy-4sample  250/300  13.92 q  eff 0.775  zero-info 18.1%
  random-fresh    156/300  16.51 q  eff 0.629  zero-info 27.7%
  sequential       58/300  17.41 q  eff 0.540  zero-info 38.1%
  random           35/300  17.63 q  eff 0.480  zero-info 44.1%
paper Fig. 5 "Success Rate (%)" on Circuit Dec.: Gemini 2.0 Pro 32.2, Flash 7.2, Flash-lite 4.1; Fig. 1: Gemini 2.5 Pro 39.8."""


def main(path):
    # Retries are appended, not rewritten in place: a (condition, key, seed)
    # that originally errored (e.g. context overflow) gets a second line once
    # it's retried, and the file keeps both. Keep only the LAST line per key
    # -- a dict preserves insertion order and a later line for the same key
    # overwrites the earlier one -- or the stale error would double-count
    # that episode alongside its successful retry.
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
        a["correct"].append(bool(res.get("correct")))
        # MTP Table 1 gives Circuit Decoding one metric, "Normalized circuit-wise
        # accuracy" (normalization undefined). We compute the per-circuit reading
        # (all 8 bits of a circuit right); bit-wise accuracy is our own diagnostic.
        g, t = res.get("guess_bits"), res.get("truth_bits")
        if g and t and len(g) == len(t):
            k = len(t) // 3
            a["bitacc"].append(sum(x == y for x, y in zip(g, t)) / len(t))
            a["ckt"].append(sum(g[i*k:(i+1)*k] == t[i*k:(i+1)*k] for i in range(3)) / 3)
        a["unique"].append(bool(res.get("unique_at_end")))
        a["parsed"].append(bool(res.get("parsed_guess")))
        a["err"].append(bool(r.get("error")))
        a["tok"].append(r.get("out_tokens", 0))
        for q in r["rows"]:
            if q.get("illegal"):
                a["illegal"].append(True); continue
            a["illegal"].append(False)
            if q["n_before"] <= 1:
                continue
            if q["efficiency"] is not None:
                a["eff"].append(q["efficiency"])
            a["zero"].append(q["zero_info"])
            a["repeat"].append(bool(q.get("repeat")))
    hdr = (f"{'condition':<12}{'circuit-wise accuracy':>23}{'joint truth table correct':>27}{'bit-acc':>9}{'uniq@end':>10}{'eff':>8}"
           f"{'zero%':>8}{'repeat%':>9}{'illegal%':>10}{'parsed%':>9}{'tok/ep':>8}")
    print(hdr); print("-" * len(hdr))
    for c, a in agg.items():
        n = len(a["correct"])
        m = lambda k, p=False, nd=3: ("n/a" if not a[k] else
            f"{100*st.mean(a[k]):.1f}%" if p else f"{st.mean(a[k]):.{nd}f}")
        print(f"{c:<12}{m('ckt'):>23}{(str(sum(a['correct'])) + '/' + str(n)):>27}{m('bitacc'):>9}"
              f"{sum(a['unique'])}/{n:<8}"
              f"{m('eff'):>8}{m('zero',1):>8}{m('repeat',1):>9}"
              f"{m('illegal',1):>10}{m('parsed',1):>9}{m('tok',nd=0):>8}")
    print("\n" + REF)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gate_cd.jsonl")
