"""Print one CircuitDecoding episode from a traces jsonl in readable form.

    python tools/show_trace.py logs/<run>/traces/<candidate>/search.jsonl --key 17 [--seed 0]
    python tools/show_trace.py <file> --list            # one line per episode: key seed uniq joint_table n_alive_end eff
    python tools/show_trace.py <file> --worst 5         # the 5 episodes with the most zero-information queries
"""

import argparse
import json
from pathlib import Path

LABELS = "ABCDEFGH"


def fmt(q, k=3):
    lb = LABELS[q // (1 << k)]
    row = q % (1 << k)
    return f"{lb}({','.join(str((row >> (k - 1 - i)) & 1) for i in range(k))})"


def load(path):
    latest = {}
    for line in Path(path).open():
        if line.strip():
            r = json.loads(line)
            latest[(r["key"], r["seed"])] = r
    return latest


def summary_line(r):
    res = r.get("result") or {}
    rows = r.get("rows") or []
    legal = [x for x in rows if not x.get("illegal") and x.get("n_before", 0) > 1]
    eff = sum(x["efficiency"] for x in legal if x.get("efficiency") is not None) / max(1, len(legal))
    zero = sum(1 for x in legal if x.get("zero_info"))
    rep = sum(1 for x in rows if x.get("repeat"))
    ill = sum(1 for x in rows if x.get("illegal"))
    return (f"key={r['key']:<4} seed={r['seed']} uniq={int(bool(res.get('unique_at_end')))} "
            f"joint_table={int(bool(res.get('correct')))} alive_end={res.get('n_alive_end')!s:<5} "
            f"eff={eff:.3f} zero={zero:<2} repeat={rep:<2} illegal={ill:<2} err={bool(r.get('error'))}")


def show(r):
    print(summary_line(r))
    print("=" * 100)
    for x in r.get("rows") or []:
        if x.get("illegal"):
            print(f"[turn {x['turn']}] ILLEGAL (no query parsed)")
        else:
            q = fmt(x["query"])
            best = fmt(x["best_query"]) if x.get("best_query") is not None else "-"
            print(f"[turn {x['turn']}] query {q} -> {x['answer']}   alive {x['n_before']}->{x['n_after']}   "
                  f"bits {x['bits_chosen']:.3f}/{x['bits_available']:.3f}   best {best}"
                  f"{'   ZERO-INFO' if x.get('zero_info') else ''}{'   REPEAT' if x.get('repeat') else ''}")
        print("  MODEL >>", (x.get("model_text") or "").strip().replace("\n", "\n           "))
        if x.get("sent"):
            print("  SENT  <<", x["sent"].strip().replace("\n", "\n           "))
        print("-" * 100)
    print("FINAL SENT <<", (r.get("final_sent") or "").strip().replace("\n", "\n              "))
    print("FINAL MODEL >>", (r.get("final_text") or "").strip().replace("\n", "\n               "))
    res = r.get("result") or {}
    print("truth  ", "".join(map(str, res.get("truth_bits") or [])))
    print("guess  ", "".join(map(str, res.get("guess_bits") or [])) or "(unparsed)")
    if r.get("error"):
        print("ERROR  ", r["error"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--key", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true")
    p.add_argument("--worst", type=int, default=0)
    a = p.parse_args()
    eps = load(a.file)
    if a.list:
        for k in sorted(eps):
            print(summary_line(eps[k]))
        return
    if a.worst:
        def nzero(r):
            return sum(1 for x in (r.get("rows") or []) if x.get("zero_info"))
        for r in sorted(eps.values(), key=nzero, reverse=True)[: a.worst]:
            show(r)
            print("\n\n")
        return
    if a.key is None:
        p.error("--key N (or --list / --worst N)")
    show(eps[(a.key, a.seed)])


if __name__ == "__main__":
    main()
