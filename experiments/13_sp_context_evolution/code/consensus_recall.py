"""Key-point recall of saved TEST episodes under three judge definitions, from cached judge replies only:
  haiku     - main judge, per-question (as scored)
  gemini    - robustness judge, batched (as rescored)
  consensus - a key point counts only if BOTH judges credit it on the same question (per-question intersection)
Per arm: mean over puzzles of the seed-averaged recall; paired bootstrap CI vs a0 and vs a2.
Only arm files with a complete Gemini rescore are used. Writes runs/judge_gap/consensus_report.md.
Usage: python -I code/consensus_recall.py"""
import json
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import evaluate  # noqa: E402
import rescore_test as rt  # noqa: E402
from llm import spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "runs" / "test"
OUT = ROOT / "runs" / "judge_gap"
KINDS = ("haiku", "gemini", "consensus")


def episode(r):
    p = rt.puz[r["episode"]["id"]]
    qs = [sp.clean_q(q) for q, _ in r["episode"]["history"]]
    gb = rt.batched_hits(p, qs)
    if gb is None:
        return None
    hb = [sp.key_hits(p, q) for q in qs]
    n = len(sp.keypoints_of(p))
    return {"haiku": len(set().union(*hb)) / n, "gemini": len(set().union(*gb)) / n,
            "consensus": len(set().union(*[a & b for a, b in zip(hb, gb)])) / n}


def main():
    done = {p.stem for p in rt.OUT.glob("*_s*.json")
            if all(v is not None for v in json.loads(p.read_text(encoding="utf-8")).values())}
    arms = sorted({s.rsplit("_s", 1)[0] for s in done})
    data = {}
    for a in arms:
        per = {}
        for stem in sorted(s for s in done if re.fullmatch(rf"{re.escape(a)}_s\d+", s)):
            for line in (TEST / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                e = episode(r)
                if e:
                    per.setdefault(r["episode"]["id"], []).append(e)
        data[a] = {pid: {k: mean(e[k] for e in v) for k in KINDS} | {"n": len(v)} for pid, v in per.items()}
    lines = ["| arm | seeds | haiku | gemini | consensus (both) |", "|---|---|---|---|---|"]
    for a in arms:
        d = data[a]
        lines.append(f"| {a} | {max(v['n'] for v in d.values())} | " +
                     " | ".join(f"{mean(v[k] for v in d.values()):.3f}" for k in KINDS) + " |")
    for base in ("a0", "a2"):
        if base not in data:
            continue
        lines += ["", f"Paired difference vs {base} (95% bootstrap CI over puzzles):", ""]
        for a in arms:
            if a == base:
                continue
            common = sorted(set(data[a]) & set(data[base]))
            cells = []
            for k in KINDS:
                m, lo, hi = evaluate.boot([data[a][p][k] - data[base][p][k] for p in common])
                cells.append(f"{k} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            lines.append(f"- {a} - {base} (n={len(common)}): " + "; ".join(cells))
    txt = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "consensus_report.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    b = spent_usd()
    main()
    print(f"\nnew spend: ${spent_usd() - b:.3f}")
