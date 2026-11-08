"""Strict-judge (Gemini) score of the official AR-Bench SP metric on saved TEST episodes: key points matched by the
7B's final story (0-6), AR-Bench guess_eval prompt (probe_use.gem_match). Dialogues and stories are unchanged.
A call that fails or does not parse is retried, then stored as None (MISSING), never as 0.
Writes runs/test/story_gemini/<arm>_s<seed>.json.  Usage: python -I code/story_test.py arm [arm ...] | --report"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import probe_use  # noqa: E402
from llm import spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "runs" / "test"
OUT = TEST / "story_gemini"
PUZ = {p["_id"]: p for p in sp.load("test")}


def one(r):
    p = PUZ[r["episode"]["id"]]
    pred = r["episode"].get("pred") or ""
    if not pred.strip():
        return p["_id"], None
    for attempt in range(4):
        v = probe_use.gem_match(p, pred)
        if v is not None:
            return p["_id"], v
        time.sleep(5 * (attempt + 1))
    return p["_id"], None


def run(arm):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(TEST.glob(f"{arm}_s*.jsonl")):
        if f.stem.rsplit("_s", 1)[0] != arm:
            continue
        out_f = OUT / f"{f.stem}.json"
        old = json.loads(out_f.read_text(encoding="utf-8")) if out_f.exists() else {}
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
        todo = [r for r in rows if old.get(r["episode"]["id"]) is None]
        with ThreadPoolExecutor(8) as ex:
            res = dict(ex.map(one, todo))
        old.update(res)
        out_f.write_text(json.dumps(old, indent=1), encoding="utf-8")
        ok = [v for v in old.values() if v is not None]
        print(f"{f.stem}: gemini story {mean(ok):.3f} (n={len(ok)}, missing {len(old) - len(ok)}) "
              f"spent ${spent_usd():.2f}", flush=True)


def load(arm):
    d = {}
    for f in OUT.glob(f"{arm}_s*.json"):
        if f.stem.rsplit("_s", 1)[0] != arm:
            continue
        s = int(f.stem.rsplit("_s", 1)[1])
        for pid, v in json.loads(f.read_text(encoding="utf-8")).items():
            if v is not None:
                d.setdefault(pid, {})[s] = v
    return d


def boot(diffs, b=10_000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    ms = sorted(mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(b))
    return mean(diffs), ms[int(.025 * b)], ms[int(.975 * b)]


def report():
    arms = sorted({f.stem.rsplit("_s", 1)[0] for f in OUT.glob("*.json")})
    data = {a: load(a) for a in arms}
    lines = ["Final-story key points matched (0-6), Gemini strict judge, TEST 100; paired by puzzle on common seeds.", ""]
    for a in arms:
        lines.append(f"- {a}: mean {mean(mean(v.values()) for v in data[a].values()):.3f} (puzzles {len(data[a])})")
    lines.append("")
    for a in arms:
        for b in ("a0", "r1"):
            if a == b or b not in data:
                continue
            diffs = []
            for pid in set(data[a]) & set(data[b]):
                ss = set(data[a][pid]) & set(data[b][pid])
                if ss:
                    diffs.append(mean(data[a][pid][s] for s in ss) - mean(data[b][pid][s] for s in ss))
            if diffs:
                m, lo, hi = boot(diffs)
                lines.append(f"- {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(diffs)})"
                             + (" SIG" if lo > 0 or hi < 0 else ""))
    txt = "\n".join(lines)
    (OUT / "REPORT.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report()
    else:
        for a in sys.argv[1:]:
            run(a)
        report()
