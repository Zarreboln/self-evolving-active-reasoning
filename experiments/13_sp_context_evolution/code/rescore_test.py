"""Re-score saved TEST episodes with a second, cross-family key-point judge (sp.ROBUST_JUDGE = gemini-3.7-flash).

Batched protocol: one call per episode lists the episode's questions (numbered) and asks for the key points hit by
each, using AR-Bench's keypoint_hits rules and examples (the per-question protocol hit Gemini's rate limit: 85/469
calls returned HTTP 429 on 2026-11-07). A call that fails or does not parse is retried; an episode that still
fails is reported as MISSING, never as "no hit". `--agree` measures batched-vs-per-question agreement on a sample.
Dialogues are unchanged. Writes runs/test/rescore_<judge>/<arm>_s<seed>.json.
Usage: python -I code/rescore_test.py arm [arm ...] | --report | --agree N"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import evaluate  # noqa: E402
from llm import chat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "runs" / "test"
J = sp.ROBUST_JUDGE
OUT = TEST / f"rescore_{J.replace('/', '_')}_batched"
puz = {p["_id"]: p for p in sp.load("test")}
THREADS = 6


def batched_hits(p, questions):
    """Return list of hit-sets (one per question) or None on failure."""
    listing = "\n".join(f"Q{i + 1}: {q}" for i, q in enumerate(questions))
    prompt = sp.KEYPOINT_PROMPT.format(question=p["surface"], answer=p["bottom"],
                                       keypoints=sp.place_keypoints(sp.keypoints_of(p))) + \
        (f"\n\nJudge EACH player question below separately, by the rules above.\n{listing}\n\n"
         f"Answer with exactly one line per question, in order, formatted 'Q<i>: Hit point: x, y' or "
         f"'Q<i>: No point is hit'.")
    n = len(sp.keypoints_of(p))
    for attempt in range(6):
        txt = chat(J, [{"role": "user", "content": prompt}], role="robustjudge", temperature=0.0, max_tokens=6000,
                   tag=p["_id"] + f"|try{attempt}" if attempt else p["_id"], use_cache=(attempt == 0))
        out = {}
        for line in (txt or "").splitlines():
            m = re.match(r"\s*\**Q(\d+)\**\s*:\s*(.*)", line)
            if m:
                i, rest = int(m.group(1)), m.group(2)
                h = re.search(r"Hit point:\s*([\d,\s]+)", rest)
                out[i] = ({int(x) for x in re.findall(r"\d+", h.group(1))} & set(range(1, n + 1))) if h else set()
        if len(out) == len(questions) and all(i + 1 in out for i in range(len(questions))):
            return [out[i + 1] for i in range(len(questions))]
        time.sleep(5 * (attempt + 1))
    return None


def episode_recall(r):
    p = puz[r["episode"]["id"]]
    qs = [sp.clean_q(q) for q, _ in r["episode"]["history"]]
    hits = batched_hits(p, qs)
    if hits is None:
        return r["episode"]["id"], None
    return r["episode"]["id"], len(set().union(*hits)) / len(sp.keypoints_of(p))


def arm_files(arm):
    return sorted(p for p in TEST.glob(f"{arm}_s*.jsonl") if re.fullmatch(rf"{re.escape(arm)}_s\d+\.jsonl", p.name))


def rescore(arm):
    for f in arm_files(arm):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
        with ThreadPoolExecutor(THREADS) as ex:
            res = dict(ex.map(episode_recall, rows))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f.name.replace(".jsonl", ".json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
        ok = [v for v in res.values() if v is not None]
        print(f"{f.stem}: {J} (batched) recall {mean(ok):.3f} over {len(ok)}/{len(res)} episodes "
              f"(haiku {mean(r['score']['key_recall'] for r in rows):.3f})", flush=True)


def agree(n):
    """Batched vs per-question Gemini on the questions of the first n a0_s0 episodes."""
    rows = [json.loads(x) for x in (TEST / "a0_s0.jsonl").read_text(encoding="utf-8").splitlines()][:n]
    same = tot = fail = 0
    for r in rows:
        p = puz[r["episode"]["id"]]
        qs = [sp.clean_q(q) for q, _ in r["episode"]["history"]]
        b = batched_hits(p, qs)
        if b is None:
            fail += 1
            continue
        for q, hb in zip(qs, b):
            for attempt in range(6):
                txt = chat(J, [{"role": "user", "content": sp.KEYPOINT_PROMPT.format(
                    question=p["surface"], answer=p["bottom"], keypoints=sp.place_keypoints(sp.keypoints_of(p))) +
                    f"\n\nPlayer's question: {q}\n\nAnswer in the format 'Hit point: x, y' or 'No point is hit'."}],
                    role="robustjudge", temperature=0.0, max_tokens=3000, tag=p["_id"])
                if txt:
                    break
                time.sleep(10)
            if not txt:
                continue
            m = re.search(r"Hit point:\s*([\d,\s]+)", txt)
            hs = {int(x) for x in re.findall(r"\d+", m.group(1))} if m else set()
            tot += 1
            same += bool(hs) == bool(hb)
    print(f"batched vs per-question {J}: hit/no-hit agreement {same}/{tot}; episodes failed {fail}")


def report():
    arms = sorted({p.stem.rsplit("_s", 1)[0] for p in OUT.glob("*_s*.json")})
    data = {}
    for a in arms:
        per = {}
        for p in OUT.glob(f"{a}_s*.json"):
            if re.fullmatch(rf"{re.escape(a)}_s\d+\.json", p.name):
                for pid, v in json.loads(p.read_text(encoding="utf-8")).items():
                    if v is not None:
                        per.setdefault(pid, []).append(v)
        data[a] = {pid: mean(v) for pid, v in per.items()}
    print(f"Key recall under {J} (batched protocol):")
    for a in arms:
        print(f"  {a:<9} {mean(data[a].values()):.3f}  (n puzzles {len(data[a])})")
    for a in arms:
        for b in ("a0", "a2", "cce_shuf", "r1"):
            if a != b and b in data:
                common = sorted(set(data[a]) & set(data[b]))
                m, lo, hi = evaluate.boot([data[a][p] - data[b][p] for p in common])
                print(f"  {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(common)})")


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report()
    elif sys.argv[1] == "--agree":
        agree(int(sys.argv[2]))
    else:
        for a in sys.argv[1:]:
            rescore(a)
        report()
