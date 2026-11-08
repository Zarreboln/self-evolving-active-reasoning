"""Third-judge adjudication of Haiku/Gemini disagreements on TEST questions.
For each arm file, take questions where exactly one of the two judges credits a key point (Haiku per-question vs
Gemini batched, both read from cache), sample up to N of each kind (seeded), and ask the Opus-5.5 reference judge with
the same per-question protocol as code/reference_judge.py. Reports how often Opus sides with each judge.
Writes runs/judge_gap/adjudication.json. Usage: python -I code/adjudicate.py N stem [stem ...]"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import rescore_test as rt  # noqa: E402
from llm import chat, spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "runs" / "test"
OUT = ROOT / "runs" / "judge_gap"
REF = "anthropic/claude-opus-5.5"


def disagreements(stem):
    rows = [json.loads(x) for x in (TEST / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()]
    h_only, g_only = [], []
    for r in rows:
        p = rt.puz[r["episode"]["id"]]
        qs = [sp.clean_q(q) for q, _ in r["episode"]["history"]]
        gb = rt.batched_hits(p, qs)
        if gb is None:
            continue
        for t, (q, gh) in enumerate(zip(qs, gb)):
            hh = sp.key_hits(p, q)
            item = {"stem": stem, "id": p["_id"], "turn": t + 1, "q": q, "haiku": sorted(hh), "gemini": sorted(gh)}
            if hh and not gh:
                h_only.append(item)
            elif gh and not hh:
                g_only.append(item)
    return h_only, g_only


def opus(it):
    p = rt.puz[it["id"]]
    prompt = sp.KEYPOINT_PROMPT.format(question=p["surface"], answer=p["bottom"],
                                       keypoints=sp.place_keypoints(sp.keypoints_of(p))) + \
        f"\n\nPlayer's question: {it['q']}\n\nAnswer in the format 'Hit point: x, y' or 'No point is hit'."
    txt = chat(REF, [{"role": "user", "content": prompt}], role="adjudicate", temperature=0.0, max_tokens=3000,
               reasoning=True, tag=it["id"])
    m = re.search(r"Hit point:\s*([\d,\s]+)", txt or "")
    n = len(sp.keypoints_of(p))
    it["opus"] = sorted({int(x) for x in re.findall(r"\d+", m.group(1))} & set(range(1, n + 1))) if m else []
    it["opus_ok"] = bool(txt)
    return it


if __name__ == "__main__":
    N = int(sys.argv[1])
    before = spent_usd()
    todo = []
    for stem in sys.argv[2:]:
        h_only, g_only = disagreements(stem)
        rng = random.Random(f"adj|{stem}")
        todo += [dict(x, kind="haiku_only") for x in rng.sample(h_only, min(N, len(h_only)))]
        todo += [dict(x, kind="gemini_only") for x in rng.sample(g_only, min(N, len(g_only)))]
        print(f"{stem}: haiku-only {len(h_only)}, gemini-only {len(g_only)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        done = list(ex.map(opus, todo))
    (OUT / "adjudication.json").write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
    for stem in sys.argv[2:]:
        for kind in ("haiku_only", "gemini_only"):
            xs = [d for d in done if d["stem"] == stem and d["kind"] == kind and d["opus_ok"]]
            if xs:
                print(f"{stem} {kind}: Opus says hit on {sum(bool(d['opus']) for d in xs)}/{len(xs)}")
    print(f"spend for adjudication: ${spent_usd() - before:.2f}")
