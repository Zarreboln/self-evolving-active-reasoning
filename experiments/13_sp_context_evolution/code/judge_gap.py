"""Per-question comparison of the main judge (Haiku, per-question) and the robustness judge (Gemini, batched) on saved
TEST episodes. Both judges' replies are read from the disk cache written during scoring/rescoring, so this normally
costs nothing (a cache miss would re-ask the judge). Reports, per arm file: share of questions each judge credits,
Haiku-only and Gemini-only credits, and examples of Haiku-only credits.
Usage: python -I code/judge_gap.py a0_s0 a2_s1 traj_s0 ..."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import rescore_test as rt  # noqa: E402
from llm import spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "runs" / "test"
OUT = ROOT / "runs" / "judge_gap"


def analyse(stem):
    rows = [json.loads(x) for x in (TEST / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()]
    n = h = g = h_only = g_only = 0
    examples = []
    for r in rows:
        p = rt.puz[r["episode"]["id"]]
        qs = [sp.clean_q(q) for q, _ in r["episode"]["history"]]
        gb = rt.batched_hits(p, qs)
        if gb is None:
            continue
        for q, gh in zip(qs, gb):
            hh = sp.key_hits(p, q)
            n += 1
            h += bool(hh)
            g += bool(gh)
            if hh and not gh:
                h_only += 1
                examples.append({"id": p["_id"], "q": q, "haiku_points": sorted(hh),
                                 "points": [sp.keypoints_of(p)[i - 1] for i in sorted(hh)]})
            g_only += bool(gh) and not hh
    return {"arm": stem, "questions": n, "haiku_credit": h / n, "gemini_credit": g / n,
            "haiku_only": h_only / n, "gemini_only": g_only / n}, examples


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    before = spent_usd()
    summary = []
    for stem in sys.argv[1:]:
        s, ex = analyse(stem)
        summary.append(s)
        random.Random(0).shuffle(ex)
        (OUT / f"{stem}_haiku_only.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{stem}: n={s['questions']} haiku credits {s['haiku_credit']:.3f}  gemini {s['gemini_credit']:.3f}  "
              f"haiku-only {s['haiku_only']:.3f}  gemini-only {s['gemini_only']:.3f}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"new spend for this analysis: ${spent_usd() - before:.3f}")
