"""Bottom row of the asking-vs-using 2x2: a STRONG asker (Qwen3-235B, the same
model family/size as the referee but in the asker role, never seeing the hidden story) plays the official SP loop on
the 100 TEST puzzles (seed 0). Its transcript is then turned into a story twice:
  strong_strong  the strong model writes the story itself (as in the episode)
  strong_7b      the frozen Qwen2.5-7B writes the story from the strong model's transcript
Question quality: key-point recall under Haiku / Gemini / consensus (same protocols as the TEST arms).
Story quality: AR-Bench guess_eval match count (0-6) under Haiku and Gemini, plus char-F1.
Usage: python -I code/strong_asker.py [n]   -> runs/probe_use/strong_*.jsonl, appended summary in strong_report.md
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import rescore_test as rt  # noqa: E402
from llm import chat, spent_usd  # noqa: E402
from probe_use import msgs_for, gem_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "probe_use"
ASKER = sp.REFEREE


def one(p):
    ep = sp.run_episode(p, playbook="", seed=0, actor_model=ASKER)
    qs = [sp.clean_q(q) for q, _ in ep["history"]]
    hb = [sp.key_hits(p, q) for q in qs]
    gb = rt.batched_hits(p, qs)
    n = len(sp.keypoints_of(p))
    rec = {"haiku": len(set().union(*hb)) / n, "unknown_rate": sum(a.lower().startswith("unknown")
                                                                    for _, a in ep["history"]) / len(qs)}
    if gb is not None:
        rec["gemini"] = len(set().union(*gb)) / n
        rec["consensus"] = len(set().union(*[a & b for a, b in zip(hb, gb)])) / n
    m = msgs_for(p, "", ep["history"]) + [{"role": "user", "content": sp.FINAL_PROMPT}]
    story7 = chat(sp.ACTOR, m, role="actor", temperature=0.7, max_tokens=512, seed=7000, extra={"top_p": 0.7},
                  tag=p["_id"])
    out = {"id": p["_id"], "episode": ep, "question_recall": rec}
    for cond, pred in (("strong_strong", ep["pred"]), ("strong_7b", story7)):
        out[cond] = {"pred": pred, "haiku": sp.answer_match(p, pred), "gemini": gem_match(p, pred),
                     "f1_char": sp.f1(pred, p["bottom"])}
    return out


def main(n):
    ps = sp.load("test")[:n]
    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(one, ps))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strong_asker.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                            encoding="utf-8")
    lines = [f"Strong asker ({ASKER}) on {len(rows)} TEST puzzles, seed 0", "",
             "Question key-point recall: " + ", ".join(
                 f"{k} {mean(r['question_recall'][k] for r in rows if k in r['question_recall']):.3f}"
                 for k in ("haiku", "gemini", "consensus")) +
             f"; Yes/No share {1 - mean(r['question_recall']['unknown_rate'] for r in rows):.3f}", ""]
    for cond in ("strong_strong", "strong_7b"):
        g = [r[cond]["gemini"] for r in rows if r[cond]["gemini"] is not None]
        lines.append(f"{cond}: story match Haiku {mean(r[cond]['haiku'] for r in rows):.2f}, Gemini {mean(g):.2f}, "
                     f"char-F1 {mean(r[cond]['f1_char'] for r in rows):.3f}")
    txt = "\n".join(lines)
    (OUT / "strong_report.md").write_text(txt + "\n", encoding="utf-8")
    print(txt, f"\nspent ${spent_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
