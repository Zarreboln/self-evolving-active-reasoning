"""Convert the BALAR baseline episodes (final_track/balar_results/full100, run by the side session on 2026-11-07 with
the same actor and referee) into the TEST-arm format, so rescore_test.py / consensus_recall.py / evaluate.py report it
with the same judges as the other arms. Scores are copied from rescore_v2 (Haiku + reasoning judge), not recomputed.
Writes runs/test/balar_s0.jsonl (arm "balar", seed 0, field "source" marks the origin)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "final_track" / "balar_results" / "full100"
puz = {p["_id"]: p for p in sp.load("test")}
rows = []
for pid in sorted(puz):
    ep = json.loads((B / "episodes" / f"{pid}.json").read_text(encoding="utf-8"))
    sc = json.loads((B / "rescore_v2" / f"{pid}.json").read_text(encoding="utf-8"))
    hist = [(c["question_text"], c["user_answer"]) for c in ep["conversation_log"]]
    answers = [a.lower() for _, a in hist]
    pred = ep.get("pred") or ""
    rows.append({"arm": "balar", "seed": 0, "source": "final_track/balar_results/full100 (side session)",
                 "episode": {"id": pid, "seed": 0, "history": hist, "pred": pred},
                 "score": {"id": pid, "seed": 0, "key_recall": sc["key_recall"], "answer_match": sc["answer_match"],
                           "f1_char": ep["scores"]["f1_char"], "f1_word": ep["scores"]["f1_word"],
                           "unknown_rate": sum(a.startswith("unknown") for a in answers) / max(1, len(answers)),
                           "pred_words": len(pred.split())}})
(ROOT / "runs" / "test" / "balar_s0.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                                       encoding="utf-8")
print(len(rows), "episodes; mean questions", sum(len(r["episode"]["history"]) for r in rows) / len(rows),
      "; haiku recall", round(sum(r["score"]["key_recall"] for r in rows) / len(rows), 3))
