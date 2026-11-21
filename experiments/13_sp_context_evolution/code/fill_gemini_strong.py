"""Fill missing Gemini story scores (None, rate limits) in runs/probe_use/strong_asker.jsonl (2026-11-08).
Same retry rule as fill_gemini_story.py; anything still failing stays None and is counted."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_gemini_story as fg  # noqa: E402
from llm import spent_usd  # noqa: E402

F = fg.ROOT / "runs" / "probe_use" / "strong_asker.jsonl"
rows = [json.loads(x) for x in F.read_text(encoding="utf-8").splitlines() if x.strip()]
todo = [(i, k) for i, r in enumerate(rows) for k in ("strong_strong", "strong_7b") if r[k].get("gemini") is None]
with ThreadPoolExecutor(6) as ex:
    vals = list(ex.map(lambda t: fg.score(rows[t[0]]["id"], rows[t[0]][t[1]]["pred"]), todo))
for (i, k), v in zip(todo, vals):
    rows[i][k]["gemini"] = v
F.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
for k in ("strong_strong", "strong_7b"):
    v = [r[k]["gemini"] for r in rows if r[k]["gemini"] is not None]
    print(k, "gemini mean", round(sum(v) / len(v), 3), "n", len(v), "missing", len(rows) - len(v))
print(f"spent ${spent_usd():.2f}")
