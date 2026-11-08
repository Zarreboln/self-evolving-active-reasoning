"""Fill missing Gemini final-story scores (None) left by rate limits / unparsed replies in earlier runs.
Found 2026-11-08: runs/probe_use/a0_orig.jsonl had 35/100 None, so the 2x2 Gemini column averaged ~65 puzzles.
Covers runs/probe_use/*.jsonl ("gemini" field) and runs/pilot episode files ("story_g" in score). Retries with
backoff; anything still failing stays None and is counted. Usage: python -I code/fill_gemini_story.py [probe|pilot|all]"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import probe_use  # noqa: E402
from llm import spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PUZ = {p["_id"]: p for p in sp.load("test")} | {p["_id"]: p for p in sp.load("train")}


def score(pid, pred):
    if not (pred or "").strip():
        return None
    for attempt in range(5):
        v = probe_use.gem_match(PUZ[pid], pred)
        if v is not None:
            return v
        time.sleep(6 * (attempt + 1))
    return None


def fill_probe():
    for f in sorted((ROOT / "runs" / "probe_use").glob("*.jsonl")):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows or "gemini" not in rows[0]:
            continue
        todo = [i for i, r in enumerate(rows) if r.get("gemini") is None]
        if not todo:
            continue
        with ThreadPoolExecutor(6) as ex:
            vals = list(ex.map(lambda i: score(rows[i]["id"], rows[i]["pred"]), todo))
        for i, v in zip(todo, vals):
            rows[i]["gemini"] = v
        f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        left = sum(r.get("gemini") is None for r in rows)
        print(f"{f.name}: filled {len(todo) - left}/{len(todo)}, still missing {left}; spent ${spent_usd():.2f}",
              flush=True)


def fill_pilot():
    for f in sorted((ROOT / "runs" / "pilot").glob("*_s*.jsonl")):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows or "score" not in rows[0] or "story_g" not in rows[0]["score"]:
            continue
        todo = [i for i, r in enumerate(rows) if r["score"].get("story_g") is None]
        if not todo:
            continue
        with ThreadPoolExecutor(6) as ex:
            vals = list(ex.map(lambda i: score(rows[i]["episode"]["id"], rows[i]["episode"]["pred"]), todo))
        for i, v in zip(todo, vals):
            rows[i]["score"]["story_g"] = v
        f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        left = sum(r["score"].get("story_g") is None for r in rows)
        print(f"{f.name}: filled {len(todo) - left}/{len(todo)}, still missing {left}; spent ${spent_usd():.2f}",
              flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("probe", "all"):
        fill_probe()
        probe_use.report()
    if what in ("pilot", "all"):
        fill_pilot()
