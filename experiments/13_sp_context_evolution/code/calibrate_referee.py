"""Choose the SP referee by measured accuracy on human-labelled TurtleBench (English, Apache-2.0).

Uses AR-Bench's own TurtleBench-derived judging prompt (simple_system_prompt: Correct / Incorrect / Unknown).
Reports accuracy, per-class recall and macro recall per candidate model on a fixed seeded sample.
Usage: python -I code/calibrate_referee.py [n=150] [model ...]
"""
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat, spent_usd  # noqa: E402
import sp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TB = ROOT / "data" / "turtlebench"
OUT = ROOT / "runs" / "referee_calibration"
DEFAULT = ["anthropic/claude-haiku-5.5", "qwen/qwen3-235b-a22b-2507", "meta-llama/llama-3.3-70b-instruct",
           "qwen/qwen3-32b", "anthropic/claude-sonnet-5.5"]


def load_cases():
    stories = {s["title"]: s for s in json.loads((TB / "stories.json").read_text(encoding="utf-8"))}
    cases = []
    for line in (TB / "cases.list").read_text(encoding="utf-8").splitlines():
        parts = [x.strip() for x in line.split("|")]
        if len(parts) == 3 and parts[1] in stories and parts[2] in ("Correct", "Incorrect", "Unknown"):
            cases.append({"guess": parts[0], "story": stories[parts[1]], "label": parts[2]})
    return cases


def judge(model, c):
    sysmsg = sp.P["simple_system_prompt"].format(question=c["story"]["surface"], answer=c["story"]["bottom"])
    txt = chat(model, [{"role": "system", "content": sysmsg}, {"role": "user", "content": c["guess"]}],
               role="referee_calib", temperature=0.0, max_tokens=8, tag="turtlebench")
    t = (txt or "").strip().lower()
    for lab in ("Incorrect", "Correct", "Unknown"):   # check Incorrect before Correct (substring)
        if t.startswith(lab.lower()):
            return lab
    return "PARSE_FAIL"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    models = sys.argv[2:] or DEFAULT
    cases = load_cases()
    rng = random.Random(7)
    sample = rng.sample(cases, n)
    print(f"TurtleBench EN cases: {len(cases)}; sample n={n}; label mix {Counter(c['label'] for c in sample)}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for m in models:
        with ThreadPoolExecutor(16) as ex:
            preds = list(ex.map(lambda c: judge(m, c), sample))
        acc = sum(p == c["label"] for p, c in zip(preds, sample)) / n
        rec = {}
        for lab in ("Correct", "Incorrect", "Unknown"):
            idx = [i for i, c in enumerate(sample) if c["label"] == lab]
            rec[lab] = sum(preds[i] == lab for i in idx) / max(1, len(idx))
        macro = sum(rec.values()) / 3
        summary[m] = {"acc": acc, "recall": rec, "macro_recall": macro,
                      "parse_fail": sum(p == "PARSE_FAIL" for p in preds)}
        print(f"{m:<45} acc {acc:.3f}  C/I/U {rec['Correct']:.2f}/{rec['Incorrect']:.2f}/{rec['Unknown']:.2f}"
              f"  macro {macro:.3f}  parse_fail {summary[m]['parse_fail']}  spent so far ${spent_usd():.2f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
