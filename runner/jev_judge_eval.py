"""Score TypeSafe's Jev (a "System One" model) as an AR-Bench referee, on the same
fixed TurtleBench sample as every other candidate in experiments/02.

Jev does not take a chat prompt: it takes a `state` plus typed `questions` and returns
a typed answer with probabilities (POST /v1/systemone, docs.typesafe.ai). So the
official TurtleBench referee prompt cannot be sent byte-identically. What is kept:
the same 150 cases (seed 7), the same Surface/Bottom/guess content, and the three
judging rules verbatim as the question's instructions. What differs: the rules arrive
as `instructions` and the three labels as `criteria` rather than as a system message.
That is the comparison -- a typed-decision model against chat judges on one task.

    export TYPESAFE_API_KEY=...        # or keychain: security add-generic-password -s typesafe_token -a $USER -w
    python runner/jev_judge_eval.py --turtlebench <dir> --out judge/turtle150_jev.json
    python runner/jev_judge_eval.py --turtlebench <dir> --dry-run --limit 2    # build payloads, no network

Writes the same {id: {pred, raw, gold}} shape as turtle_judge_eval.py, plus Jev's
confidence and per-label probabilities, so the predictions drop straight into the
comparison in experiments/02.
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turtle_judge_eval import build_items, LABELS  # same sample, same labels

API = "https://api.typesafe.ai/v1/systemone"

# The three judging rules of prompts/simple_prompt_en.txt, verbatim, as the criteria.
CRITERIA = {
    "Correct": "The player's guess is correct, or the answer is affirmative.",
    "Incorrect": "The player's guess is incorrect, or the answer is negative.",
    "Unknown": "The player's guess cannot be answered from the <Surface> and <Bottom>, "
               "and cannot be concluded through reasoning.",
}
INSTRUCTIONS = (
    "You are a referee in a game. Players are shown the <Surface>; you are told the <Bottom>. "
    "Understand the entire story from both, then judge the player's guess.\n"
    "Players can only see the <Surface>, so they guess about the <Surface>. For example, if a player "
    "asks \"He didn't drink turtle soup\", they are asking whether he drank turtle soup in the "
    "<Surface>; even if he drank other soups in the <Bottom>, judge the <Surface>.\n"
    "For any conclusion that cannot be drawn from the provided story, answer Unknown: if the guess "
    "concerns details not mentioned in the story and not deducible by reasoning, answer Unknown."
)


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k.strip()
    try:
        return subprocess.run(["security", "find-generic-password", "-s", "typesafe_token", "-w"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sys.exit("no API key: set TYPESAFE_API_KEY or store it as keychain service 'typesafe_token'")


def payload(item, model):
    # Surface and Bottom live in the item's system text; split them back out so the
    # story is state (what is true) and the guess is what the question is about.
    sys_text = item["system"]
    surface = sys_text.split("### Surface", 1)[1].split("### Bottom", 1)[0].strip()
    bottom = sys_text.split("### Bottom", 1)[1].split("Now, please judge", 1)[0].strip()
    return {
        "model": model,
        "state": {"Surface": surface, "Bottom": bottom, "player_guess": item["guess"]},
        "questions": {"verdict": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}},
    }


async def one(client, sem, item, model, key, retries=4):
    async with sem:
        for a in range(retries):
            try:
                r = await client.post(API, json=payload(item, model),
                                      headers={"Authorization": f"Bearer {key}"}, timeout=60)
                if r.status_code in (429, 500, 502, 503, 529):
                    await asyncio.sleep(1.5 * (a + 1)); continue
                r.raise_for_status()
                d = r.json()
                ans = d["answers"]["verdict"]
                return {"id": item["id"], "gold": item["gold"], "pred": ans.get("choice"),
                        "raw": ans.get("choice"), "confidence": ans.get("confidence"),
                        "probabilities": ans.get("probabilities"), "usage": d.get("usage"),
                        "model": d.get("model")}
            except Exception as e:
                if a == retries - 1:
                    return {"id": item["id"], "gold": item["gold"], "pred": None, "raw": f"ERROR {e}"}
                await asyncio.sleep(1.5 * (a + 1))


def report(rows, seconds, model):
    rows = sorted(rows, key=lambda r: r["id"])
    n = len(rows)
    ok = sum(1 for r in rows if r["pred"] == r["gold"])
    unparsed = sum(1 for r in rows if r["pred"] not in LABELS)
    print(f"model: {model}")
    print(f"ACCURACY: {ok}/{n} = {100*ok/n:.1f}%   (unparsed {unparsed})")
    print("pred dist:", Counter(r["pred"] for r in rows).most_common())
    print()
    for lab in LABELS:
        g = [r for r in rows if r["gold"] == lab]
        if g:
            h = sum(1 for r in g if r["pred"] == lab)
            print(f"  gold={lab:<10} {h}/{len(g)} = {100*h/len(g):.1f}%")
    conf = [r for r in rows if isinstance(r.get("confidence"), (int, float))]
    if conf:
        # expected calibration error over 10 equal-width bins of the reported confidence
        bins = [[] for _ in range(10)]
        for r in conf:
            bins[min(9, int(r["confidence"] * 10))].append(r["pred"] == r["gold"])
        ece = sum(len(b) / len(conf) * abs(sum(b) / len(b) - (i + 0.5) / 10) for i, b in enumerate(bins) if b)
        print(f"\n  mean confidence {sum(r['confidence'] for r in conf)/len(conf):.3f}  "
              f"accuracy {ok/n:.3f}  ECE {ece:.3f}  (n={len(conf)})")
    tok = sum((r.get("usage") or {}).get("input_tokens", 0) for r in rows)
    print(f"\n  input tokens {tok}  cost ${tok*0.042/1e6:.4f} at $0.042/MTok  wall {seconds:.0f}s  "
          f"({seconds/n*1000:.0f} ms/case at the chosen concurrency)")
    print("\nreference on this exact sample:")
    print("  Llama-3.1-405B-INT4 83.3% | Opus 5 88.0% | Llama-3.1-405B(0shot) 82.7% | "
          "Qwen3.8-Flash-Next(think) 80.7% | GPT-4o(2shot) 63.3%")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--turtlebench", required=True)
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--n", type=int, default=150); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", default=None); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    items = build_items(a.turtlebench, a.n, a.seed)
    if a.limit:
        items = items[: a.limit]
    if a.dry_run:
        print(json.dumps(payload(items[0], a.model), indent=1)[:2000])
        print(f"\n[dry run] {len(items)} cases, no request sent. "
              f"est. input tokens ~{sum(len(json.dumps(payload(i, a.model)))//4 for i in items)}")
        return
    import httpx
    key = api_key()
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(a.concurrency)
        rows = await asyncio.gather(*(one(client, sem, i, a.model, key) for i in items))
    report(rows, time.time() - t0, a.model)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({str(r["id"]): r for r in sorted(rows, key=lambda r: r["id"])}, indent=1))
        print("->", a.out)


if __name__ == "__main__":
    asyncio.run(main())
