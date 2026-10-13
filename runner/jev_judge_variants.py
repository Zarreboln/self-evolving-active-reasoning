"""Does Jev do better on the referee task with the phrasings its own docs recommend?

Three levers from docs.typesafe.ai/primitives/advanced: sharpen option boundaries with
structured `what` / `not_for` / `examples` criteria, split a hard question into several,
and keep supporting data as JSON. The baseline (`plain`) is runner/jev_judge_eval.py:
the three TurtleBench rules verbatim, one choice question.

    python runner/jev_judge_variants.py --turtlebench <dir> --variant boundaries
    python runner/jev_judge_variants.py --turtlebench <dir> --variant all --out-dir judge/jev_variants

Variants:
  plain       one choice, rules verbatim                      (the committed baseline)
  boundaries  one choice, criteria as what/not_for/examples
  split       noul "is it decidable from the story" + choice Correct/Incorrect,
              Unknown taken when P(decidable) falls below a threshold
  both        split, with boundaries on the Correct/Incorrect choice

Same 150 cases and seed as every other candidate in experiments/02.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_judge_eval import api_key, INSTRUCTIONS, CRITERIA
from turtle_judge_eval import build_items, LABELS

API = "https://api.typesafe.ai/v1/systemone"

# Lever 1: the Incorrect/Unknown boundary is where every judge on this task loses, so it
# is the boundary the docs' what / not_for / examples form is meant for.
CRITERIA_BOUNDED = {
    "Correct": {
        "what": "The guess states something that is true in the story, or the answer to it is yes.",
        "not_for": "A guess that is merely plausible, or that the story never establishes.",
        "examples": ["the guess matches what the Bottom says happened",
                     "the guess asks whether X happened and X did happen"],
    },
    "Incorrect": {
        "what": "The story settles the guess and settles it the other way: the answer is no.",
        "not_for": "A guess the story is simply silent about. Silence is Unknown, not Incorrect.",
        "examples": ["the guess names a cause the Bottom rules out",
                     "the guess asks whether X happened and the story shows X did not"],
    },
    "Unknown": {
        "what": "Neither the Surface nor the Bottom decides the guess, and it cannot be "
                "deduced from them by reasoning.",
        "not_for": "A guess the story does decide, even if only by implication.",
        "examples": ["the guess concerns a detail the story never mentions",
                     "the guess asks about something outside what either text covers"],
    },
}

DECIDABLE = (
    "Do the <Surface> and the <Bottom> together decide the player's guess, either directly or by "
    "reasoning from them? Answer yes if the story settles it one way or the other. Answer no if "
    "the story is silent on it and it cannot be deduced -- a guess about a detail the story never "
    "mentions is not decided."
)
VERDICT_2 = (
    "Assume the story does decide the player's guess. Is the guess true, or is the answer to it yes? "
    "Judge only against the <Surface> and the <Bottom>. Players can only see the <Surface>, so they "
    "guess about the <Surface>."
)
CRITERIA_2 = {"Correct": "The guess is true, or the answer to it is yes.",
              "Incorrect": "The guess is false, or the answer to it is no."}
CRITERIA_2_BOUNDED = {k: CRITERIA_BOUNDED[k] for k in ("Correct", "Incorrect")}


def state_of(item):
    t = item["system"]
    return {"Surface": t.split("### Surface", 1)[1].split("### Bottom", 1)[0].strip(),
            "Bottom": t.split("### Bottom", 1)[1].split("Now, please judge", 1)[0].strip(),
            "player_guess": item["guess"]}


def questions(variant):
    if variant == "plain":
        return {"verdict": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}
    if variant == "boundaries":
        return {"verdict": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA_BOUNDED}}
    crit = CRITERIA_2_BOUNDED if variant == "both" else CRITERIA_2
    return {"decidable": {"type": "noul", "instructions": DECIDABLE},
            "verdict": {"type": "choice", "instructions": VERDICT_2, "criteria": crit}}


def verdict(variant, ans, thresh):
    if variant in ("plain", "boundaries"):
        a = ans["verdict"]
        return a.get("choice"), a.get("probabilities"), a.get("confidence")
    dec = ans["decidable"].get("noul")
    a = ans["verdict"]
    p = dict(a.get("probabilities") or {})
    if dec is not None:
        p = {k: v * dec for k, v in p.items()}
        p["Unknown"] = 1 - dec
    pred = "Unknown" if (dec is not None and dec < thresh) else a.get("choice")
    return pred, p, a.get("confidence")


async def one(client, sem, item, variant, key, thresh, retries=4):
    async with sem:
        body = {"model": "jev-latest", "state": state_of(item), "questions": questions(variant)}
        for i in range(retries):
            try:
                r = await client.post(API, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
                if r.status_code in (429, 500, 502, 503, 529):
                    await asyncio.sleep(1.5 * (i + 1)); continue
                r.raise_for_status()
                d = r.json()
                pred, probs, conf = verdict(variant, d["answers"], thresh)
                return {"id": item["id"], "gold": item["gold"], "pred": pred, "raw": pred,
                        "probabilities": probs, "confidence": conf, "usage": d.get("usage")}
            except Exception as e:
                if i == retries - 1:
                    return {"id": item["id"], "gold": item["gold"], "pred": None, "raw": f"ERROR {e}"}
                await asyncio.sleep(1.5 * (i + 1))


def score(rows):
    n = len(rows)
    acc = sum(1 for r in rows if r["pred"] == r["gold"]) / n
    rec, f1 = [], []
    for L in LABELS:
        g = [r for r in rows if r["gold"] == L]
        tp = sum(1 for r in g if r["pred"] == L)
        pp = sum(1 for r in rows if r["pred"] == L)
        rc = tp / len(g) if g else 0.0
        pr = tp / pp if pp else 0.0
        rec.append(rc); f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, sum(rec) / 3, sum(f1) / 3, rec


def main_report(name, rows, seconds):
    acc, mr, mf, rec = score(rows)
    tok = sum((r.get("usage") or {}).get("input_tokens", 0) for r in rows)
    print(f"{name:<12} acc {100*acc:5.1f}%  macro recall {100*mr:5.1f}%  macro F1 {100*mf:5.1f}%  "
          f"C/I/U {100*rec[0]:.1f}/{100*rec[1]:.1f}/{100*rec[2]:.1f}  "
          f"{tok} tok  ${tok*0.042/1e6:.4f}  {seconds:.0f}s  "
          f"dist {dict(Counter(r['pred'] for r in rows))}")
    return acc, mr, mf


async def run(variant, items, key, conc, thresh):
    import httpx
    t0 = time.time()
    async with httpx.AsyncClient() as c:
        sem = asyncio.Semaphore(conc)
        rows = await asyncio.gather(*(one(c, sem, i, variant, key, thresh) for i in items))
    return sorted(rows, key=lambda r: r["id"]), time.time() - t0


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--turtlebench", required=True)
    p.add_argument("--variant", default="all",
                   choices=["plain", "boundaries", "split", "both", "all"])
    p.add_argument("--n", type=int, default=150); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=None); p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--thresh", type=float, default=0.5, help="P(decidable) below this means Unknown")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    items = build_items(a.turtlebench, a.n, a.seed)
    if a.limit:
        items = items[: a.limit]
    key = api_key()
    names = ["plain", "boundaries", "split", "both"] if a.variant == "all" else [a.variant]
    print(f"Jev 1.13 as referee, {len(items)} TurtleBench cases (seed {a.seed}). "
          f"Reference: 405B-INT4 acc 83.3 / macro 69.6; Jev plain acc 78.7 / macro 68.3\n")
    saved = {}
    for v in names:
        rows, secs = await run(v, items, key, a.concurrency, a.thresh)
        main_report(v, rows, secs)
        saved[v] = rows
        if a.out_dir:
            d = Path(a.out_dir); d.mkdir(parents=True, exist_ok=True)
            (d / f"turtle150_jev_{v}.json").write_text(
                json.dumps({str(r["id"]): r for r in rows}, indent=1))
    # for the split variants the Unknown threshold is a free knob: show the trade-off
    for v in names:
        if v in ("split", "both"):
            print(f"\n{v}: P(decidable) threshold sweep")
            print(f"  {'thresh':>7} {'acc':>7} {'macro':>7} {'U recall':>9} {'U said':>7}")
            for th in (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
                rows = [dict(r, pred=("Unknown" if (r.get("probabilities") or {}).get("Unknown", 0) > 1 - th
                                      else max({k: x for k, x in (r.get("probabilities") or {}).items()
                                                if k != "Unknown"} or {"Correct": 1}, key=lambda k: (r.get("probabilities") or {}).get(k, 0))))
                        for r in saved[v] if r.get("probabilities")]
                acc, mr, _, rec = score(rows)
                said = sum(1 for r in rows if r["pred"] == "Unknown")
                print(f"  {th:>7.2f} {100*acc:>6.1f}% {100*mr:>6.1f}% {100*rec[2]:>8.1f}% {said:>7}")


if __name__ == "__main__":
    asyncio.run(main())
