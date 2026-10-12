"""Measure any served model's referee accuracy on a fixed TurtleBench sample.

TurtleBench (arXiv 2410.05262) is the benchmark AR-Bench itself used to
cross-check its LLM judges (paper Table 13). Its 1531 English cases are real
user guesses with human-annotated Correct/Incorrect/Unknown labels, and its
referee prompt is the one AR-Bench's SP answerer prompt was adapted from --
same three-way scheme, same two few-shot examples.

Sample is fixed (seed 7, 150 cases across 32 stories) so every candidate
judge is scored on exactly the same items as the archived model predictions
shipped in the TurtleBench repo. Reference points on this sample:

    Opus 5 (one fresh context per item)   88.0%
    Llama-3.1-405B  0-shot                82.7%   <- AR-Bench's official judge
    Llama-3.1-70B   2-shot                74.0%
    GPT-4o          2-shot                63.3%

Each item is one independent call: system = rules + Surface + Bottom, user =
the single guess, output = one word. No conversation history -- which matches
how the judge is actually used, since AR-Bench's own loop rebuilds the
answerer's message list from scratch every turn.
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

LABELS = ("Correct", "Incorrect", "Unknown")


def build_items(turtlebench_dir, n=150, seed=7):
    import random
    random.seed(seed)
    root = Path(turtlebench_dir)
    rows = []
    for line in (root / "data/en/cases.list").read_text().splitlines():
        p = [x.strip() for x in line.split("\t|\t")]
        if len(p) == 3:
            rows.append({"guess": p[0], "story": p[1], "gold": p[2]})
    stories = {s["title"]: s for s in json.loads((root / "data/en/stories.json").read_text())}
    tmpl = (root / "prompts/simple_prompt_en.txt").read_text()
    samp = random.sample(rows, n)
    out = []
    for i, r in enumerate(samp):
        s = stories[r["story"]]
        out.append({
            "id": i + 1, "gold": r["gold"], "story": r["story"],
            "system": tmpl.replace("{surface}", s["surface"]).replace("{bottom}", s["bottom"]),
            "guess": r["guess"],
        })
    return out


LABEL_RE = re.compile(r"\b(Incorrect|Correct|Unknown)", re.I)
# Llama-3.1 end-of-turn ids; passed as stop_token_ids because some checkpoints'
# generation_config lists only <|end_of_text|>, and the model then runs on past
# <|eot_id|> into a new "assistant" header ("Incorrectassistant\n\n## Step 1").
LLAMA3_STOP_IDS = [128001, 128008, 128009]


def parse(text):
    """First label to appear wins. Leading word boundary only: the reply may be
    glued to whatever follows ("Correctassistant..."), and the leading boundary
    keeps 'correct' inside 'Incorrect' from matching. Case-insensitive, so a
    reasoning model's think block before the answer is searched too."""
    m = LABEL_RE.search(text or "")
    return m.group(1).capitalize() if m else None


async def main_async(a):
    from openai import AsyncOpenAI

    items = build_items(a.turtlebench, a.n)
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", max_retries=3, timeout=600.0)
    sem = asyncio.Semaphore(a.concurrency)
    results = {}

    async def one(it):
        async with sem:
            extra = {"stop_token_ids": LLAMA3_STOP_IDS} if "llama" in a.model.lower() else {}
            r = await client.chat.completions.create(
                model=a.model, temperature=a.temperature, top_p=a.top_p,
                max_tokens=a.max_tokens, extra_body=extra,
                messages=[{"role": "system", "content": it["system"]},
                          {"role": "user", "content": it["guess"]}])
            raw = r.choices[0].message.content or ""
            results[it["id"]] = {"pred": parse(raw), "raw": raw[:300], "gold": it["gold"]}

    await asyncio.gather(*(one(it) for it in items))

    n = len(results)
    ok = sum(1 for v in results.values() if v["pred"] == v["gold"])
    unparsed = sum(1 for v in results.values() if v["pred"] is None)
    print(f"model: {a.model}")
    print(f"ACCURACY: {ok}/{n} = {100*ok/n:.1f}%   (unparsed {unparsed})")
    print("pred dist:", Counter(v["pred"] for v in results.values()).most_common())
    print()
    for g in LABELS:
        ks = [k for k, v in results.items() if v["gold"] == g]
        c = sum(1 for k in ks if results[k]["pred"] == g)
        print(f"  gold={g:10} {c}/{len(ks)} = {100*c/max(len(ks),1):.1f}%")
    print("\nreference on this exact sample:")
    print("  Opus 5 88.0% | Llama-3.1-405B(0shot) 82.7% | Llama-3.1-70B(2shot) 74.0% | GPT-4o(2shot) 63.3%")
    print("  (Llama-3.1-405B reference = TurtleBench's archived bf16 predictions; ours is the AWQ-INT4 checkpoint)")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
        print(f"\n-> {a.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--turtlebench", default="/tmp/TurtleBench")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", default="")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
