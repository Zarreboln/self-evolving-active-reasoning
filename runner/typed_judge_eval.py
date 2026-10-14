"""Judge TurtleBench by reading the label probabilities off our own model, instead of
sampling a word and parsing it.

This is the mechanism behind "System One" decision models, applied to a model we already
serve: ask for one token with logprobs, then renormalise over the three label tokens.
Nothing is parsed, so an answer cannot go off-schema, and every verdict carries a
distribution rather than a bare word.

Why it matters here: the 405B's weak class is Unknown (recall 26.3 on TurtleBench-150).
A hard label gives no way to trade that off; P(Unknown) does. The report prints accuracy
at argmax, the calibration error, and what accuracy would be if Unknown were taken
whenever its probability clears a threshold.

The prompt is AR-Bench's / TurtleBench's official referee prompt, unchanged -- the same
system message and the same 150 cases (seed 7) as runner/turtle_judge_eval.py, so the
numbers sit in the same table as every other candidate in experiments/02.

    python runner/typed_judge_eval.py --turtlebench ~/TurtleBench \
        --base-url http://127.0.0.1:18001/v1 --model meta-llama/Llama-3.1-405B-Instruct-INT4 \
        --out judge/turtle150_405b_typed.json
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
from turtle_judge_eval import build_items, LABELS  # same sample, same three labels


def label_probs(top_logprobs):
    """Renormalise one position's distribution over the three labels.

    With guided decoding the model can only emit one of the three strings, so the first
    generated token already separates them -- their first characters differ (C / I / U) --
    and a token is credited to the label it is a prefix of. Without guided decoding this
    rule is unsafe: a model writing prose emits "I" for "I need to...", which would be
    credited to Incorrect. That is why --guided is the default.
    """
    acc = {l: 0.0 for l in LABELS}
    for t in top_logprobs:
        tok = (t.get("token") or "").strip().lower()
        if not tok:
            continue
        p = math.exp(t["logprob"])
        for l in LABELS:
            # "in" must not steal probability from "incorrect"'s rivals: longest match wins
            if l.lower().startswith(tok) or tok.startswith(l.lower()):
                acc[l] += p
                break
    tot = sum(acc.values())
    return {l: v / tot for l, v in acc.items()} if tot > 0 else None


async def one(client, sem, item, base_url, model, top_k, max_tokens=8, guided=True,
              no_think=False, retries=4):
    async with sem:
        # A reasoning model's first token is its thinking marker, not the answer, so ask for a
        # few tokens and read the distribution at the first position that carries label mass.
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": True,
                "top_logprobs": top_k,
                "messages": [{"role": "system", "content": item["system"]},
                             {"role": "user", "content": item["guess"]}]}
        if guided:
            # vLLM constrains generation to these three strings, which is the whole mechanism:
            # the answer cannot go off-schema and the distribution is over the options only.
            # The field is `structured_outputs`; the older `guided_choice` is accepted and then
            # silently ignored by vLLM 0.19, which looks exactly like a working run.
            body["structured_outputs"] = {"choice": list(LABELS)}
        if no_think:
            # Qwen3 and friends open with a thinking block, so the label is not at position 0
            # and its own probability is never exposed. This turns the block off in the template.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        for a in range(retries):
            try:
                r = await client.post(f"{base_url}/chat/completions", json=body, timeout=120)
                r.raise_for_status()
                ch = r.json()["choices"][0]
                content = ch["logprobs"]["content"]
                probs, pos = None, None
                for i, step in enumerate(content):
                    probs = label_probs(step["top_logprobs"])
                    if probs is not None:
                        pos = i
                        break
                if probs is None:
                    return {"id": item["id"], "gold": item["gold"], "pred": None,
                            "raw": "no label in top-%d over %d positions: %s"
                                   % (top_k, len(content), [t["token"] for t in content])}
                pred = max(probs, key=probs.get)
                H = -sum(p * math.log(p) for p in probs.values() if p > 0)
                return {"id": item["id"], "gold": item["gold"], "pred": pred,
                        "raw": ch["message"]["content"], "probabilities": probs,
                        "read_at_token": pos,
                        "confidence": 1 - H / math.log(len(LABELS))}
            except Exception as e:
                if a == retries - 1:
                    return {"id": item["id"], "gold": item["gold"], "pred": None, "raw": f"ERROR {e}"}
                await asyncio.sleep(1.5 * (a + 1))


def report(rows, seconds, model):
    rows = sorted(rows, key=lambda r: r["id"])
    n = len(rows)
    ok = sum(1 for r in rows if r["pred"] == r["gold"])
    print(f"model: {model}  (typed read: guided to the 3 labels, top-k logprobs renormalised)")
    print(f"ACCURACY: {ok}/{n} = {100*ok/n:.1f}%   "
          f"(no label in top-k: {sum(1 for r in rows if r['pred'] is None)})")
    print("pred dist:", Counter(r["pred"] for r in rows).most_common())
    print()
    for lab in LABELS:
        g = [r for r in rows if r["gold"] == lab]
        if g:
            h = sum(1 for r in g if r["pred"] == lab)
            print(f"  gold={lab:<10} {h}/{len(g)} = {100*h/len(g):.1f}%")
    scored = [r for r in rows if r.get("probabilities")]
    if scored:
        bins = [[] for _ in range(10)]
        for r in scored:
            conf = r["probabilities"][r["pred"]]
            bins[min(9, int(conf * 10))].append(r["pred"] == r["gold"])
        ece = sum(len(b) / len(scored) * abs(sum(b) / len(b) - (i + 0.5) / 10)
                  for i, b in enumerate(bins) if b)
        mean_p = sum(r["probabilities"][r["pred"]] for r in scored) / len(scored)
        print(f"\n  mean P(argmax) {mean_p:.3f}  accuracy {ok/n:.3f}  ECE {ece:.3f}")
        print("\n  taking Unknown whenever P(Unknown) clears a threshold:")
        print(f"    {'thresh':>7} {'accuracy':>9} {'Unknown recall':>15} {'Unknown said':>13}")
        for th in (1.01, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
            pr = [("Unknown" if r["probabilities"]["Unknown"] >= th else
                   max({k: v for k, v in r["probabilities"].items() if k != "Unknown"},
                       key=lambda k: r["probabilities"][k]), r["gold"]) for r in scored]
            acc = sum(1 for p, g in pr if p == g) / len(pr)
            gu = [1 for p, g in pr if g == "Unknown" and p == "Unknown"]
            nu = sum(1 for r in scored if r["gold"] == "Unknown")
            said = sum(1 for p, _ in pr if p == "Unknown")
            lab = "argmax" if th > 1 else f"{th:.2f}"
            print(f"    {lab:>7} {100*acc:>8.1f}% {100*len(gu)/nu:>14.1f}% {said:>13}")
    at = [r["read_at_token"] for r in rows if r.get("read_at_token") is not None]
    if at:
        print(f"\n  label found at token position: {dict(Counter(at))}")
    print(f"\n  wall {seconds:.0f}s")
    print("\nreference on this exact sample (sampled word, not typed read):")
    print("  Llama-3.1-405B-INT4 83.3% | Opus 5 88.0% | Qwen3.8-Flash-Next(think) 80.7%")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--turtlebench", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=150); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=8,
                   help="positions to scan for the label; >1 lets a reasoning model emit its "
                        "thinking marker first")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--disable-thinking", dest="no_think", action="store_true",
                   help="pass chat_template_kwargs enable_thinking=false (Qwen3 and similar)")
    p.add_argument("--no-guided", dest="guided", action="store_false",
                   help="do not constrain generation to the three labels (unsafe for models "
                        "that emit a thinking block or prose first)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    items = build_items(a.turtlebench, a.n, a.seed)
    if a.limit:
        items = items[: a.limit]
    import httpx
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(a.concurrency)
        rows = await asyncio.gather(*(one(client, sem, i, a.base_url, a.model, a.top_k, a.max_tokens, a.guided, a.no_think)
                                      for i in items))
    report(rows, time.time() - t0, a.model)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({str(r["id"]): r for r in sorted(rows, key=lambda r: r["id"])}, indent=1))
        print("->", a.out)


if __name__ == "__main__":
    asyncio.run(main())
