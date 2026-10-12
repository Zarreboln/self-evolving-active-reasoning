"""Concurrent SP (Situation Puzzle / "turtle soup") runner, bare + belief_injected.

AR-Bench's own evaluator (sp_evaluator.py, _run_traditional_evaluation) is
correct but serial, and has no belief_injected condition -- same gap GN/CD
had. "bare" here reimplements that exact zero_shot loop using AR-Bench's own
prompt templates verbatim (propose_template, system_prompt_with_2shots,
get_answer_prompt from arbench.reasoner.sp.prompt), so it should reproduce
run_arbench_sp.py's numbers; this file adds concurrency across episodes and
the belief_injected condition on top.

SP has no exact oracle: the hidden state is free text (story_tree/bottom),
not an enumerable hypothesis space like GN/CD, so there is no mechanical
zero_info ground truth here (see stage0/dc_oracle.py's docstring for why the
same limit applies to DC's evidence fields). key_question coverage is the
best available proxy, not a ground truth.

belief_injected's mechanism: AR-Bench already ships keypoint_hits_prompt in
arbench/reasoner/sp/prompt.py -- a judge prompt that scores whether a given
question hits any of the case's key_question list, using the same "Hit
point: i, j" convention DC's live answerer uses -- but sp_evaluator.py never
calls it. After each turn here, an extra local judge call (same served
model) applies that prompt to the question just asked, accumulates hit
indices across turns, and the running confirmed/total count is appended to
the respondent's answer before it goes back to the policy agent -- the same
"restate what the transcript already implies, don't add new info" spirit as
GN/CD's belief_injected, just using AR-Bench's own judge prompt instead of
an invented heuristic.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AR = ROOT / "AR-Bench"
sys.path.insert(0, str(AR))
from arbench.reasoner.sp.prompt import (  # noqa: E402
    propose_template, system_prompt_with_2shots, get_answer_prompt,
    keypoint_hits_prompt,
)
from arbench.utils.utils_sp import f1_score, place_keypoints  # noqa: E402

HIT_RE = re.compile(r"Hit point:\s*([\d,\s]+)")


def parse_hits(text):
    m = HIT_RE.search(text)
    if not m:
        return set()
    return {int(x) for x in re.findall(r"\d+", m.group(1))}


def load_instances(path, n):
    return json.load(open(path))[:n]


async def run_episode(client, model, inst, max_turn, temperature, seed, inject_belief,
                      jclient=None, jmodel=None):
    # The policy (question asker) talks to `client`/`model`; the respondent and
    # the keypoint judge talk to `jclient`/`jmodel` -- AR-Bench's protocol uses
    # Llama-3.1-405B for both roles, so they must not default to the 7B.
    jclient, jmodel = (jclient or client), (jmodel or model)
    surface, bottom, keypoints = inst["surface"], inst["bottom"], inst["key_question"]
    propose_msgs = [{"role": "system",
                      "content": propose_template.format(turn=max_turn, question=surface)}]
    respond_sys = {"role": "system",
                   "content": system_prompt_with_2shots.format(question=surface, answer=bottom)}
    confirmed = set()
    in_tok = out_tok = 0
    rows = []

    async def call(messages, judge=False):
        nonlocal in_tok, out_tok
        c, m = (jclient, jmodel) if judge else (client, model)
        extra = {"stop_token_ids": [128001, 128008, 128009]} if "llama" in (m or "").lower() else {}
        r = await c.chat.completions.create(
            model=m, messages=messages, temperature=temperature, top_p=0.7, seed=seed,
            extra_body=extra)
        if r.usage:
            in_tok += r.usage.prompt_tokens
            out_tok += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    try:
        for turn in range(max_turn):
            propose_msgs.append({"role": "user",
                                   "content": f"Turn {turn+1}: please propose your next question."})
            question = await call(propose_msgs)
            propose_msgs.append({"role": "assistant", "content": question})

            answer = (await call([respond_sys, {"role": "user", "content": question}], judge=True)).strip().strip(".")

            hits = set()
            if inject_belief:
                judge = await call([{"role": "user", "content": keypoint_hits_prompt.format(
                    question=surface, answer=bottom,
                    keypoints=place_keypoints(keypoints))}], judge=True)
                hits = parse_hits(judge)
                confirmed |= hits

            rows.append({"turn": turn + 1, "question": question, "answer": answer,
                         "hits": sorted(hits)})

            feedback = answer
            if inject_belief:
                remain = len(keypoints) - len(confirmed)
                feedback += (f"\n[{len(confirmed)}/{len(keypoints)} key aspects of the "
                             f"solution confirmed by your questions so far. {remain} "
                             f"still unconfirmed -- avoid re-asking about what's "
                             f"already settled.]")
            propose_msgs.append({"role": "user", "content": feedback})

        propose_msgs.append({"role": "user", "content": get_answer_prompt})
        prediction = await call(propose_msgs)
    except Exception as e:
        return None, rows, in_tok, out_tok, str(e)[:200]

    return ({"prediction": prediction,
             "f1_char": f1_score(prediction, bottom),
             "f1_word": f1_score(prediction.split(), bottom.split()),
             "confirmed_keypoints": sorted(confirmed),
             "n_keypoints": len(keypoints)},
            rows, in_tok, out_tok, None)


async def main_async(a):
    from openai import AsyncOpenAI

    insts = load_instances(AR / "data/sp/test.json", a.n_instances)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                if r.get("error"):
                    continue
                done.add((r["condition"], r["key"], r["seed"]))
            except Exception:
                pass
    todo = [(c, i, s) for c in a.conditions.split(",")
            for i in range(len(insts)) for s in range(a.seeds)
            if (c, i, s) not in done]
    print(f"resuming past {len(done)}; to run {len(todo)} at "
          f"concurrency {a.concurrency}", flush=True)
    if not todo:
        return

    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY",
                         max_retries=3, timeout=1200.0)
    # respondent + keypoint judge: AR-Bench's 405B role, served separately
    jclient = (AsyncOpenAI(base_url=a.judge_base_url, api_key="EMPTY", max_retries=3, timeout=1200.0)
               if a.judge_base_url else None)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    fh, t0, cnt = out.open("a"), time.time(), {"n": 0}

    async def one(cond, i, seed):
        inst = insts[i]
        inject = cond == "belief_injected"
        async with sem:
            res, rows, in_tok, out_tok, err = await run_episode(
                client, a.model, inst, a.max_turn, a.temperature, seed, inject,
                jclient=jclient, jmodel=a.judge_model)
        rec = {"condition": cond, "key": i, "seed": seed, "error": err,
               "result": res, "in_tokens": in_tok, "out_tokens": out_tok, "rows": rows}
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            cnt["n"] += 1
            n = cnt["n"]
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el:.0f}s  {el/n:.2f}s/ep  "
                      f"eta {(len(todo)-n)*el/n/60:.1f}min", flush=True)

    await asyncio.gather(*(one(*x) for x in todo))
    fh.close()
    print(f"done in {time.time()-t0:.0f}s -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--judge-base-url", default=None,
                   help="endpoint for the respondent + keypoint judge (AR-Bench: Llama-3.1-405B)")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", default="bare")
    p.add_argument("--n-instances", type=int, default=100)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--max-turn", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
