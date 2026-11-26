"""Step 1c (ScienceWorld): SwiftSage's ReAct baseline (Lin et al. 2023), branch `science_world`.

From baselines/eval_agent_react.py, line for line: the prompt ('Interact with a household...' + the
task's one demo from baselines/ReAct_baseline/prompt.jsonl), `clean`, the episode loop (think steps
answered 'OK.', every other action mapped onto a valid one by eval_utils.findValidActionNew, the prompt
cut from the front to fit the model's window, a negative score ends the episode at 0, stop after
2*env_step_limit steps or on a loop), and the settings (simplification "easy", env_step_limit 100;
`llm_gpt` at temperature 0, max_tokens 50, stop "\n"). `findValidActionNew` is copied from
eval_utils.py unchanged; the files are vendored in mb/evolve/seeds_src/. The changes, and only these:
  * `llm_gpt` calls our model; the window and token counting stay gpt-3.5-turbo's (4096, tiktoken),
    as the script's default model is gpt-3.5-turbo;
  * variations are our split's (ScienceWorld's own train/dev/test lists), not test_mini;
  * `act` (the act-only version) is SwiftSage's SayCan setup with prompt_no_think.jsonl, but maps
    actions with the same token-overlap rule as ReAct instead of SayCan's sentence-embedding model
    (decided 2026-11-25; the harness process cannot load that model).

score = SwiftSage's final score / 100

    python -m mb.official.run_swiftsage --out DIR --agent react --split test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC = os.path.join(os.path.dirname(__file__), "..", "evolve", "seeds_src")
ENV_STEP_LIMIT = 100

rooms = ["hallway", "greenhouse", "green house", "kitchen", "bathroom", "outside", "workshop", "art studio", "foundry", "bedroom", "living room"]


class _Log:
    def info(self, *a, **k):
        pass


def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s


def findValidActionNew(predictions, env, look, recent_actions, sbert_model, logger, k=5):
    global rooms
    valid_open_door = ["open door to " + i for i in rooms]
    invalid_focus = ["focus on "+x for x in ["agent", "air"]+rooms]
    validActions = set(env.getValidActionObjectCombinations())
    validActions.update(valid_open_door)
    validActions.difference_update(invalid_focus)

    inventory = env.inventory().lower()

    validActions.difference_update(recent_actions[-3:])

    for va in list(validActions):
        if "door" in va and "open" not in va:
            validActions.remove(va)
            continue
        if va.startswith("focus on"):
            pattern = re.compile(r"\b(?:focus|on|in|to)\b", re.IGNORECASE)
            used_objs = pattern.sub("", va).split(" ")
            valid = True
            for obj in used_objs:
                if obj not in look + " " + inventory:
                    valid = False
            if not valid:
                validActions.remove(va)


    # 1) if acton in top k is valid, choose it
    found_valid_in_top = False
    action = None
    for pred in predictions[:k]:
        pred = pred.replace("green house", "greenhouse")
        if pred.strip() in validActions:
            found_valid_in_top = True
            action = pred.strip()
            break
    if found_valid_in_top:
        return action
    else:
        logger.info(f"No valid action found in top k={k} predictions.")
        validActions = list(validActions)
        validActions.sort(key=lambda x: len(x))
        logger.info("Valid Predictions: "+ str(validActions))


    # 2) else, find most similar action

    if sbert_model:
        raise NotImplementedError("sentence-embedding matching is not used here (see module docstring)")
    else:
        # jaccard
        topValue = 0.0
        topAction = predictions[0]
        # embPred = sbert_model.encode(pred, convert_to_tensor=True)
        tokensPred = predictions[0].split(" ")
        uniqueTokensPred = set(tokensPred)

        for validAction in validActions:
            tokensAction = validAction.split(" ")
            uniqueTokensAction = set(tokensAction)

            intersection = uniqueTokensPred.intersection(uniqueTokensAction)
            if (len(intersection) > topValue):
                topAction = validAction
                topValue = len(intersection)

        logger.info("TOP VALID ACTION: " + topAction)
        # Sanitize top action
        topAction = re.sub(r'[^A-Za-z0-9 ]+', '', topAction)
        action = topAction
    return action


def episode(job):
    out, agent_kind, task, variation, url, model = job
    import tiktoken
    from openai import OpenAI
    from scienceworld import ScienceWorldEnv
    client = OpenAI(api_key="EMPTY", base_url=url)
    logger = _Log()

    def llm_gpt(prompt, stop=["\n"], model_name=model):
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], n=1, temperature=0,
            max_tokens=50, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=stop)
        return response.choices[0].message.content or ""

    pf = "swiftsage_prompt.jsonl" if agent_kind == "react" else "swiftsage_prompt_no_think.jsonl"
    d = json.load(open(os.path.join(SRC, pf)))
    t0 = time.time()
    rec = {"task": task, "variation": variation, "agent": agent_kind}
    try:
        env = ScienceWorldEnv("", envStepLimit=ENV_STEP_LIMIT)
        task_num = env.getTaskNames().index(task)        # SwiftSage: taskName = env.getTaskNames()[task_num]
        env.load(task, int(variation), "easy", generateGoldPath=True)
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        task_description = env.taskdescription()[18:]
        recent_actions = ["look around"]
        obs, info = env.reset()
        done = False
        score = 0.0
        last_score = 0.0
        step = 0
        max_steps = ENV_STEP_LIMIT * 2
        init_prompt = 'Interact with a household to solve a task. Here is an example.\n' + d[str(task_num)]
        prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'
        max_len = 4096
        trace = []
        while not done:
            while len(encoding.encode(init_prompt + prompt)) > max_len - 60:
                index1 = init_prompt.find('>')
                if index1 == -1:
                    index1_prompt = prompt.find('>')
                    index2_prompt = prompt.find('>', index1_prompt+1)
                    prompt = prompt[:index1_prompt] + prompt[index2_prompt:]
                else:
                    index2 = init_prompt.find('>', index1+1)
                    if index2 == -1:
                        init_prompt = init_prompt[:index1]
                    else:
                        init_prompt = init_prompt[:index1] + init_prompt[index2:]
            action = llm_gpt(init_prompt + prompt, stop=['\n']).strip()
            if action.startswith('think:'):
                obs = 'OK.'
            else:
                action = findValidActionNew([action], env, info['look'], recent_actions, None, logger)
                obs, reward, done, info = env.step(action)
                score = info['score']
                if score < 0:
                    done = True
                    score = 0
                last_score = score
            obs = clean(obs)
            prompt += f' {action}\n{obs}\n>'
            recent_actions.append(action)
            trace.append([action, obs])
            step += 1
            if (step >= max_steps) or done:
                break
            if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                break
        rec.update(score=score / 100.0, success=score >= 100, steps=step, trace=trace)
    except Exception as e:
        rec.update(score=float("nan"), error=repr(e)[:500])
    rec["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(out, agent_kind, task, f"{variation}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    return rec.get("score"), rec.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="react", choices=["react", "act"])
    ap.add_argument("--task", default="test-conductivity")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th pending episode starting at i (one GPU job each)")
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    from mb.benches.sciworld import instances
    urls = [u for u in os.environ["POLICY_BASE_URLS"].split(",") if u]
    model = os.environ["POLICY_MODEL"]
    vs = instances(a.task, a.split, limit=a.n)
    d = os.path.join(a.out, a.agent, a.task)
    jobs = []
    for v in vs:
        p = os.path.join(d, f"{v}.json")
        if os.path.exists(p) and "error" not in json.load(open(p)):
            continue
        jobs.append((a.out, a.agent, a.task, v, urls[len(jobs) % len(urls)], model))
    jobs = [j for j in jobs if zlib.crc32(str(j[3]).encode()) % sk == si]   # fixed across requeues
    print(f"{len(jobs)} episodes to run of {len(vs)}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(episode, j) for j in jobs]):
            score, err = f.result()
            if err:
                print("error", err[:200], flush=True)
    recs = [json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")]
    ok = [r for r in recs if "error" not in r]
    sc = [r["score"] for r in ok if not math.isnan(r["score"])]
    row = {"bench": "sciworld", "task": a.task, "agent": a.agent, "split": a.split, "episodes": len(recs),
           "errors": len(recs) - len(ok), "mean_score": round(sum(sc) / len(sc), 4) if sc else None}
    json.dump(row, open(os.path.join(a.out, f"summary_{a.agent}_{a.task}_{a.split}.json"), "w"), indent=1)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
