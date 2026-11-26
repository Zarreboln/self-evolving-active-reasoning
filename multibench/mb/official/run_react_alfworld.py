"""Step 1c (ALFWorld): the ReAct agent (Yao et al. 2022), from ysymyth/ReAct alfworld.ipynb.

`alfworld_run`, `process_ob` and the prompt assembly are the notebook's, line for line, with the
two-example prompt for the task type (`react_put_1` + `react_put_0` for pick_and_place) from the
repo's prompts/alfworld_3prompts.json (vendored unchanged in mb/evolve/seeds_src/). The changes, and
only these:
  * `llm`: text-davinci-002 is retired; the prompt goes to our model as one user message with the
    notebook's settings (temperature 0, max_tokens 100, stop "\n"), as SwiftSage's port of ReAct to
    gpt-3.5 does (baselines/eval_agent_react.py `llm_gpt`);
  * the env is one game at a time (our ALFWorld adapter: the same game files and ALFWorld wrappers),
    so the notebook's batch indexing `[0]` is dropped;
  * the initial observation is cut as the notebook cuts it ('\n'.join(ob.split('\n\n')[1:])).
  * a leading '>' is stripped from the reply: the notebook's completion model continues after the
    prompt's final '>', a chat model repeats it ('> go to drawer 1'), which the env does not parse.
  * 'put X in/on Y' (ReAct's prompts, older ALFWorld) is sent as 'move X to Y', the same command
    in ALFWorld 0.4's text env.
`react` is the reasoning version; `act` uses the repo's act-only examples (`act_put_*`).

score = success = the game's `won`

    python -m mb.official.run_react_alfworld --out DIR --agent react --split test
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

PROMPTS = os.path.join(os.path.dirname(__file__), "..", "evolve", "seeds_src", "alfworld_3prompts.json")
PREFIXES = {"pick_and_place": "put", "pick_clean_then_place": "clean", "pick_heat_then_place": "heat",
            "pick_cool_then_place": "cool", "look_at_obj": "examine", "pick_two_obj": "puttwo"}


def episode(job):
    out, agent_kind, sub, game, url, model = job
    from openai import OpenAI
    from mb.benches.alfworld import AlfWorldEnv
    client = OpenAI(api_key="EMPTY", base_url=url)

    def llm(prompt, stop=["\n"]):
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=100, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=stop)
        return response.choices[0].message.content or ""

    def process_ob(ob):
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ')+2:]
        return ob

    def alfworld_run(env, prompt, ob=''):
        init_prompt = prompt + ob + '\n>'
        prompt = ''
        steps = []
        for i in range(1, 50):
            action = llm(init_prompt + prompt, stop=['\n']).strip()
            # chat-model glue: the completion prompt ends in '>', which a completion model continues
            # after; a chat model repeats it at the start of its reply ('> go to drawer 1')
            action = action.lstrip('>').strip()
            # env-version glue: ReAct's prompts use ALFWorld's old 'put X in/on Y'; ALFWorld 0.4's text
            # env renamed it 'move X to Y' (its own handcoded_expert_tw issues 'move {} to {}')
            m = re.match(r'^put (.+?) in/on (.+)$', action)
            if m:
                action = f'move {m.group(1)} to {m.group(2)}'
            observation = env.act(action)
            observation, reward, done = process_ob(observation), env.success(), env.done
            if action.startswith('think:'):
                observation = 'OK.'
            steps.append([action, observation])
            prompt += f' {action}\n{observation}\n>'
            if done:
                return reward, steps
        return 0, steps

    d = json.load(open(PROMPTS))
    t0 = time.time()
    rec = {"subtask": sub, "game": game, "agent": agent_kind}
    try:
        env = AlfWorldEnv(sub, game)
        ob = '\n'.join(env.instructions.split('\n\n')[1:])
        v = next(v for k, v in PREFIXES.items() if sub.startswith(k))
        prompt = ('Interact with a household to solve a task. Here are two examples.\n'
                  + d[f'{agent_kind}_{v}_1'] + d[f'{agent_kind}_{v}_0'] + '\nHere is the task.\n')
        r, steps = alfworld_run(env, prompt, ob=ob)
        rec.update(score=float(bool(r)), success=bool(r), steps=len(steps), trace=steps)
    except Exception as e:
        rec.update(score=float("nan"), error=repr(e)[:500])
    rec["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(out, agent_kind, sub, game.replace("/", "__") + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    return rec.get("score"), rec.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="react", choices=["react", "act"])
    ap.add_argument("--subtask", default="pick_and_place_simple")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th pending episode starting at i (one GPU job each)")
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    from mb.benches.alfworld import instances
    urls = [u for u in os.environ["POLICY_BASE_URLS"].split(",") if u]
    model = os.environ["POLICY_MODEL"]
    games = instances(a.subtask, a.split, limit=a.n or None)
    d = os.path.join(a.out, a.agent, a.subtask)
    jobs = []
    for g in games:
        p = os.path.join(d, g.replace("/", "__") + ".json")
        if os.path.exists(p) and "error" not in json.load(open(p)):
            continue
        jobs.append((a.out, a.agent, a.subtask, g, urls[len(jobs) % len(urls)], model))
    jobs = [j for j in jobs if zlib.crc32(str(j[3]).encode()) % sk == si]   # fixed across requeues
    print(f"{len(jobs)} episodes to run of {len(games)}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(episode, j) for j in jobs]):
            score, err = f.result()
            if err:
                print("error", err[:200], flush=True)
    recs = [json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")]
    ok = [r for r in recs if "error" not in r]
    sc = [r["score"] for r in ok if not math.isnan(r["score"])]
    row = {"bench": "alfworld", "subtask": a.subtask, "agent": a.agent, "split": a.split, "episodes": len(recs),
           "errors": len(recs) - len(ok), "mean_score": round(sum(sc) / len(sc), 4) if sc else None}
    json.dump(row, open(os.path.join(a.out, f"summary_{a.agent}_{a.subtask}_{a.split}.json"), "w"), indent=1)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
