"""Step 1c (BabyAI): BALROG's own evaluator and agents, unchanged, on our BabyAI level and seeds.

BALROG's `Evaluator.run_episode` plays the episode with BALROG's agent (`cot` for the reasoning
version, `naive` otherwise), prompt builder and vLLM client, reading BALROG's own config.yaml. Our glue,
and only that: `make_env` is pointed at the official BabyAI level we use (GoToLocal is not one of
BALROG's five MixedTrainLocal tasks), built with BALROG's BabyAITextCleanLangWrapper and EnvWrapper,
and the seed is our instance (`config.envs.env_kwargs.seed`, which run_episode reads).

score = success = BALROG's progression (1 once the mission is done)

    python -m mb.official.run_balrog --out DIR --agent cot --level GoToLocal --split test --n 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

BALROG = os.path.expanduser("~/code/multibench/BALROG")


def episode(job):
    out, agent_kind, level, seed, url, model, idx = job
    import gymnasium as gym
    import minigrid  # noqa: F401  (registers the BabyAI levels)
    from omegaconf import OmegaConf
    import balrog.evaluator as BE
    from balrog.agents import AgentFactory
    from balrog.environments.babyai_text.clean_lang_wrapper import BabyAITextCleanLangWrapper
    from balrog.environments.env_wrapper import EnvWrapper

    def make_env(env_name, task, config, render_mode=None):
        # the official level with its own defaults, wrapped as BALROG wraps BabyAI
        return EnvWrapper(BabyAITextCleanLangWrapper(gym.make(f"BabyAI-{task}-v0", render_mode=render_mode)),
                          env_name, task)

    BE.make_env = make_env
    cfg = OmegaConf.load(os.path.join(BALROG, "balrog/config/config.yaml"))
    cfg.agent.type = agent_kind
    cfg.client.client_name = "vllm"
    cfg.client.model_id = model
    cfg.client.base_url = url
    cfg.envs.env_kwargs.seed = int(seed)
    cfg.tasks.babyai_tasks = [level]
    rundir = os.path.join(out, agent_kind, level, "balrog_runs", str(seed))
    os.makedirs(rundir, exist_ok=True)
    t0 = time.time()
    rec = {"level": level, "seed": seed, "agent": agent_kind}
    try:
        ev = BE.Evaluator("babyai", cfg, original_cwd=BALROG, output_dir=rundir)
        log = ev.run_episode(level, AgentFactory(cfg).create_agent(), episode_idx=idx)
        prog = log.get("progression", 0.0)
        rec.update(score=float(prog), success=float(prog) >= 1.0, steps=log.get("num_steps"),
                   input_tokens=log.get("input_tokens"), output_tokens=log.get("output_tokens"))
    except Exception as e:
        rec.update(score=float("nan"), error=repr(e)[:500])
    rec["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(out, agent_kind, level, f"{seed}.json")
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    return rec.get("score"), rec.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="cot", choices=["naive", "cot"])
    ap.add_argument("--level", default="GoToLocal")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th pending episode starting at i (one GPU job each)")
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    from mb.benches import get
    urls = [u for u in os.environ["POLICY_BASE_URLS"].split(",") if u]
    model = os.environ["POLICY_MODEL"]
    seeds = get("babyai").instances(a.level, a.split, limit=a.n)       # the same seeds as our adapter
    d = os.path.join(a.out, a.agent, a.level)
    os.makedirs(d, exist_ok=True)
    jobs = []
    for i, s in enumerate(seeds):
        p = os.path.join(d, f"{s}.json")
        if os.path.exists(p) and "error" not in json.load(open(p)):
            continue
        jobs.append((a.out, a.agent, a.level, s, urls[len(jobs) % len(urls)], model, i))
    jobs = [j for j in jobs if zlib.crc32(str(j[3]).encode()) % sk == si]   # fixed across requeues
    print(f"{len(jobs)} episodes to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(episode, j) for j in jobs]):
            score, err = f.result()
            if err:
                print("error", err[:200], flush=True)
    recs = [json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")]
    ok = [r for r in recs if "error" not in r]
    sc = [r["score"] for r in ok if not math.isnan(r["score"])]
    row = {"bench": "babyai", "level": a.level, "agent": a.agent, "split": a.split, "episodes": len(recs),
           "errors": len(recs) - len(ok), "mean_score": round(sum(sc) / len(sc), 4) if sc else None}
    json.dump(row, open(os.path.join(a.out, f"summary_{a.agent}_{a.level}_{a.split}.json"), "w"), indent=1)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
