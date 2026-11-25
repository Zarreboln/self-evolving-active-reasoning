"""Step 1b: TextArena's own agents, unchanged, on our split, against our model.

The agent is TextArena's `OpenAIAgent` (STANDARD_GAME_PROMPT, one stateless call per turn) pointed at
our vLLM server, wrapped in TextArena's `ThoughtAgentWrapper` for the reasoning version; the loop is
TextArena's (get_observation -> agent -> step, until done; then close). The env is TextArena's with
its default wrappers. Our glue, and only that: the instance is fixed per split as in our adapters
(mb/benches/{wordle,tagames,twentyq}.py), and 20Q's gamemaster is Jev instead of TextArena's
hard-coded GPT-4o, as in the rest of the suite.

score = TextArena's reward, floored at 0 (raw_reward keeps the -1 of a loss by invalid moves)

    python -m mb.official.run_textarena --out DIR --agent thought --split test --n 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

CELLS = ["wordle:basic5", "mastermind:basic", "hangman:basic", "twentyq:basic-things"]


def _path(out, agent, cell, inst):
    return os.path.join(out, agent, cell.replace(":", "_"), f"{inst}.json")


def episode(job):
    out, agent_kind, cell, inst, url, model = job
    import textarena as ta
    from textarena.agents.wrappers import ThoughtAgentWrapper
    from mb.benches import get
    bench, sub = cell.split(":", 1)
    b = get(bench)
    responder = None
    if b.responder:
        from mb.jev import Jev, responder_for
        responder = responder_for(bench, Jev())
    t0 = time.time()
    rec = {"cell": cell, "instance": inst, "agent": agent_kind}
    try:
        env = b.make(sub, inst, responder)._env          # TextArena env, default wrappers, instance fixed
        agent = ta.agents.OpenAIAgent(model_name=model, api_key="EMPTY", base_url=url)
        if agent_kind == "thought":
            agent = ThoughtAgentWrapper(agent)
        done, turns = False, 0
        while not done:
            _, obs = env.get_observation()
            action = agent(obs)
            done, _ = env.step(action=action)
            turns += 1
        rewards, info = env.close()
        raw = float(rewards[0])
        rec.update(raw_reward=raw, score=max(raw, 0.0), success=raw == 1.0, turns=turns,
                   reason=info[0].get("reason", ""))
    except Exception as e:
        rec.update(score=float("nan"), error=repr(e)[:500])
    rec["seconds"] = round(time.time() - t0, 1)
    p = _path(out, agent_kind, cell, inst)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    return cell, rec.get("score"), rec.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="thought", choices=["standard", "thought"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th pending episode starting at i (one GPU job each)")
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    from mb.benches import get
    urls = [u for u in os.environ["POLICY_BASE_URLS"].split(",") if u]
    model = os.environ["POLICY_MODEL"]
    jobs = []
    for cell in a.cells.split(","):
        bench, sub = cell.split(":", 1)
        for inst in get(bench).instances(sub, a.split, limit=a.n):
            p = _path(a.out, a.agent, cell, inst)
            if os.path.exists(p) and "error" not in json.load(open(p)):
                continue
            jobs.append((a.out, a.agent, cell, inst, urls[len(jobs) % len(urls)], model))
    jobs = [j for j in jobs if zlib.crc32(str(j[3]).encode()) % sk == si]   # fixed across requeues
    print(f"{len(jobs)} episodes to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(episode, j) for j in jobs]):
            cell, score, err = f.result()
            if err:
                print(f"{cell} error {err[:200]}", flush=True)
    rows = []
    for cell in a.cells.split(","):
        d = os.path.join(a.out, a.agent, cell.replace(":", "_"))
        recs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")] if os.path.isdir(d) else []
        ok = [r for r in recs if "error" not in r]
        sc = [r["score"] for r in ok if not math.isnan(r["score"])]
        rows.append({"cell": cell, "agent": a.agent, "split": a.split, "episodes": len(recs), "errors": len(recs) - len(ok),
                     "mean_score": round(sum(sc) / len(sc), 4) if sc else None,
                     "success_rate": round(sum(r["success"] for r in ok) / len(ok), 4) if ok else None})
    json.dump(rows, open(os.path.join(a.out, f"summary_{a.agent}_{a.split}.json"), "w"), indent=1)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
