"""Concurrent GuessNumbers runner. Same measurements as gn_runner.py, but N
episodes are in flight at once so vLLM can batch them.

Why this exists: the serial runner issues one request at a time, so a 7B on an
L40S decodes at single-stream speed (~80 tok/s) and the 900-episode gate step
needs ~15 h. With 32 episodes in flight the same work is ~40 min, and
concurrency changes only speed, never results -- each episode is independent
and carries its own conversation.

Stage 0 scoring is numpy and CPU-bound. The turn-1 available-bits value is the
expensive one (|S| = 5040 against all 5040 guesses) and is identical for every
episode, so it is computed once at startup and shared; after turn 1 the
feasible set collapses to a few hundred and scoring is cheap.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "stage0"))
from gn_oracle import GNGame, build_feedback_matrix, build_space  # noqa: E402

GUESS_RE = re.compile(r"Guess:\s*(\d{4})")
ANY4_RE = re.compile(r"\d{4}")


def load_prompts(arbench_dir):
    sys.path.insert(0, str(arbench_dir))
    from arbench.reasoner.gn.prompt import (  # noqa: E402
        eval_prompt,
        guess_prompt,
        proactive_cot_system_prompt,
        propose_template,
        propose_template_with_1_shot_inst,
        refine_prompt,
    )
    return ({"bare": propose_template,
             "taught_incremental": propose_template_with_1_shot_inst,
             "cot": proactive_cot_system_prompt},
            guess_prompt, eval_prompt, refine_prompt)


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def belief_summary(game, alive, list_cap=20):
    """The 'belief_injected' condition's only mechanism: compute, externally,
    the sufficient statistic the model is failing to derive from its own
    history (dead-guess rate 88.9-95.7% across all three AR-Bench conditions,
    stable regardless of prompt -- see experiments.md 2026-10-06), and hand it
    over as text instead of asking the model to reconstruct it from a
    transcript it already has in full.

    O(1) in output size regardless of |alive|: a per-digit in/out/unknown
    classification plus any positions pinned down, not a function of how many
    candidates remain. Falls back to listing candidates verbatim only once
    the set is small enough that doing so is cheap and unambiguous.
    """
    sub = game.digits[alive]
    n = len(alive)
    parts = [f"{n} numbers remain consistent with every clue so far."]

    pinned = [f"position {p + 1} is {vals[0]}"
              for p in range(4)
              if len(vals := sorted(set(sub[:, p].tolist()))) == 1]
    if pinned:
        parts.append("Confirmed positions: " + "; ".join(pinned) + ".")

    present, absent = [], []
    for d in range(10):
        hit = (sub == d).any(axis=1)
        if hit.all():
            present.append(d)
        elif not hit.any():
            absent.append(d)
    if present:
        parts.append("Digits confirmed to be IN the secret: "
                      + ", ".join(map(str, present)) + ".")
    if absent:
        parts.append("Digits confirmed NOT in the secret: "
                      + ", ".join(map(str, absent)) + ".")

    if 1 < n <= list_cap:
        opts = sorted("".join(map(str, row)) for row in sub.tolist())
        parts.append("All numbers still consistent with every clue: "
                      + ", ".join(opts) + ".")
    parts.append("Do not guess a number you have already ruled out above.")
    return " ".join(parts)


class Scorer:
    def __init__(self, game):
        self.game = game
        self.full_avail = game.available_bits(np.arange(len(game.cands)))[0]

    def available(self, alive):
        if len(alive) == len(self.game.cands):
            return self.full_avail
        return self.game.available_bits(alive)[0]


async def run_episode(client, model, scorer, system, prompts, secret, seed,
                      max_turns, temperature, max_tokens, inject_belief=False):
    guess_prompt, eval_prompt, refine_prompt = prompts
    game = scorer.game
    msgs = [{"role": "system", "content": system.format(turn=max_turns)}]
    alive = np.arange(len(game.cands))
    rows, prev, out_tokens = [], None, 0

    async def ask():
        nonlocal out_tokens
        r = await client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature,
            seed=seed, max_tokens=max_tokens)
        if r.usage:
            out_tokens += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    for t in range(1, max_turns + 1):
        msgs.append({"role": "user", "content": guess_prompt.format(turn=t)})
        try:
            text = await ask()
        except Exception as e:                      # keep one bad episode from
            rows.append({"turn": t, "error": str(e)[:200]})   # killing the run
            break
        m = GUESS_RE.search(text) or ANY4_RE.search(text)
        if not m:
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": refine_prompt})
            try:
                text = await ask()
            except Exception as e:
                rows.append({"turn": t, "error": str(e)[:200]})
                break
            m = GUESS_RE.search(text) or ANY4_RE.search(text)
        msgs.append({"role": "assistant", "content": text})

        guess = (m.group(1) if m and m.lastindex else (m.group(0) if m else None))
        if guess is None or guess not in game.index:
            rows.append({"turn": t, "guess": guess, "illegal": True})
            prev = None
            continue

        gi = game.index[guess]
        dead = gi not in set(alive.tolist())
        avail = scorer.available(alive)
        chosen = game.expected_bits(gi, alive)
        code = game.code(guess, secret)
        after = alive[game.fb[gi, alive] == code]
        rows.append({
            "turn": t, "guess": guess, "code": code,
            "n_before": int(len(alive)), "n_after": int(len(after)),
            "bits_chosen": chosen, "bits_available": avail,
            "efficiency": None if avail <= 0 else chosen / avail,
            "dead_guess": dead,
            "anchor_dist": None if prev is None else hamming(prev, guess),
            "solved": guess == secret,
        })
        prev = guess
        if guess == secret:
            return t, rows, out_tokens
        alive = after if len(after) else alive
        feedback = eval_prompt.format(same_pos=code // 5, diff_pos=code % 5)
        if inject_belief:
            # The only difference from "bare": the raw numeric feedback is
            # kept (it is the true observation), but a belief summary computed
            # externally from the full history is appended, so the model is
            # not required to re-derive it from the transcript it already has.
            feedback = feedback + "\n" + belief_summary(game, alive)
        msgs.append({"role": "user", "content": feedback})

    return None, rows, out_tokens


async def main_async(args):
    from openai import AsyncOpenAI

    arbench = Path(args.arbench)
    systems, gp, ep, rp = load_prompts(arbench)
    secrets = json.load(open(arbench / "data/gn/test.json"))[: args.n_instances]

    t_build = time.time()
    cands, digits, masks = build_space()
    game = GNGame(build_feedback_matrix(digits, masks), cands)
    scorer = Scorer(game)
    print(f"stage0 ready in {time.time()-t_build:.1f}s "
          f"(|H|={len(cands)}, turn-1 available bits={scorer.full_avail:.3f})",
          flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                done.add((r["condition"], r["secret"], r["seed"]))
            except Exception:
                pass
    todo = [(c, s, k) for c in args.conditions.split(",")
            for s in secrets for k in range(args.seeds)
            if (c, s, k) not in done]
    print(f"resuming past {len(done)}; to run {len(todo)} episodes "
          f"at concurrency {args.concurrency}", flush=True)
    if not todo:
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY",
                         max_retries=3, timeout=600.0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    fh = out.open("a")
    t0 = time.time()
    counter = {"n": 0}

    async def one(cond, secret, seed):
        # "belief_injected" reuses the "bare" system prompt verbatim -- same
        # rules text, same guess format -- so the only variable between the
        # two conditions is the per-turn belief summary. Any difference in
        # outcome is therefore attributable to that one mechanism, not to a
        # confound from wording the rules differently.
        inject = cond == "belief_injected"
        sys_prompt = systems["bare"] if inject else systems[cond]
        async with sem:
            k, rows, tok = await run_episode(
                client, args.model, scorer, sys_prompt, (gp, ep, rp),
                secret, seed, args.max_turns, args.temperature, args.max_tokens,
                inject_belief=inject)
        rec = {"condition": cond, "secret": secret, "seed": seed,
               "turns_to_solve": k, "solved": k is not None,
               "out_tokens": tok, "rows": rows}
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counter["n"] += 1
            n = counter["n"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el:.0f}s  {el/n:.2f}s/ep  "
                      f"eta {(len(todo)-n)*el/n/60:.1f}min", flush=True)

    await asyncio.gather(*(one(*x) for x in todo))
    fh.close()
    print(f"done in {time.time()-t0:.0f}s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arbench", default=str(ROOT / "AR-Bench"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default="bare,taught_incremental,cot")
    ap.add_argument("--n-instances", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
