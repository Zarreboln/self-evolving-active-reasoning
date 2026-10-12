"""Resumable GuessNumbers runner with Stage 0 scoring inline.

Backends:
  vllm        OpenAI-compatible endpoint (local vLLM serving Qwen2.5-7B-Instruct)
  synthetic   the probe policies, for testing the plumbing with no GPU

Writes append-only JSONL. Re-running skips (condition, instance, seed) keys that
are already complete, so a preempted Slurm job resumes where it stopped --
mit_preemptable kills jobs mid-run, so this is not optional there.

Gate question: under the AR-Bench prompt that teaches bit-by-bit digit
replacement, does the model show the "anchored" signature -- low ex-ante
information efficiency and Hamming distance 1 from its own previous guess?
"""

import argparse
import json
import os
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
    return {
        # the three harness conditions for the gate step
        "bare": propose_template,
        "taught_incremental": propose_template_with_1_shot_inst,
        "cot": proactive_cot_system_prompt,
    }, guess_prompt, eval_prompt, refine_prompt


# ---------------------------------------------------------------- backends


class VLLMBackend:
    def __init__(self, base_url, model, temperature, seed, max_tokens=800):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.out_tokens = 0

    def chat(self, messages):
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, seed=self.seed,
            max_tokens=self.max_tokens,
        )
        if r.usage:
            self.out_tokens += r.usage.completion_tokens
        return r.choices[0].message.content or ""


class SyntheticBackend:
    """Plumbing test with no GPU: replays a known policy as if it were a model."""

    def __init__(self, game, policy, rng):
        self.game, self.policy, self.rng = game, policy, rng
        self.alive = np.arange(len(game.cands))
        self.state = {}
        self.out_tokens = 0

    def chat(self, messages):
        g = self.policy(self.game, self.alive, self.rng, self.state)
        self.out_tokens += 8
        return f"Guess: {g}"

    def observe(self, guess, code):
        gi = self.game.index[guess]
        after = self.alive[self.game.fb[gi, self.alive] == code]
        self.alive = after if len(after) else self.alive


# ---------------------------------------------------------------- episode


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def run_episode(game, backend, system, guess_prompt, eval_prompt, refine_prompt,
                secret, max_turns):
    msgs = [{"role": "system", "content": system.format(turn=max_turns)}]
    alive = np.arange(len(game.cands))
    rows, prev = [], None
    full_pool_avail = None

    for t in range(1, max_turns + 1):
        msgs.append({"role": "user", "content": guess_prompt.format(turn=t)})
        text = backend.chat(msgs)
        m = GUESS_RE.search(text) or ANY4_RE.search(text)
        if not m:
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": refine_prompt})
            text = backend.chat(msgs)
            m = GUESS_RE.search(text) or ANY4_RE.search(text)
        msgs.append({"role": "assistant", "content": text})

        guess = m.group(1) if m and m.lastindex else (m.group(0) if m else None)
        if guess is None or guess not in game.index:
            rows.append({"turn": t, "guess": guess, "illegal": True})
            prev = None
            continue

        gi = game.index[guess]
        dead = gi not in set(alive.tolist())
        if len(alive) == len(game.cands):
            if full_pool_avail is None:
                full_pool_avail = game.available_bits(alive)[0]
            avail = full_pool_avail
        else:
            avail = game.available_bits(alive)[0]
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
        if hasattr(backend, "observe"):
            backend.observe(guess, code)
        if guess == secret:
            return t, rows
        alive = after if len(after) else alive
        msgs.append({"role": "user", "content": eval_prompt.format(
            same_pos=code // 5, diff_pos=code % 5)})

    return None, rows


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "synthetic"], default="vllm")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arbench", default=str(ROOT / "AR-Bench"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default="bare,taught_incremental,cot")
    ap.add_argument("--n-instances", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    arbench = Path(args.arbench)
    systems, guess_prompt, eval_prompt, refine_prompt = load_prompts(arbench)
    secrets = json.load(open(arbench / "data/gn/test.json"))[: args.n_instances]

    cands, digits, masks = build_space()
    game = GNGame(build_feedback_matrix(digits, masks), cands)

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
    print(f"resuming: {len(done)} episodes already in {out}", flush=True)

    todo = [(c, s, k) for c in args.conditions.split(",")
            for s in secrets for k in range(args.seeds)
            if (c, s, k) not in done]
    print(f"to run: {len(todo)} episodes", flush=True)

    t0 = time.time()
    with out.open("a") as fh:
        for i, (cond, secret, seed) in enumerate(todo, 1):
            if args.backend == "vllm":
                backend = VLLMBackend(args.base_url, args.model,
                                      args.temperature, seed)
            else:
                from probe_synthetic import make_sloppy, p_anchored, p_consistent
                pol = {"bare": p_consistent,
                       "taught_incremental": p_anchored,
                       "cot": make_sloppy(0.3)}[cond]
                backend = SyntheticBackend(game, pol,
                                           np.random.default_rng(seed))
            k, rows = run_episode(game, backend, systems[cond], guess_prompt,
                                  eval_prompt, refine_prompt, secret,
                                  args.max_turns)
            fh.write(json.dumps({
                "condition": cond, "secret": secret, "seed": seed,
                "turns_to_solve": k, "solved": k is not None,
                "out_tokens": backend.out_tokens, "rows": rows,
            }) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el:.0f}s  "
                      f"{el/i:.2f}s/ep  eta {(len(todo)-i)*el/i/60:.1f}min",
                      flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
