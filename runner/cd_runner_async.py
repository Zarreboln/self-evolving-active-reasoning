"""Concurrent CircuitDecoding runner with Stage 0 scoring inline.

CircuitDecoding is the family with room for a belief trap: feedback is one
bit, and querying an entry the inventory plus history already pins down is
legal, non-contradictory and worth exactly zero bits. Measured on the 300
released instances (stage0/survey_families.py, 2026-10-06):

    policy          solved   queries   ex-ante eff   zero-info queries
    oracle          300/300    11.60      1.000             0.0%
    greedy-4sample  250/300    13.92      0.775            18.1%
    random-fresh    156/300    16.51      0.629            27.7%
    sequential       58/300    17.41      0.540            38.1%
    random           35/300    17.63      0.480            44.1%

Read the model against those rows. The paper reports Gemini 2.5 Pro at 32.2%,
i.e. between random and sequential, so there should be plenty of trap to see.

Same append-only JSONL and (condition, instance, seed) resume as the GN runner.
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
from cd_oracle import CDGame, load_instances  # noqa: E402

LABELS = "ABCDEFGH"
QUERY_RE = re.compile(r"<query>\s*([A-H])\s*\(([^)]*)\)\s*</query>", re.I)
LOOSE_RE = re.compile(r"\b([A-H])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)")
GUESS_RE = re.compile(r"<guess>\s*([01\s]+)\s*</guess>", re.I)


def fill_instruction(tmpl, components, c, k, budget):
    labels = list(LABELS[:c])
    comp = ", ".join(f"{v} {g} gate{'s' if v != 1 else ''}"
                     for g, v in sorted(components.items()) if v)
    order = ", ".join(
        f"{lb}({','.join(str((r >> (k - 1 - i)) & 1) for i in range(k))})"
        for lb in labels for r in range(1 << k))
    return (tmpl.replace("{components}", comp)
                .replace("{num_circuits}", str(c))
                .replace("{circuit_labels}", ", ".join(labels))
                .replace("{circuit_labels_str}", ", ".join(labels))
                .replace("{num_inputs}", str(k))
                .replace("{num_questions_per_episode}", str(budget))
                .replace("{truth_table_description}", order))


def parse_query(text, c, k):
    """-> joint truth-table index, or None."""
    m = QUERY_RE.search(text)
    if m:
        lb, args = m.group(1).upper(), m.group(2)
        bits = re.findall(r"[01]", args)
    else:
        m = LOOSE_RE.search(text)
        if not m:
            return None
        lb, bits = m.group(1).upper(), [m.group(2), m.group(3), m.group(4)]
    if lb not in LABELS[:c] or len(bits) != k:
        return None
    row = int("".join(bits), 2)
    return LABELS.index(lb) * (1 << k) + row


def belief_summary(game, alive, c, k, list_cap=15):
    """CD's analogue of gn_runner_async.belief_summary: an externally computed
    sufficient statistic of what remains possible, handed to the model
    instead of asking it to re-derive it from the query/answer transcript.

    Per-bit confirmed status is the CD equivalent of GN's per-digit in/out
    classification: for each of the 24 truth-table entries, is it forced to 0,
    forced to 1, or still undetermined given every remaining hypothesis?
    O(1) regardless of |alive|. Falls back to listing the exact remaining
    joint truth tables once the set is small enough to do so unambiguously --
    the CD analogue of GN's "list every remaining candidate number."
    """
    sub = game.H[alive]
    n = len(alive)
    rows_per_circuit = 1 << k
    parts = [f"{n} circuit configurations remain consistent with every "
             f"answer given so far."]

    all_one = (sub == 1).all(axis=0)
    all_zero = (sub == 0).all(axis=0)
    determined = [(idx, 1) for idx in np.flatnonzero(all_one)]
    determined += [(idx, 0) for idx in np.flatnonzero(all_zero)]
    if determined:
        def fmt(idx):
            lb = LABELS[idx // rows_per_circuit]
            row = idx % rows_per_circuit
            bits = ",".join(str((row >> (k - 1 - i)) & 1) for i in range(k))
            return f"{lb}({bits})"
        txts = [f"{fmt(idx)}={v}" for idx, v in sorted(determined)]
        parts.append("Already determined outputs (do not ask these again): "
                      + ", ".join(txts) + ".")

    if 1 < n <= list_cap:
        opts = sorted("".join(map(str, row)) for row in sub.tolist())
        parts.append("All circuit configurations still consistent with every "
                      "answer (as 24-bit joint truth tables): "
                      + ", ".join(opts) + ".")
    return " ".join(parts)


def parse_guess(text, length):
    m = GUESS_RE.search(text)
    s = re.sub(r"\s", "", m.group(1)) if m else ""
    if len(s) != length:
        cand = re.findall(r"[01]{%d}" % length, re.sub(r"\s", "", text))
        s = cand[-1] if cand else ""
    return np.array([int(ch) for ch in s], dtype=np.uint8) if len(s) == length else None


async def run_episode(client, model, game, inst, system, last_prompt, seed,
                      budget, temperature, max_tokens, c, k, inject_belief=False):
    msgs = [{"role": "system", "content": system}]
    alive = np.arange(len(game.H))
    truth = inst["truth"]
    rows, out_tokens, prev_q = [], 0, None

    async def ask(content):
        nonlocal out_tokens
        msgs.append({"role": "user", "content": content})
        r = await client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature,
            seed=seed, max_tokens=max_tokens)
        if r.usage:
            out_tokens += r.usage.completion_tokens
        txt = r.choices[0].message.content or ""
        msgs.append({"role": "assistant", "content": txt})
        return txt

    try:
        txt = await ask("Please start with your first query.")
        for t in range(1, budget + 1):
            q = parse_query(txt, c, k)
            if q is None:
                rows.append({"turn": t, "illegal": True})
                txt = await ask("Your message did not contain a query in the "
                                "required format. Reply with exactly one "
                                "<query>X(a, b, c)</query>.")
                continue
            avail, qb = game.available_bits(alive)
            chosen = game.expected_bits(q, alive)
            bit = int(truth[q])
            after = alive[game.H[alive, q] == bit]
            rows.append({
                "turn": t, "query": q, "circuit": q // (1 << k),
                "row": q % (1 << k), "answer": bit,
                "n_before": int(len(alive)), "n_after": int(len(after)),
                "bits_chosen": chosen, "bits_available": avail,
                "efficiency": None if avail <= 0 else chosen / avail,
                "zero_info": chosen < 1e-9,
                "repeat": prev_q is not None and q in {r.get("query") for r in rows[:-1]},
            })
            prev_q = q
            alive = after if len(after) else alive
            if t < budget:
                feedback = f"The output of that query is {bit}. Please make your next query."
                if inject_belief:
                    # Same mechanism as GN's belief_injected: the raw bit
                    # answer is kept (it is the true observation), but a
                    # belief summary computed externally from the full
                    # history is appended, so the model is not required to
                    # re-derive it from the transcript it already has.
                    feedback = feedback + "\n" + belief_summary(game, alive, c, k)
                txt = await ask(feedback)
        final = await ask(last_prompt)
        guess = parse_guess(final, len(truth))
    except Exception as e:
        return None, rows, out_tokens, str(e)[:200]

    correct = guess is not None and bool((guess == truth).all())
    # what the feasible set alone would have given, had the agent just read it
    forced = len(alive) == 1
    return {"correct": correct, "unique_at_end": forced,
            "guess_bits": guess.tolist() if guess is not None else None,
            "truth_bits": truth.tolist(),
            "n_alive_end": int(len(alive)),
            "parsed_guess": guess is not None}, rows, out_tokens, None


async def main_async(a):
    from openai import AsyncOpenAI

    tmpl = (ROOT / "mt_puzzles/cd_instruction.txt").read_text()
    last_prompt = (ROOT / "mt_puzzles/cd_last_turn_prompt.txt").read_text()
    insts = load_instances(ROOT / "mt_puzzles/cd_test.jsonl")[: a.n_instances]

    t = time.time()
    games = {}
    for inst in insts:
        key = tuple(sorted(inst["components"].items()))
        if key not in games:
            games[key] = CDGame(dict(key))
    print(f"stage0 ready in {time.time()-t:.1f}s; "
          f"|H| = {sorted(len(g.H) for g in games.values())}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                if r.get("error"):
                    continue  # retry errored episodes (e.g. context overflow)
                      # on the next run instead of treating them as finished
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
                         max_retries=3, timeout=900.0)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    fh, t0, cnt = out.open("a"), time.time(), {"n": 0}

    async def one(cond, i, seed):
        inst = insts[i]
        game = games[tuple(sorted(inst["components"].items()))]
        system = fill_instruction(tmpl, inst["components"], a.circuits,
                                  a.inputs, a.budget)
        if cond == "no_strategy":
            system = system.replace("**Think Strategically:**", "").replace(
                "* Plan your {} queries carefully to maximize information "
                "gained.".format(a.budget), "")
        # "belief_injected" reuses the plain instruction verbatim -- same
        # rules, same query format -- so the per-turn belief summary is the
        # only variable between it and "bare". Mirrors gn_runner_async.py.
        inject = cond == "belief_injected"
        async with sem:
            res, rows, tok, err = await run_episode(
                client, a.model, game, inst, system, last_prompt, seed,
                a.budget, a.temperature, a.max_tokens, a.circuits, a.inputs,
                inject_belief=inject)
        rec = {"condition": cond, "key": i, "seed": seed, "error": err,
               "result": res, "out_tokens": tok, "rows": rows}
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            cnt["n"] += 1
            n = cnt["n"]
            if n % 20 == 0 or n == len(todo):
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
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", default="bare")
    p.add_argument("--n-instances", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=18)
    p.add_argument("--circuits", type=int, default=3)
    p.add_argument("--inputs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
