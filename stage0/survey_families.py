"""Run the Stage 0 instrument across every task family whose latent state is
enumerable and whose answerer is a program. The point is to find which family
has room for a belief trap: a legal, non-contradictory move worth zero bits.

Excluded, with reasons:
  word_chaining      no latent state to infer -- it is a game, not inference
  twenty_questions   the MODEL holds the secret and answers; the instrument
                     measures query choice, so it does not apply to its role
  movie_recommend.   latent preference is a continuous 8-vector; needs a
                     discretisation before it is a hypothesis set
  AR-Bench sp / dc   natural-language latents, not enumerable; answerer is a
                     405B model. Cross-family held-out only.
"""

import json
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from generic import DiscreteGame, benchmark, print_table  # noqa: E402


# ---------------------------------------------------------------- word_guess

def wordle_code(guess, secret):
    """Standard Wordle scoring, greens claimed before yellows. Returns a
    base-3 code: 2 green, 1 yellow, 0 grey."""
    n = len(guess)
    res = [0] * n
    pool = {}
    for i in range(n):
        if guess[i] == secret[i]:
            res[i] = 2
        else:
            pool[secret[i]] = pool.get(secret[i], 0) + 1
    for i in range(n):
        if res[i] == 0 and pool.get(guess[i], 0) > 0:
            res[i] = 1
            pool[guess[i]] -= 1
    c = 0
    for v in res:
        c = c * 3 + v
    return c


def build_word_guess():
    rows = [json.loads(l)["controller"] for l in
            open(ROOT / "mt_puzzles/wg_test.jsonl")]
    cache, games = {}, []
    for r in rows:
        vocab = tuple(r["wordle_vocab"])
        if vocab not in cache:
            F = np.array([[wordle_code(g, s) for s in vocab] for g in vocab],
                         dtype=np.int16)
            cache[vocab] = DiscreteGame(F, n_codes=243)
        games.append((cache[vocab], vocab.index(r["secret_word"])))
    return games, rows[0]["num_tries"], len(cache)


# ---------------------------------------------------------------- circuits

def build_circuit_decoding():
    from cd_oracle import CDGame, load_instances
    insts = load_instances(ROOT / "mt_puzzles/cd_test.jsonl")
    cache, games = {}, []
    for inst in insts:
        key = tuple(sorted(inst["components"].items()))
        if key not in cache:
            cd = CDGame(dict(key))
            # query q = joint-truth-table entry q; answer is that bit
            cache[key] = (DiscreteGame(cd.H.T.astype(np.int16), n_codes=2), cd)
        g, cd = cache[key]
        truth_idx = int(np.flatnonzero((cd.H == inst["truth"]).all(1))[0])
        games.append((g, truth_idx))
    return games, 18, len(cache)


# ---------------------------------------------------------------- numbers

def build_guess_numbers(limit=100):
    from gn_oracle import build_feedback_matrix, build_space
    cands, digits, masks = build_space()
    fb = build_feedback_matrix(digits, masks)
    g = DiscreteGame(fb.astype(np.int16), n_codes=25)
    idx = {s: i for i, s in enumerate(cands)}
    secrets = json.load(open(ROOT / "AR-Bench/data/gn/test.json"))[:limit]
    return [(g, idx[s]) for s in secrets], 25, 1


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    import time

    t0 = time.time()
    specs = [
        ("GuessNumbers (AR-Bench)", build_guess_numbers),
        ("WordGuess (MTP)", build_word_guess),
        ("CircuitDecoding (MTP)", build_circuit_decoding),
    ]
    for title, builder in specs:
        t = time.time()
        games, budget, n_distinct = builder()
        g0 = games[0][0]
        extra = (f"{len(games)} instances, |H|={g0.n_h}, |Q|={g0.n_q}, "
                 f"budget={budget}, {n_distinct} distinct game(s), "
                 f"built {time.time()-t:.1f}s")
        print_table(title, benchmark(games, budget), extra)
    print(f"\ntotal {time.time()-t0:.1f}s")
