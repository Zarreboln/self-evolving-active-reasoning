"""A ruler for the other judge: the one inside the TTHE loop that commits a harness.

TurtleBench measures the referee -- the model that answers the player. Nothing measures the
judge that reads the traces of a batch and decides which candidate harness is deployed next.
It needs no external benchmark: every run already records the pool's gold scores, which the
judge never saw, and which candidate it committed. Pick accuracy against those scores is the
measurement.

    python runner/judge_pick_eval.py meta/cd_tthe/logs/sp_strict1_b10g3r2 [more runs...]

Prints, per batch: the pool size, the committed candidate's gold score, the pool's best, the
tie-aware rank, and what the choice cost. Then per run: how often the judge committed a
best-scoring candidate, its mean regret (best minus committed), and its mean rank.

Regret is the number that matters. Rank is reported too but reads badly on a saturated pool:
when six candidates all score 1.000, rank 5 and rank 1 are the same decision.
"""

import json
import sys
from pathlib import Path

GOLD_KEY = {"sp": "f1_char", "cd": "circuit_wise_accuracy"}


def run_rows(run):
    rows = [json.loads(l) for l in (Path(run) / "opt_log.jsonl").open()]
    task = "sp" if "f1_char" in rows[0]["prequential"] else "cd"
    return rows, task, GOLD_KEY[task]


def main(runs):
    print(f"{'run':<26} {'batch':>5} {'pool':>5} {'committed':>10} {'gold':>7} {'best':>7} "
          f"{'regret':>7} {'rank':>8}")
    summary = []
    for run in runs:
        rows, task, key = run_rows(run)
        name = Path(run).name
        regrets, hits, ranks, n = [], 0, [], 0
        for d in rows:
            gold = {k: v.get(key) for k, v in (d["pool"] or {}).items() if v.get(key) is not None}
            chosen = d["committed"]
            if len(gold) < 2 or chosen not in gold:
                continue
            best = max(gold.values())
            reg = best - gold[chosen]
            rank = 1 + sum(1 for v in gold.values() if v > gold[chosen] + 1e-9)  # tie-aware
            n += 1; regrets.append(reg); ranks.append(rank); hits += (reg < 1e-9)
            print(f"{name:<26} {d['batch']:>5} {len(gold):>5} {chosen.split('_')[-1]:>10} "
                  f"{gold[chosen]:>7.3f} {best:>7.3f} {reg:>7.3f} {rank:>5}/{len(gold):<3}")
        if n:
            summary.append((name, task, hits, n, sum(regrets) / n, max(regrets), sum(ranks) / n))
    print(f"\n{'run':<26} {'task':>5} {'picked a best':>14} {'mean regret':>12} "
          f"{'worst':>7} {'mean rank':>10}")
    for name, task, hits, n, mreg, wreg, mrank in summary:
        print(f"{name:<26} {task:>5} {f'{hits}/{n}':>14} {mreg:>12.3f} {wreg:>7.3f} {mrank:>10.2f}")
    print("\nRegret is in the task's own metric: circuit-wise accuracy for Circuit Decoding, "
          "character-level F1 for Situation Puzzles.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
