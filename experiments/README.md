# Experiments

> **This folder is the backing store.** The readable entry point is the folded tree in
> [`../notes/EXPERIMENTS_LOG.md`](../notes/EXPERIMENTS_LOG.md), where these same runs appear as
> A1–A8 (evaluation substrate) and B1–B6 (harness evolution) with dates, wall times and caveats.
> The numeric ids below are the folder names it links into.

One folder per experiment: what was run, with which model / protocol / split, the exact
command, timing, the result table, and the takeaway. Score files (`*.json`, figures) are
committed; raw per-episode traces (`*.jsonl`, EvoHarness `trajectories/`) are attached to the
GitHub Release named in each README.

| id | experiment | task model | headline |
|---|---|---|---|
| 01 | Stage-0 oracle and synthetic policies | none (code) | 18 queries suffice: greedy max-entropy solves 300/300, mean 11.6 queries |
| 02 | Judge calibration on TurtleBench-150 | — | Qwen2.5-7B 50.5%, Llama-3.1-405B-INT4 82.0% (= official 405B 82.7%), Opus 5 88.0% |
| 03 | Actionability ladder (7 hand-written harnesses) | Qwen2.5-7B | uniq@end (ours): state +18, decision table +30, decision +73 points over bare |
| 04 | Evolution run `open1` (regime open) | Qwen2.5-7B | proposer reaches the ceiling in one iteration: copy_query held-out: joint truth table correct 80.4%, uniq@end 99.7% |
| 05 | Evolution run `so1` (regime state_only) | Qwen2.5-7B | veto_only held-out: circuit-wise accuracy 91.3%, joint truth table correct 86.2%, uniq@end 96.7%; no information-gain computation, tokens = bare |
| 06 | EvoHarness-RL as baseline | Qwen2.5-7B | uniq@end (ours): ReAct 26-29, EvoHarness-Base (inline) 30-32, always-on panels 71-75 |
| 07 | Situation Puzzles bare with the official-grade judge | Qwen2.5-7B + 405B-INT4 | f1_char 0.621 (the earlier 0.659 was self-judged by the 7B) |
| 09 | Test-time harness evolution with label isolation + session audit (`strict6` / `labeled6`), then frozen evaluation of the finished harnesses on 200 held-out | same | strict stream 0/10 → 9/10 → 10/10 (cwa 67.8 / joint 63.3 prequential), every judge clean; on 200 instances the label-free state-only harness reaches joint 98.5 / cwa 99.5, bare 0 / 11.8 |
| 10 | Test-time harness evolution on AR-Bench Situation Puzzles (strict, 405B referee) | Qwen2.5-7B + Opus 5.5 | prequential F1 63.7 → 74.5 → 79.9; on 100 test puzzles the evolved harnesses reach F1 76–77 (bare 60.4), above every zero-shot model in AR-Bench's table |
| 11 | What the SP metric measures: F1 against answer length and question quality | — (re-analysis of 500 stored episodes) | answer length alone explains 65% of F1 variance (r=+0.806); key-question recall explains almost none (r=+0.050); the headline bare→evolved +15.2 shrinks to +0.4…+7.2 within a length band |

Common protocol for 03-06: Multi-Turn Puzzles Circuit Decoding, 300 released instances, 18
queries + final guess, environment wording from the paper's transcript, Qwen2.5-7B-Instruct
served with vLLM (temperature 0.7, 1200 output tokens), metrics from `runner/cd_score.py`.
Metric names follow the paper (see [`OFFICIAL_DEFINITIONS.md`](OFFICIAL_DEFINITIONS.md)):
`circuit-wise accuracy` = the benchmark's own metric (MTP Table 1), fraction of the three circuits
whose full truth table is right; `joint truth table correct` = the whole 24-bit joint truth table
is right (the paper's noun, our reading of a successful episode). Everything else is our
diagnostic: `uniq@end` = feasible set collapsed to one hypothesis after the 18 queries (information
gathering, luck-free); `eff` = ex-ante bits of the chosen query / best available query, averaged
over legal turns.
