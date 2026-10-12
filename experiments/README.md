# Experiments

One folder per experiment: what was run, with which model / protocol / split, the exact
command, timing, the result table, and the takeaway. Score files (`*.json`, figures) are
committed; raw per-episode traces (`*.jsonl`, EvoHarness `trajectories/`) are attached to the
GitHub Release named in each README.

| id | experiment | task model | headline |
|---|---|---|---|
| 01 | Stage-0 oracle and synthetic policies | none (code) | 18 queries suffice: greedy max-entropy solves 300/300, mean 11.6 queries |
| 02 | Judge calibration on TurtleBench-150 | — | Qwen2.5-7B 50.5%, Llama-3.1-405B-INT4 82.0% (= official 405B 82.7%), Opus 5 88.0% |
| 03 | Actionability ladder (7 hand-written harnesses) | Qwen2.5-7B | state +18, decision table +30, decision +73 points of uniq@end over bare |
| 04 | Evolution run `open1` (regime open) | Qwen2.5-7B | proposer reaches the ceiling in one iteration: copy_query 99.7% / exact 80.4% held-out |
| 05 | Evolution run `so1` (regime state_only) | Qwen2.5-7B | veto_only 96.7% / exact 86.2% held-out, no information-gain computation, tokens = bare |
| 06 | EvoHarness-RL as baseline | Qwen2.5-7B | ReAct 26-29, EvoHarness-Base (inline) 30-32, always-on panels 71-75 |
| 07 | Situation Puzzles bare with the official-grade judge | Qwen2.5-7B + 405B-INT4 | f1_char 0.621 (the earlier 0.659 was self-judged by the 7B) |

Common protocol for 03-06: Multi-Turn Puzzles CircuitDecoding, 300 released instances, 18
queries + final guess, environment wording from the paper's transcript, Qwen2.5-7B-Instruct
served with vLLM (temperature 0.7, 1200 output tokens), metrics from `runner/cd_score.py`.
`uniq@end` = feasible set collapsed to one hypothesis after the 18 queries (information
gathering, luck-free); `exact` = the final 24-bit guess is right (the task's success metric);
`eff` = ex-ante bits of the chosen query / best available query, averaged over legal turns.
