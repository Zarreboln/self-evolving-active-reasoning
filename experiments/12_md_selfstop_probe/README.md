# 12 — Self-stop probe on iCraft-MD (2026-10-25)

**Not a reproduction of anything.** These numbers do not belong in a table that claims to follow
AR-Bench or BALAR, and the files are named `md_*_selfstop_probe.json` so they cannot be mistaken for
one. They were briefly tagged `_balar`, which was wrong on both counts below.

## What it measures

One question: **if the reasoner is allowed to decide when it has learned enough, does it ever say
so?** Every other stopping rule in this project is computed *about* the model rather than asked *of*
it:

| mechanism | who decides | signal | where |
|---|---|---|---|
| oracle confirms the answer | the environment | `same_pos == 4` | AR-Bench GN, `uot` and `proactive_cot` |
| candidate set collapses | the method's bookkeeping | `len(answer_set) <= 1` | AR-Bench `uot` on GN / SP / DC |
| belief entropy threshold | an explicit joint belief | MAP probability > 1 − α | BALAR |
| **the model says so** | **the model** | it replies `Answer: X` | **this probe only** |

AR-Bench has no fourth-row mechanism at all: grepping its three evaluators for a stop signal from
the model returns nothing, and its Situation Puzzles prompt forbids one outright — *"You should keep
asking me questions until I ask you to give your answer... without my permission, you should not
tell me your answer."* So this probe tests something the benchmark deliberately excludes.

## Setup

`runner/md_evaluator.py --allow_stop`, 10 questions, Qwen2.5-7B-Instruct at temperature 0.1 /
top_p 1.0, patient simulator `kimi-k3`, all 140 iCraft-MD instances, three prompt variants.
On any turn the reasoner may answer `Answer: X` instead of asking; the forced final turn is then
skipped and the turn it committed on is recorded.

## Result

| variant | accuracy | mean questions | nominally stopped early |
|---|---|---|---|
| zero_shot | 85/140 (60.7 %) | 9.1 | 120/140 |
| few_shot | 84/140 (60.0 %) | 8.4 | 137/140 |
| few_shot_inst | 82/140 (58.6 %) | 9.1 | 116/140 |

The high "stopped early" counts are an artefact of counting any commit before the forced turn.
Pooling all 420 episodes by the turn actually committed on:

| committed after | accuracy |
|---|---|
| 2 questions | 5/7 |
| 3 | 3/7 |
| 5 | 3/3 |
| 8 | 2/2 |
| **9** | **207/354** |
| 10 | 31/47 |
| **by turn 7 — genuinely early** | **11/17 (64.7 %)** |
| **turn 8–10 — used the budget** | **240/403 (59.6 %)** |

**Given the right to stop, the model essentially does not take it: 17 of 420 episodes, 4 %.** The
five-point gap between the early and late groups rests on 17 episodes, a standard error near 12
points, so it says nothing. What the probe does say is that a 7B asked "do you know enough yet?"
answers "one more question" almost every time — which is the behaviour a confidence-threshold rule
like BALAR's would have to overrule, and cannot be reproduced by asking.

## Why it is not a BALAR run

Two independent reasons, either one sufficient:

1. **The budget is wrong.** BALAR's paper is explicit: *"All methods receive the same maximum budget
   of 25 user-directed questions."* The 10 came from the repository README's example command, where
   `--max_ask_rounds` is a required flag with no default. Our 25-turn iCraft-MD runs are the ones
   that match BALAR's budget.
2. **The stopping mechanism is a different kind of thing** — row four of the table above against row
   three. BALAR stops on a computed belief; this asks the model.

## Files

`~/results/md_{zero_shot,few_shot,few_shot_inst}_selfstop_probe.json` on `bath`, shards under
`~/results/shards/md_*_selfstop_probe/`, scored by `runner/md_score.py`, commit turns broken out by
`runner/stop_vs_accuracy.py`.

A follow-up that would have isolated stopping from budget — 10 questions *without* the stop — was
started and stopped at 194 episodes because it is not a BALAR setting either; its partial output is
parked under `~/results/aborted/` rather than scored.
