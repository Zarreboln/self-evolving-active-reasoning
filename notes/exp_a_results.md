# Experiment A: results (2026-11-25)

Pre-registration: `exp_a_prereg.md`. Analysis: `multibench/mb/analyze_a.py`. Raw output on bath:
`~/results/mb/exp_a/analysis_a.txt`, `analysis_a_keep_errors.txt`, `analysis_a.json`.

**Run.** Job 11590, ml-16-node-001, 2× H100, lowprio. Started 13:11 UTC, preempted and requeued
13:42, resumed from disk, ended 14:24 UTC. About 73 min of running, ≈ 2.4 GPU-h; the Jev rerun of
the 20Q/SP/Guess Who seed baselines took 3 min of that. 1,450 A episodes. 8 were still errored at
the end (vLLM 400: prompt + max_tokens over the 64K context, all in long-reasoning arms) and were
dropped pairwise as pre-registered. Scoring them instead changes no conclusion (sensitivity run).

## Pre-registered verdict: **not passed**
- Sign part: **passed**. All 4 components have significant cells of both signs (needed ≥ 2).
- Structure part: **failed**. Split-half held-out MSE: G 0.0842, H 0.0729, L1 0.0306, L2 0.0302,
  S 0.0299. The low-rank models beat G and H in essentially every half (G 100%, H 99.5%) but tie
  the saturated model (L2 beats S in 41% of halves). The rule needed low-rank to beat S.

What the data rules out: one global component set (G), and the three dimension groups (H). What it
cannot tell apart at 10 tasks: a low-rank shared structure from fully per-task gains.

## Gains over seed (Δ, 95% bootstrap CI; + / − = significant)
| cell | seed | belief | plan | check | brief |
|---|---|---|---|---|---|
| gn | 0.15 | +0.85+ | +0.85+ | +0.85+ | +0.38+ |
| dc | 0.37 | −0.10 | −0.20 | −0.13 | −0.07 |
| mastermind basic | 0.60 | +0.36+ | +0.01 | −0.14− | +0.01 |
| hangman basic | 0.56 | −0.09 | +0.01 | +0.06 | +0.05 |
| 20Q basic-things | 0.40 | 0.00 | −0.10 | +0.13 | 0.00 |
| wordle basic5 | 0.34 | +0.34+ | +0.12 | +0.03 | +0.15+ |
| blicket n5 | 0.91 | +0.05+ | −0.09− | +0.02 | −0.10− |
| sciworld conductivity | 0.53 | −0.41− | −0.27− | −0.33− | +0.01 |
| alfworld pick&place | 0.77 | +0.23+ | +0.13 | +0.20+ | +0.10 |
| babyai GoToLocal | 0.63 | +0.30+ | +0.15 | +0.30+ | −0.11 |

## Caveats found while reading the episodes (not in the pre-registration)
1. **The first factor is "does eliciting reasoning help here".** With thinking off, the seed
   barely reasons on several tasks: the median seed reply is 12 characters on GN, 15 on Wordle,
   21 on ALFWorld, 44 on ScienceWorld. On GN it answers `Action: 1234` and repeats earlier
   guesses. Any component that asks for written content makes the model reason, so belief, plan
   and check all move GN from 0.15 to 1.0. The rank-1 component loadings all have the same sign
   (belief −0.59, plan −0.56, check −0.55, brief −0.20) and GN alone carries a task loading of
   −0.80. Singular values of the interaction: 1.65, 0.43, 0.33, 0.11.
2. **ScienceWorld drops are premature `focus`.** Focus is irreversible there. Reasoning arms
   focus on a wrong object early and the episode ends at 0. The seed explores for ~78 steps first.
3. Post-hoc, without GN (not pre-registered): G 0.0403, H 0.0320, L1 0.0300, L2 0.0298, S 0.0307.
   Low-rank comes out best, but its margin over H is small next to the spread across halves
   (sd ≈ 0.009).

## Proposed next steps (need approval)
- A `think` control arm: free reasoning with no structure, ~290 episodes, < 1 GPU-h. Δ is then
  measured against `think`, so a component's content is separated from reasoning elicitation.
- More task columns (~20 subtasks). With 10 columns the saturated model costs too few extra
  parameters to lose to a low-rank one; L vs S needs more tasks, not more instances.
