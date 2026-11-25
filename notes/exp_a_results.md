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

## Amendment 1 result: few-shot prediction of a held-out task (spec in exp_a_prereg.md)
Held-out MSE of the predicted 4 gains. Episodes spent on the new task: G/H 0; S, L1, L2 5k; L1-probe 2k (seed + belief only).

| k | G | H | S | L1 | L2 | L1-probe |
|---|---|---|---|---|---|---|
| 1 | 0.094 | 0.100 | 0.221 | 0.149 | 0.169 | 0.167 |
| 2 | 0.094 | 0.101 | 0.113 | 0.083 | 0.092 | 0.103 |
| 3 | 0.095 | 0.101 | 0.082 | 0.066 | 0.071 | 0.082 |
| 5 | 0.095 | 0.102 | 0.049 | 0.046 | 0.048 | 0.062 |
| 8 | 0.097 | 0.103 | 0.037 | 0.039 | 0.039 | 0.052 |
| 12 | 0.099 | 0.105 | 0.031 | 0.037 | 0.037 | 0.049 |

Without GN: G 0.038 at 0 episodes. S 0.119 / 0.082 / 0.053 / 0.038 / 0.032 at k = 2 / 3 / 5 / 8 / 12. L2 reaches 0.029 at k = 12. L1-probe: 0.062 at 4 episodes, 0.047 at 6, 0.036 at 10.

Reading:
- **At the same number of episodes, the shared structure predicts better than starting per task from scratch.**
  - All tasks: L1-probe has 0.062 at 10 episodes; S has 0.113 at 10 and 0.082 at 15. The two meet at about 24–25 episodes (0.049).
  - At the same k, L1 beats S up to k = 5. From k = 8 on, S wins.
- **Without GN, the global mean (0 episodes, 0.038) is about as good as anything below ~40 episodes.** At these sample sizes, most per-task variation outside GN is noise. The task-specific signal is concentrated in a few cells.
- The reasoning-elicitation confound from the first analysis still applies to all of this.

## think control (job 11607, 2026-11-25 14:46–15:16 UTC, 1× H100, ~0.5 GPU-h; post-hoc arm, see strategies.py)
`think` = seed + free reasoning in a <reasoning> block, nothing else. Its scores: gn 1.00, alfworld 1.00, mastermind 0.95, blicket 0.94, babyai 0.80, wordle 0.65, hangman 0.61, 20Q 0.47, sciworld 0.16, dc 0.13.
So most of the seed-relative gains in the first table were reasoning elicitation: think alone takes gn 0.15→1.00 and alfworld 0.77→1.00. Reasoning also hurts sciworld (0.53→0.16, premature focus) and dc (0.37→0.13).

Gains against think (same pre-registered analysis, `--baseline think`):
- belief: no significant cell anywhere. Its earlier gains were all reasoning.
- plan and check: significantly negative on mastermind/wordle (plan also on blicket), positive nowhere.
- brief (less reasoning): +0.38 on sciworld; negative on gn, mastermind, blicket, alfworld and babyai.
- Sign part: only brief has both signs → **fails** (1/4). Structure part: L2 0.0258 < S 0.0263 < L1 0.0301 < H 0.0328 < G 0.0348; L2 beats G in 86% of halves, H in 85%, S in 60% → **passes**. Overall not passed.
- Rank-1 interaction = "how much reasoning": component loadings brief 0.76, check 0.53, plan 0.38, belief −0.01. Tasks that want less reasoning are sciworld (+0.57) and dc (+0.38); tasks that want more are mastermind (−0.62), wordle and gn.
- Few-shot against think: the global mean (0 episodes) 0.033 is the best predictor up to ~25 episodes. L2 beats S at every k (k=5: 0.036 vs 0.044; k=12: 0.0245 vs 0.0256), but only by a little. The belief-only probe is undefined: belief's loading is ~0, so the probe divides by zero (the table's 10^3 values).

Reading: with the reasoning confound removed, four hand-written prompt components show one real axis (reasoning amount, which tasks want in opposite directions) and little else. They are too coarse to test partial sharing. Structure, if any, has to come from task-specific components that B evolves, e.g. a gn enumerator or a sciworld focus guard. Recommendation: B starts from `think`, not from the thinking-off seed.
