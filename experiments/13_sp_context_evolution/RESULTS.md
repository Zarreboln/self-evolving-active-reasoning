# Results tables

100 official SP test puzzles. Each number is a paired difference: for every puzzle we average over seeds, take the
difference between the two methods, and bootstrap over puzzles (10,000 resamples) for the 95% interval.
"Lenient" = Claude Haiku 5.5, "strict" = Gemini 3.7 Flash.

## 1. Answer-step memory (story score, 0 to 6)

Compared with resampling the final answer from the same transcript without examples (`plain7b`). Same frozen 7B, same
transcript, 5 seeds.

| Condition | Lenient | Strict |
|---|---|---|
| 2 similar past puzzles + true story (`ex7b`) | +0.254 [+0.160, +0.354] | +0.174 [+0.082, +0.264] |
| 2 similar puzzles, questions asked by a stronger model, + story (`tex7b`) | +0.250 [+0.144, +0.370] | +0.198 [+0.094, +0.308] |
| 2 random past puzzles + true story (`ex7b_rand`) | +0.156 [+0.084, +0.234] | +0.102 [+0.034, +0.174] |
| near variants removed (`ex7b_far`) | +0.126 [+0.022, +0.226] | +0.078 [-0.014, +0.164] |
| list confirmed facts first, then write (`facts7b`) | +0.140 [+0.064, +0.224] | +0.086 [+0.010, +0.162] |
| 2 similar puzzles, Q&A only, no story (`ex7b_nostory`) | -0.002 [-0.082, +0.082] | -0.032 [-0.110, +0.050] |

Mean story score: `plain7b` 0.78 lenient / 0.49 strict; `ex7b` 1.04 / 0.67.

## 2. Question recall vs. the empty playbook

| Method | Lenient | Strict |
|---|---|---|
| Strong-model playbook, no experience (`a2`) | -0.007 [-0.044, +0.031] | -0.059 [-0.100, -0.020] |
| Evolved: hidden-story prior (`prior`) | +0.008 [-0.039, +0.056] | +0.011 [-0.028, +0.049] |
| Evolved: step-level credit (`cce`) | +0.014 [-0.024, +0.053] | -0.032 [-0.067, +0.004] |
| Evolved: step credit, shuffled (`cce_shuf`) | +0.014 [-0.022, +0.050] | -0.041 [-0.077, -0.004] |
| Evolved: success vs. failure (`traj`) | +0.029 [-0.007, +0.066] | -0.007 [-0.041, +0.027] |
| Evolved, selected by strict judge (`traj_gem`) | -0.008 [-0.034, +0.019] | -0.023 [-0.049, +0.002] |
| Evolved, selected by both judges (`traj_cons`) | +0.024 [-0.016, +0.064] | -0.020 [-0.056, +0.017] |
| ACE (`ace`) | -0.016 [-0.054, +0.021] | -0.050 [-0.083, -0.017] |
| Dynamic Cheatsheet (`dc`) | +0.001 [-0.031, +0.034] | -0.014 [-0.053, +0.025] |
| Best-of-8 written into rules (`bo8`) | -0.046 [-0.082, -0.009] | -0.071 [-0.110, -0.029] |
| Strong-model examples in the prompt (`tex`) | +0.019 [-0.018, +0.056] | +0.005 [-0.034, +0.043] |
| Best-of-8 at test time (`r1`, ~5x calls) | +0.064 [+0.032, +0.096] | +0.031 [-0.004, +0.066] |

BALAR (1 seed, about 297 calls per episode): recall 0.355 vs. 0.386 for the empty playbook (lenient).

## 3. Story score of full episodes, strict judge

| Method | Mean | vs. empty playbook |
|---|---|---|
| Empty playbook (`a0`) | 0.417 | |
| Best-of-8 at test time (`r1`) | 0.557 | +0.140 [+0.003, +0.273] |
| Strong-model examples in prompt (`tex`) | 0.595 | +0.173 [-0.013, +0.347] |
| Similar stories in prompt (`tex_sto`) | 0.760 | +0.367 [+0.180, +0.557] |
| Evolved, selected by both judges (`traj_cons`) | 0.460 | +0.043 [-0.100, +0.180] |

`tex_sto` looks strong, but almost all of its gain is on the third of puzzles with the closest TRAIN story (+0.94),
with none on the other two thirds (+0.11 and -0.04). We count it as near-variant retrieval, not a general gain.

## 4. Asking vs. using (test, seed 0, story score)

| Who asks | Who writes | Lenient | Strict |
|---|---|---|---|
| 7B | 7B | 0.77 | 0.50 |
| 7B | Qwen3-235B | 1.21 | 0.92 |
| 7B | Claude Opus 5.5 | 2.06 | 1.56 |
| Qwen3-235B | 7B | 1.82 | 1.33 |
| Qwen3-235B | Qwen3-235B | 2.19 | 1.73 |

## 5. Judge check

On 100 items graded by Claude Opus 5.5 as a reference: Gemini agreed on 91 of 98, Haiku on 80 of 100. Where the two
judges disagreed, the reference sided with Haiku on 33 to 50% of Haiku-only credits and with Gemini on 77 to 84% of
Gemini-only credits.
