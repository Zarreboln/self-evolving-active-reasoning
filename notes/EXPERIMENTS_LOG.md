# Reference: official task definitions, metrics and baselines

<details>
<summary><b>Situation Puzzles · Guess Numbers (AR-Bench) · judge reference points</b></summary>

<dl><dd>

> Sources: the AR-Bench (ICML 2025) paper, checked line by line on 2026-10-12, and its released code.
> [`experiments/OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md) is the reference
> for what each metric means and what the paper leaves undefined. We use the official metric names
> throughout. Anything we added is marked (ours) and kept out of the columns that hold official
> metrics. Circuit Decoding baselines are in the appendix.

#### 1. Official baselines, Situation Puzzles (AR-Bench Table 6, zero-shot, character-level F1)

| model | F1 (char) |
|---|---|
| GPT-4o | 63 |
| Llama-3.1-405B | 55 |
| Llama-3.1-70B | 54 |
| Qwen-2.5-7B | 51 |
| QwQ-32B | 49 |
| Llama-3.1-8B | 43 |
| GPT-4o-mini | 41 |

AR-Bench defines three official metrics: accuracy for Detective Cases, character-level F1 for
Situation Puzzles (`arbench.utils.utils_sp.f1_score`) and exact match for Guess Number. Its official
referee is Llama-3.1-405B. Key-question recall is our addition, not an AR-Bench metric.

#### 2. Official baselines, Guess Numbers (AR-Bench Table 6, zero-shot, exact match)

| model | exact match |
|---|---|
| GPT-4o | 35 |
| Llama-3.1-405B | 12 |
| Llama-3.1-70B | 8 |
| GPT-4o-mini | 6 |
| Qwen-2.5-7B | 4 |
| QwQ-32B | 1 |
| Llama-3.1-8B | 0 |
| Qwen-2.5-3B | 0 |

The secret is a 4-digit number with distinct digits (5,040 candidates, leading zeros allowed in the
test set). Feedback comes from a rule-based oracle, not an LLM, and the game runs 25 rounds. In the
official zero-shot loop only the extra final guess after round 25 is scored; a correct guess
earlier in the game does not count on its own (see A6).

#### 3. Judge reference points (TurtleBench-150, seed 7, zero-shot)

| judge | accuracy | source |
|---|---|---|
| Claude Opus 5 (one fresh context per case) | 88.0 | ours, `labels_opus.json` |
| Llama-3.1-405B bf16 | 82.7 | TurtleBench's archived predictions |
| **Llama-3.1-405B-INT4 (in use)** | **82.0 / 83.3 on two runs** | ours; the gap is sampling at temperature 0.7 |
| Llama-3.1-70B (2-shot) | 74.0 | archived predictions |
| GPT-4o (2-shot) | 63.3 | archived predictions |
| Qwen2.5-7B-Instruct | 50.5 | ours |

We use TurtleBench because AR-Bench used it to cross-check its own LLM judges (its Table 13), and
because AR-Bench's SP answerer prompt was adapted from TurtleBench's referee prompt: the same
three-way labels and the same two few-shot examples.

</dd></dl>

</details>

> ---

# Experiment log

<details>
<summary><b>Index (id · old id · date · wall time · headline) and recording conventions</b></summary>

<dl><dd>

| id | old id | date | wall | headline |
|---|---|---|---|---|
| A1 | A1 | 10-12 | — | **Official definitions fixed** for Situation Puzzles (character-level F1, 405B referee) and Guess Numbers (exact match, rule-based oracle feedback). Key-question recall is ours. Circuit Decoding definitions are in the appendix |
| **A2** | A3 | 10-10–10-14 | see children | **Referee calibration on TurtleBench-150.** We keep the official Llama-3.1-405B-INT4 (83.3). Compared case by case, none of the eleven candidates can replace it |
| A2.1 | A3.1 | 10-13 | ~10 min each, 1×H200 | Eight candidates with thinking off: Qwen3.8-Flash-Next 79.3, Qwen3-235B 74.0, Qwen3-32B 74.0, Llama-3.3-70B 73.3, Qwen3.6-35B-A3B 72.7, Qwen3.8-27B 64.0, gpt-oss-120b 62.7 |
| A2.2 | A3.2 | 10-13 | 10 min ×2 | With thinking on, Flash-Next goes from 79.3 to 80.7 and Qwen3.8-27B from 64.0 to 78.7. Neither reaches 83.3 |
| **A2.3** | A3.3 | 10-13 | 12 s, $0.0041 | **Jev 1.13 (TypeSafe's closed, hosted "System One" model) scores 78.7.** Its Unknown recall is 36.8 against the 405B's 26.3, and it is the only candidate that returns a usable probability distribution: accuracy rises steadily with its confidence, from 50.0% below 0.2 to 95.6% above 0.8. It cannot simply replace the 405B: they agree on 80.7% of cases, and it fixes 10 of the 405B's 25 errors but gets 17 of its 125 correct answers wrong. It is also not deterministic; five identical runs gave 78.7 / 80.0 / 79.3 / 80.0 / 78.0 |
| A2.4 | A3.4 | 10-13 | 12 s each | Jev's three documented prompting options (structured criteria, question splitting, both) each give up 3–5 points of accuracy for 10–25 points of Unknown recall. Thresholding the probabilities gives the same trade without a second request |
| **A2.5** | A3.5 | 10-13 | — | **Macro-averaging reverses the ranking.** The gold labels are skewed (C 55 / I 76 / U 19), so a judge that never answers Unknown loses very little accuracy. Ranked by macro recall, Flash-Next with thinking (79.9) comes first and the 405B (69.6) fifth. In the SP pilot the referee answered Unknown to 13.8% of 4,500 questions, so about one question in seven falls in the class the 405B handles worst |
| **A2.6** | A3.6 | 10-14 | ~1 min per model | **Reading the label probabilities never changes the verdict**; it agrees with greedy decoding to within one point and only exposes the distribution. Whether that distribution is calibrated depends on the model, not on how it is read: ECE is 0.56 for Qwen2.5-7B, 0.16 for Qwen3-32B and 0.055 for Qwen3.8-27B, whose macro recall (69.3) matches the 405B's 69.6. The 7B has one specific problem: it answers Unknown on 99 of 150 cases, and forbidding Unknown raises it from 36.0 to 68.0 |
| A3 | A4 | 10-10 | 7 min | **Situation Puzzles with no harness, scored by the official-grade referee**: F1 0.621. |
| **A4** | A7 | 10-21 | — | **What the SP metric actually measures** (re-analysis of 500 stored episodes, no new inference). Answer length alone explains 65.0% of the variance in character-level F1 (r = +0.806). Key-question recall gives r = +0.050, and +0.068 once length is controlled. Mean F1 keeps rising with length and never turns down, so in practice there is no length penalty. With length controlled, the bare→final gain of +15.2 shrinks to between +0.4 and +7.2. The task also never checks that the final answer follows from the recorded dialogue, so an agent can skip the reasoning steps; we have not yet measured the degenerate baseline. |
| **A5** | A8 | 10-15–21 | — | **Independent preregistered replication by Yangzi Sun.** Recorded dialogues were held fixed and paired with two answer writers: 65 cases, 260 generations, 520 blind ratings, plus two new ratings, Semantic Correctness and Evidence Faithfulness. F1 and semantic correctness rank answers in opposite order 82 times across 29 cases, and the evidence-constrained writer did not improve semantic correctness (−0.0308 [−0.0577, −0.0058]). We disagree with two of its readings: it evaluates the first generation, whose recall we had already shown to be flat, and its own data shows the evolved dialogue doubling the number of fully correct answers (15/65 against 7/65) even though the ordinal mean goes the other way |
| **A6** | — | 10-21 | — | **Code and data audit of the official GN and SP evaluators** (no model calls). GN: the zero-shot loop scores only the extra final guess; ToT picks among candidate guesses by comparing them with the gold number; UoT filters candidates in Python and fills in the answer; guesses are not validated (repeated digits accepted, the last 4-digit run in the text is taken), and an output with no number raises an uncaught error. SP: another puzzle's bottom scores 0.833 character-level F1; a 137-word string of common function words scores word-level F1 0.349 |
| A7 | — | pending | pending | **Guess Numbers without a harness** under the official loop. *Not yet run* |
| B1 | — | pending | pending | **Context evolution on Guess Numbers.** *Not yet run* |
| B2 | — | pending | pending | **Context evolution on Situation Puzzles.** *Not yet run* |
| **C1** | B4 | 10-13 05:03–07:17 | 2 h 14 | **Test-time harness evolution on Situation Puzzles** (strict, 405B referee). Prequential F1 goes 63.7 → 74.5 → 79.9 over the 30-puzzle stream. On all 100 test puzzles (70 of them never seen by the stream), bare scores 60.4 and the evolved harnesses 73.0–76.7. That is above every zero-shot model in AR-Bench Table 6, including GPT-4o at 63, and the 7B still writes every question. The cost is 3–8× more output tokens; `b2r0g1`, which was not chosen, does as well at half the tokens. Read this together with A4 |
| **C2** | B5 | 10-13 | — | **Checking the TTHE judge without an external benchmark.** Each run stores the gold scores of the candidate pool, which the judge never sees. `strict6` picks a best-scoring candidate 2/3 times (mean regret 0.011), `labeled6` 3/3 (0.000) and `sp_strict1` 1/3 (mean regret 0.063, worst 0.114). The Circuit Decoding pools are saturated, so the check says little there; on Situation Puzzles the judge is clearly noisy |
| C3 | — | pending | pending | **Test-time harness evolution on Guess Numbers.** *Not yet run* |
| CD1 | A2 | 10-06 | pending (CPU) | **The 18-query budget is enough.** The greedy max-entropy oracle solves 300/300 using 11.60 queries on average. Greedy over 4 sampled queries solves 250/300, a random fresh entry 156/300, a sequential scan 58/300 and uniform random 35/300. Later failures are not a budget problem |
| CD2 | A5 | 10-10/11 | pending | **Actionability ladder, seven hand-written harnesses** (cwa / joint / uniq@end): bare 12.9 / 0.5 / 23.2, symbolic 20.5 / 2.2 / 41.2, enum15 23.7 / 5.8 / 41.8, **T3-gated 31.3 / 10.2**, splits 22.3 / 5.7 / 53.7, oracle hint 27.3 / 6.8 / 96.5. Better questioning does not turn into better answers: the oracle hint almost saturates uniq@end at 96.5, but joint accuracy stays at 6.8, so the bottleneck is the final inference, not the questions |
| CD3 | A6 | 10-11 | 1 h 08 | **EvoHarness-RL as an external baseline**: env_only (ReAct) 4.3/4.9, inline (the model calls track/commit itself) 5.7/6.9, always_on (a STATE panel every turn) 74.8/79.0 with joint 62.0/65.0. With the same information available, showing the state every turn works far better than asking the model to keep it, which agrees with CD2 |
| CD4 | B2 | 10-10 21:22 – 10-11 ~01:00 | ~3 h 38 | One-shot evolution `so1` (regime state_only: no information gain, no split counts, no recommended queries). `veto_only` scores **91.3 / 86.2** with uniq@end 96.7, at the same token cost as bare. This is the main claim in its simplest form: take away everything that computes for the model, keep the state and the constraints, and the score still goes up |
| **CD5** | B3 | 10-12 19:01–22:33 | 3 h 32 | **Test-time evolution with label isolation.** The strict stream goes 0/10 → 9/10 → 10/10. 29 of 32 sessions were clean, including every judge session. Frozen evaluation on 200 held-out instances: bare 11.8 / 0.0, the label-free state-only harness **99.5 / 98.5** (98.2 on the unseen split), the labeled stream 100/100 at 215 tokens per episode. The judge rejected three harnesses that scored 10/10 because they gave the model a menu; the model has to write every query itself |

> **What this log is for.** Each experiment should be traceable from its conditions (task, model,
> regime, signals, budget, split) to the script, its input, its output and log, and the result.
>
> **Markers.** 📁 local path, verified · ☁️ on a server (Engaging `~/results/ar_harness`, H100
> `/home/tmp_l`) · 🔒 private to the controller (`~/.ar_harness_controller/<run>/`, never committed).
>
> **How entries are recorded.** Each formal experiment lists its date and wall time before its name.
> Dates are written `YYYY-MM-DD`; a span of days is `MM-DD–MM-DD`, and a repeated run gets two dates.
> Clock times go in the experiment's own README, not in this index. Times come from the start and end
> of the log; where there is no log we use file modification times and say so. If there is no
> reliable record, the entry says `pending` rather than an estimate. The last column of every results
> table points to the script and the log that produced that row.
>
> 🔗 Each leaf ends with a traceability block listing upstream scripts, artefact paths and batch
> logs. It does not repeat the per-row pointers.

> **Three parts, with Guess Numbers and Situation Puzzles as the main tasks.**
> - **A, the evaluation substrate (A1–A7):** official definitions, referee calibration, baselines
>   without a harness, an audit of the SP metric, an independent replication, and an audit of the
>   official evaluator code. This part tells us whether our numbers mean anything. A4's answer is
>   that the official SP F1 mostly measures answer length, so the main tables should be read
>   alongside it.
> - **B, context evolution (B1–B2):** the evolved object is the context the frozen questioner reads
>   (system prompt, instructions, accumulated notes). The episode loop, the parser and the
>   environment stay fixed.
> - **C, harness evolution (C1–C3):** the evolved object is the harness, the code around the model
>   that decides what it is shown, what it is asked again and what is refused.
> - **Appendix, Circuit Decoding (CD1–CD5):** the earlier testbed, where the harness-evolution loop
>   and label isolation were first developed. Kept for reference and for the label-free evidence.
>
> Old ids from earlier versions of this log are listed in the index.
>
> **Metric definitions (used the same way throughout).**
> - **circuit-wise accuracy (appendix):** Circuit Decoding's only official metric, the fraction of the three
>   circuits whose full truth table is correct.
> - **joint truth table correct (appendix):** the whole 24-bit joint table is correct. This is the paper's term
>   under our literal reading.
> - **exact match:** the official Guess Numbers metric, whether the scored guess equals the secret
>   number.
> - **f1_char:** the official Situation Puzzles metric, character-level F1 against the hidden bottom
>   (`arbench.utils.utils_sp.f1_score`).
> - **key-question recall (ours):** the share of a puzzle's key questions that the referee judges
>   were hit. A process metric, not an official one.
> - **uniq@end, eff, zero%, oracle%, rep% (ours):** diagnostics for analysing the mechanism. Never
>   reported next to an official metric.
> - **prequential:** the score of the incumbent harness on a new batch before it adapts, i.e. what
>   deployment would see. **transductive:** the score of the committed harness on that same batch,
>   i.e. how well it fits. We always report the two separately.
> - **ECE:** expected calibration error over ten equal-width bins. **macro recall:** the unweighted
>   mean of the three per-class recalls, which corrects for the skewed gold distribution.
>
> **What we can and cannot claim.** The method is not training-free. It is not data-free either,
> since it consumes a stream of instances. The accurate description is label-free (transductive),
> and the evidence for it is that the strict stream matches the labeled one. A data-free claim would
> need the task family to be generated by the system itself.

**Data and judge assets**

| item | path | note |
|---|---|---|
| Circuit Decoding instances | ☁️ `~/code/ar_harness/mt_puzzles/` | 300 released instances, 18 queries then the joint table |
| Guess Numbers instances | ☁️ `~/code/ar_harness/AR-Bench/data/gn/` | official split, train 4,940 / test 100 |
| Situation Puzzles instances | ☁️ `~/code/ar_harness/AR-Bench/data/sp/` | official split; bottoms and key questions are controller-private |
| TurtleBench | ☁️ `~/TurtleBench` (both machines) | 1,531 English cases; a fixed sample of 150 (seed 7) across 32 stories |
| Per-case judge predictions | 📁 [`experiments/02_judge_calibration/turtle150_predictions/`](../experiments/02_judge_calibration/turtle150_predictions/) | eleven candidates plus Jev and the typed read, with probabilities where available |
| Controller-private records | 🔒 `~/.ar_harness_controller/<run>/` | solve jobs, full optimisation log, lineage. **Never committed** |
| Raw traces | GitHub Release `v0.1-experiments-2026-10-11` | `cd_tthe_isolated_raw`, `cd_tthe_pilots_raw`, `sp_tthe_pilot_raw` |

</dd></dl>

</details>

> ---

# Part A: the evaluation substrate

<details>
<summary><b>A1 Official definitions and names (10-12)</b></summary>

<dl><dd>

Before any number here can be compared with a published one, it has to be computed with the metric
the benchmark defines. This entry pins that down, and the result is
[`experiments/OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md).

**Situation Puzzles.** The official metric is character-level F1 against the hidden bottom
(`arbench.utils.utils_sp.f1_score`). The official referee is Llama-3.1-405B, and the protocol is 25
yes/no questions followed by a written answer. Key-question recall is a process metric we added; we
report it next to F1, never instead of it.

**Guess Numbers.** The official metric is exact match on a 4-digit secret with distinct digits.
Feedback is given by a rule-based oracle (digits in the right position, digits in the wrong
position), and the game runs 25 rounds. In the official zero-shot loop, the score is taken from one
extra final guess requested after round 25.

**Circuit Decoding** is now covered only in the appendix; its definitions are unchanged and remain in
`OFFICIAL_DEFINITIONS.md`.

**Standing rule.** Tasks, metrics and baselines are reported with their official definitions and
names. We can point out their limitations, as [A4](#), [A5](#) and [A6](#) do, but we do not swap in
a metric of our own in the reported tables.

<details>
<summary>🔗 Traceability (A1)</summary>

<dl><dd>

- Definitions: 📁 [`experiments/OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md)
- Scorers carrying the official keys: [`runner/cd_score.py`](../runner/cd_score.py) ·
  [`runner/cd_summarize.py`](../runner/cd_summarize.py) ·
  [`runner/cd_score_evoharness.py`](../runner/cd_score_evoharness.py)
- Index of all runs under these names: 📁 [`experiments/README.md`](../experiments/README.md)

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A2 Referee calibration on TurtleBench-150</b></summary>

<dl><dd>

> This section calibrates the **referee**, the model that answers the player's questions in
> Situation Puzzles. That is a different model from the judge inside the evolution loop, which
> decides which harness to commit and is checked separately in [C2](#). The paper should keep the
> two names apart.

<details>
<summary>A2.1–A2.2 Eleven candidates, thinking off and on (10-13)</summary>

<dl><dd>

Accuracy first, then recall for each class in the order Correct / Incorrect / Unknown (gold counts
55 / 76 / 19).

| candidate | job | thinking off | C/I/U | thinking on | C/I/U |
|---|---|---|---|---|---|
| **Llama-3.1-405B-INT4 (in use)** | 25698041 | **83.3** | 89.1 / 93.4 / 26.3 | n/a | |
| Qwen3.8-Flash-Next-FP8 | 25718457 / 25724076 | 79.3 | 90.9 / 81.6 / 36.8 | **80.7** | 90.9 / 75.0 / 73.7 |
| Qwen3.8-27B-FP8 | 25718456 / 25724077 | 64.0 | 74.5 / 59.2 / 52.6 | **78.7** | 85.5 / 77.6 / 63.2 |
| Qwen3-235B-A22B-2507-FP8 | 25718459 | 74.0 | 83.6 / 69.7 / 63.2 | not run | |
| Qwen3-32B | 25698272 | 74.0 | 80.0 / 81.6 / 26.3 | not run | |
| Llama-3.3-70B-FP8-dynamic | 25698273 | 73.3 | 76.4 / 84.2 / 21.1 | not run | |
| Qwen3.6-35B-A3B-FP8 | 25718458 | 72.7 | 87.3 / 69.7 / 42.1 | not run | |
| gpt-oss-120b | 25698271 | 62.7 | 61.8 / 65.8 / 52.6 | not run | |

Every candidate does badly on Unknown, which is the class that decides whether a question can be
answered from the story at all. The 405B's lead comes mostly from Incorrect (93.4; no other candidate
is above 85), and its own Unknown recall of 26.3 is the second lowest. Turning thinking on recovers
exactly that class for the Qwen3.8 models (Flash-Next 36.8 → 73.7), though not enough to overtake
the 405B.

Compared case by case with the incumbent (125 correct, 25 wrong), the strongest candidate,
Flash-Next with thinking, fixes 13 of the 25 errors and gets 17 of the 125 correct answers wrong.
The others do worse. None of them can simply replace the 405B, and switching referees would shift
the Situation Puzzles numbers for reasons that have nothing to do with the harness, so we keep the
official 405B.

⚠️ Serving notes. FP8 on H200 needs `VLLM_USE_DEEP_GEMM=0`, because the DeepGEMM JIT fails against
the nvcc 13.4 headers. Flash-Next needs `max_num_seqs<=57` for its Mamba cache. The 405B-INT4 needs
pipeline parallelism, since tensor parallelism triggers a FlashInfer JIT build. Its
`generation_config` also leaves out `<|eot_id|>`, so `stop_token_ids [128001, 128008, 128009]` has
to be passed explicitly.

</dd></dl>
</details>

<details>
<summary>A2.3–A2.4 A closed typed-decision model as referee: Jev 1.13 (10-13)</summary>

<dl><dd>

Jev takes a `state` and typed `questions` and returns a typed answer with probabilities
(`POST /v1/systemone`). That matches the shape of this task, so it was worth one run. We copied the
three TurtleBench judging rules verbatim into the question's `instructions` and used the three labels
as its `criteria`. ⚠️ Jev does not accept a system message, so the official prompt cannot be sent
byte for byte. This is the one respect in which the comparison is not like for like.

| | Jev 1.13 | Llama-3.1-405B-INT4 |
|---|---|---|
| accuracy | 78.7 | 83.3 |
| recall C / I / U | 80.0 / 88.2 / **36.8** | 89.1 / 93.4 / 26.3 |
| wall for 150 cases | **12 s** (16 concurrent) | ~15 s on 2×H200 after a 270 s load |
| cost | **$0.0041** (98,480 input tokens at $0.042/MTok) | an H200 pair for the run |

It is the only candidate whose verdicts come back as a distribution, and its confidence
(`1 − H(p)/ln K`) tracks accuracy monotonically:

| confidence | n | accuracy |
|---|---|---|
| 0.0–0.2 | 12 | 50.0% |
| 0.2–0.4 | 26 | 61.5% |
| 0.4–0.6 | 37 | 73.0% |
| 0.6–0.8 | 30 | 86.7% |
| 0.8–1.0 | 45 | 95.6% |

None of the vendor's own prompting options helps (A2.4). Structured `what` / `not_for` /
`examples` criteria give 74.0; splitting the decision into a decidability `noul` plus a binary
choice gives 74.7; using both gives 76.0, with the best macro recall at 70.9. Each buys 10–25 points
of Unknown recall for 3–5 points of accuracy, which is the same trade that thresholding the
probabilities already gives without an extra request.

⚠️ Jev is not deterministic. Five identical requests returned 78.7 / 80.0 / 79.3 / 80.0 / 78.0, with
Unknown recall ranging from 26.3 to 36.8, and the per-case probabilities moved as well. A single run
cannot resolve differences smaller than about two points.

We did not adopt it, for three reasons. Case by case it does not match the 405B. AR-Bench's referee
is the 405B, and AR-Bench reports accuracy. And it is a closed hosted service, so every story and
every guess would leave our infrastructure. It would fit best as a cheap filter, or as a second
opinion whose confidence decides which verdicts to re-check.

</dd></dl>
</details>

<details>
<summary>A2.5 Macro-averaging reverses the ranking (10-13)</summary>

<dl><dd>

Accuracy weights each class by how often it appears in the gold labels (Correct 55, Incorrect 76,
Unknown 19), so a judge that never says Unknown loses very little. Using macro-averaged recall and
macro F1 instead:

| judge | accuracy | macro recall | macro F1 | C / I / U |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (thinking) | 80.7 | **79.9** | **76.0** | 90.9 / 75.0 / 73.7 |
| Qwen3.8-27B (thinking) | 78.7 | 75.4 | 72.5 | 85.5 / 77.6 / 63.2 |
| Qwen3-235B-A22B-2507 | 74.0 | 72.2 | 68.6 | 83.6 / 69.7 / 63.2 |
| Qwen3.8-Flash-Next | 79.3 | 69.8 | 70.1 | 90.9 / 81.6 / 36.8 |
| **Llama-3.1-405B-INT4 (in use)** | **83.3** | 69.6 | 70.9 | 89.1 / 93.4 / 26.3 |
| Jev 1.13 | 78.7 | 68.3 | 69.8 | 80.0 / 88.2 / 36.8 |
| Qwen3.6-35B-A3B | 72.7 | 66.4 | 65.9 | 87.3 / 69.7 / 42.1 |
| Qwen3-32B | 74.0 | 62.6 | 63.6 | 80.0 / 81.6 / 26.3 |
| Qwen3.8-27B | 64.0 | 62.1 | 60.0 | 74.5 / 59.2 / 52.6 |
| Llama-3.3-70B | 73.3 | 60.5 | 60.8 | 76.4 / 84.2 / 21.1 |
| gpt-oss-120b | 62.7 | 60.1 | 57.7 | 61.8 / 65.8 / 52.6 |

This matters in actual play. Over the 180 episodes and 4,500 referee answers of the SP pilot, the
referee answered No 58.5% of the time, Yes 27.6% and Unknown 13.8%, so about one question in seven
falls in the class the 405B gets right least often. A referee that rarely says Unknown has to push
those questions into a Yes or a No, which adds noise to the game itself, not only to our metric.

**Decision.** We keep the referee, since AR-Bench uses the 405B and reports accuracy. The macro
table should still go in the paper next to accuracy, and once the main results are in, the SP
numbers should be re-run with Flash-Next (thinking) as a robustness check. *Not yet run.*

</dd></dl>
</details>

<details>
<summary>A2.6 Reading label probabilities instead of sampling a word (10-14, H100)</summary>

<dl><dd>

What set Jev apart from the chat judges was not its accuracy but the fact that its verdicts come
back as a distribution. Nothing about that is proprietary: any served model can be restricted to
the three labels and read at the first token.
[`runner/typed_judge_eval.py`](../runner/typed_judge_eval.py) does this with
`structured_outputs: {"choice": [...]}` and the official referee prompt, and
[`h100/typed_bench.sh`](../h100/typed_bench.sh) scores one model under three protocols.

| model | sampled, T=0.7 | sampled, T=0 | typed read | ECE | C / I / U (typed) | macro |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 34.7–37.3 | 34.7–35.3 | 36.0–36.7 | 0.56 | 50.9 / 17.1 / 68.4 | 45.5 |
| Qwen3-32B-FP8 (no thinking) | 72.7–74.7 | 73.3–74.0 | 73.3 | 0.16 | 80.0 / 80.3 / 26.3 | 62.2 |
| **Qwen3.8-27B-FP8 (no thinking)** | 64.7 | 70.0 | **70.7** | **0.055** | 76.4 / 68.4 / 63.2 | **69.3** |

The typed read never changes the verdict: the argmax is the same token, so it agrees with greedy
decoding to within one point. All it adds is the distribution, and that is only useful when the
distribution carries information. For Qwen3-32B it does (ECE 0.16), and the threshold becomes a
usable control: calling Unknown whenever its probability is above 0.30 raises Unknown recall from
26.3 to 36.8 and macro recall from 62.2 to 64.4, for 0.6 points of accuracy. For the 7B it does not:
its mean P(argmax) is 0.945 while its accuracy is 0.360. Whether the probabilities are calibrated
depends on the model, not on how they are read, and that is also the fair way to read the Jev
result.

Qwen3.8-27B is the best-calibrated judge we measured. Its ECE is 0.055, it is slightly
under-confident (mean P(argmax) 0.665 against accuracy 0.707), and its macro recall of 69.3 matches
the 405B's 69.6, from a 27B checkpoint on one H100, because it is the only one that does not
collapse on Unknown. Its accuracy is 12.6 points lower, so it cannot replace the referee on a
benchmark that reports accuracy, but it is the model to use where the third class matters. It is
also the only model for which sampling temperature makes a real difference: T=0.7 costs it 5.3
points compared with T=0.

</dd></dl>
</details>

<details>
<summary>🔗 Traceability (A2)</summary>

<dl><dd>

- Evaluation: [`runner/turtle_judge_eval.py`](../runner/turtle_judge_eval.py) (sampled) ·
  [`runner/typed_judge_eval.py`](../runner/typed_judge_eval.py) (typed read) ·
  [`runner/jev_judge_eval.py`](../runner/jev_judge_eval.py) and
  [`runner/jev_judge_variants.py`](../runner/jev_judge_variants.py) (Jev) ·
  [`runner/probe_guided.py`](../runner/probe_guided.py) (constrained-output probe)
- Serving: ☁️ `engaging/calib.sbatch`, `engaging/serve_referee.sbatch`, `engaging/judge405.sbatch` ·
  📁 [`h100/typed_bench.sh`](../h100/typed_bench.sh)
- Predictions and logs: 📁 [`experiments/02_judge_calibration/turtle150_predictions/`](../experiments/02_judge_calibration/turtle150_predictions/)
- Details: 📁 [`experiments/02_judge_calibration/README.md`](../experiments/02_judge_calibration/README.md)
- Credentials: the Jev key is in the macOS keychain under `typesafe_token`; **the value is never committed**

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A3 Situation Puzzles without a harness, official-grade referee (10-10 21:29–21:36, 7 min)</b></summary>

<dl><dd>

F1 0.621, with the 405B-INT4 referee answering the questions (see A2). All SP numbers in this log
use this referee.

<details>
<summary>🔗 Traceability (A3)</summary>

<dl><dd>

- Details: 📁 [`experiments/07_sp_bare_405judge/`](../experiments/07_sp_bare_405judge/)
- Referee calibration: [A2](#)
- Raw traces: Release `sp405_traces.tar.gz`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A4 What the Situation Puzzles metric measures (10-21)</b></summary>

<dl><dd>

This re-analyses records we already had: the frozen evaluation of five harnesses on all 100 test
puzzles, 500 episodes in total, with no new inference. Script
[`runner/metric_audit.py`](../runner/metric_audit.py); details
[`experiments/11_metric_audit/`](../experiments/11_metric_audit/).

| correlated with character-level F1 | r |
|---|---|
| **answer length in words** | **+0.806** |
| output tokens for the episode | +0.403 |
| **key-question recall (ours)** | **+0.050** |

Length alone gives R² = 0.650; adding recall brings it to 0.652, with a recall coefficient of
+0.021. With length partialled out, recall correlates with F1 at +0.068. Mean F1 rises steadily with
length and never turns down (25–50 words 0.458, 50–75 0.634, 75–100 0.766, 100–125 0.826, 125–150
0.868), so in practice the metric does not penalise length.

How much of the headline gain is left after controlling for length: +0.005 in the 25–50 word band,
+0.072 in 50–75, +0.043 in 75–100 and +0.004 in 100–125, compared with +0.152 overall. The final
harness writes 91 words where bare writes 61, and that difference accounts for most of the gap. The
advantage shrinking at longer lengths could mean the advantage is small, or that F1 saturates with
length. F1 alone cannot tell these apart, which is what semantic grading is for.

This has three consequences for the benchmark. First, the metric rewards verbosity, and our harness
drifted in exactly that direction, from 670 to 5,263 output tokens per episode. Second, an agent can
skip steps: nothing requires the final answer to follow from the questions and answers on record, so
it could waste all 25 questions and then write a long answer from the surface alone. The degenerate
baseline that would measure this needs the gold bottoms and has *not been measured yet*. Third, the
environment is itself noisy: the referee is right about 83.3% of the time, so roughly one in six of
the answers the questioner receives may be wrong.

**No change to what we report.** AR-Bench's metric is character-level F1, and the main tables keep
it, as set out in [`OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md). A4 belongs in
the limitations section; it does not replace the metric.

</dd></dl>

</details>

<details>
<summary><b>A5 Independent replication: a fixed-dialogue writer probe (10-15–21)</b></summary>

<dl><dd>

A preregistered experiment by Yangzi Sun that separates the questioning from the final answer. Our
recorded dialogues (bare and first generation) are held fixed, and each is given to two answer
writers (plain and evidence-constrained). That gives four conditions over 65 cases, 260 generations
and 520 blind ratings, scored with two new ordinal ratings, Semantic Correctness and Evidence
Faithfulness.
[Report](https://github.com/sun-tze/self-evolving-active-reasoning/blob/sun/c2-confirmatory-65/experiments/sun_c2_confirmatory/REPORT.md)

| condition | F1 | semantic (0–2) | faithfulness (0–2) | fully correct by both reviewers |
|---|---|---|---|---|
| A bare / plain | 0.681 | 0.785 | 0.662 | 7/65 |
| B bare / evidence | 0.683 | 0.754 | 0.669 | 8/65 |
| C first-selected / plain | 0.693 | 0.754 | 0.331 | **15/65** |
| D first-selected / evidence | 0.695 | 0.662 | 0.400 | 12/65 |

**What we accept.** F1 and semantic correctness rank answers in opposite order 82 times across 29
cases, so word overlap and actually understanding the cause can diverge completely. The
evidence-constrained instruction did not improve semantic correctness (−0.0308, 95% CI [−0.0577,
−0.0058]). A likely reason is that a situation puzzle's solution is, by design, not in the evidence:
tying the model to confirmed facts suppresses the abductive leap the task needs. This reaches the
same conclusion as [A4](#) by a different route.

**What we do not accept.** (i) It evaluates the first generation, whose key-question recall we had
already shown to be flat; by the final harness, recall on the unseen split rises from 57.6 to 66.4.
(ii) The candidate that was not chosen, `b2r0g1`, gets a higher F1 than the first generation with
half the tokens and 10.5 more points of recall, which length alone does not explain. (iii) Its own
data shows the evolved dialogue doubling the number of answers that both reviewers scored fully
correct (15/65 against 7/65), even though the ordinal mean moves the other way. Leaving out the
fully correct cases, the evolved dialogue averages 0.38 against 0.64 for bare, so the evolved
condition is bimodal while bare collects partial credit.

**Limitations, as the report itself notes.** It does not reproduce our final-answer procedure (three
drafts, scoring, three checks), so none of its conditions matches the system we actually run. The
two reviewers are two separate runs of the same model family, so their 96.9% agreement does not show
that they are accurate. We would add one more confound: faithfulness may depend mostly on how much
evidence there is to contradict. A weaker dialogue produces fewer constraints and so fewer chances to
violate them, which means the measure should be normalised per recorded constraint.

**Next step, not yet run.** Blind semantic grading of the complete harness answers for bare, the
final harness and `b2r0g1`, by a judge from a different model family (Opus 5 scores 88.0 on
TurtleBench, against 83.3 for the 405B), with 30 answers checked by a human, reporting both the mean
and the fully-correct rate.

</dd></dl>

</details>

<details>
<summary><b>A6 Code and data audit of the official GN and SP evaluators (10-21)</b></summary>

<dl><dd>

A reading of the released AR-Bench code (`arbench/reasoner/{gn,sp}/*_evaluator.py`,
`arbench/utils/utils_{gn,sp}.py`) and test data, with the scoring functions called directly. No
model was run.

**Guess Numbers.**
- The official methods do not share a success rule. The zero-shot / few-shot loop always plays all
  25 rounds and scores only the extra final guess. ToT scores the last guess on its search path.
  Proactive CoT counts a hit in any round and stops there. UoT counts a hit in any round, and when
  one candidate remains it fills that candidate in as the answer.
- ToT chooses among the model's candidate guesses with `compare_guess` against the gold number, so
  its search is guided by the answer. UoT keeps the 5,040-candidate set exactly in Python, so the
  elimination is done by code rather than by the model.
- Guesses are not validated. A guess with repeated digits such as 1111 is accepted and scored, which
  lets a player probe one digit at a time. The parser takes the last run of four digits anywhere in
  the text, so "2024" in the reasoning becomes the guess and "12345" becomes 1234.
- If the reply contains no 4-digit number, `compare_guess` raises a `ValueError` that the loop does
  not catch. The script resumes from the number of logged episodes, so the failed episode is
  silently re-sampled.

**Situation Puzzles.**
- `f1_score(prediction, bottom)` is called on raw strings, so the character-level F1 counts letters
  regardless of order. Using another puzzle's bottom as the prediction scores 0.833 on average over
  the 100 test puzzles (0.848 with its first 137 words, 0.136 with its first 10).
- The word-level variant applies no lowercasing, punctuation stripping or stop-word removal. A
  137-word string made of the corpus's most frequent function words scores word-level F1 0.349 and
  character-level F1 0.692. Another puzzle's bottom scores word-level F1 0.223.
- The referee's reply is not checked against Yes / No / Unknown; an off-format reply is logged and
  passed back to the questioner unchanged.

**What this means for testing.** For GN we use one success rule across all methods, validate every
guess (four distinct digits; an invalid guess uses up a round), and exclude any method that reads the
gold number. For SP we report F1 next to the two floors above (another puzzle's bottom, and the
function-word string) and next to key-question recall, at matched answer length.

<details>
<summary>🔗 Traceability (A6)</summary>

<dl><dd>

- Code read: `AR-Bench/arbench/reasoner/gn/gn_evaluator.py`, `AR-Bench/arbench/reasoner/sp/sp_evaluator.py`,
  `AR-Bench/arbench/utils/utils_gn.py`, `AR-Bench/arbench/utils/utils_sp.py`
- Data: `AR-Bench/data/gn/test.json`, `AR-Bench/data/sp/test.json`
- Audit script and numbers: *not yet committed*

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A7 Guess Numbers without a harness (pending)</b></summary>

<dl><dd>

*Not yet run.* Frozen Qwen2.5-7B-Instruct under the official zero-shot loop, scored both with the
official rule (final guess after round 25) and with a hit in any round, using the validated guesses
described in A6. Reference point: AR-Bench Table 6 reports 4 for Qwen-2.5-7B.

Existing code: [`runner/run_arbench_gn.py`](../runner/run_arbench_gn.py) ·
[`runner/gn_runner.py`](../runner/gn_runner.py) · [`runner/gn_runner_async.py`](../runner/gn_runner_async.py) ·
`stage0/gn_oracle.py` (candidate set and feasible-set bookkeeping) · `stage0/selftest_gn.py`

</dd></dl>

</details>

> ---

# Part B: context evolution

> In context evolution, the evolved object is the text the frozen questioner reads: the system
> prompt, task instructions and any notes accumulated across episodes. The episode loop, the answer
> parser and the environment are fixed code and do not change between candidates. In C the reverse
> holds: the evolved object is the harness code. Keeping the two separate lets us say whether a gain
> came from what the model is told or from what the code around it does.

<details>
<summary><b>B1 Context evolution on Guess Numbers (pending)</b></summary>

<dl><dd>

*Not yet run.* Conditions to fill in when it runs: task (Guess Numbers), questioner model, what the
proposer may change, signals the judge sees (strict / labeled), batch size and budget, split, and
the frozen evaluation set.

</dd></dl>

</details>

<details>
<summary><b>B2 Context evolution on Situation Puzzles (pending)</b></summary>

<dl><dd>

*Not yet run.* Conditions to fill in when it runs: task (Situation Puzzles), questioner model, what the
proposer may change, signals the judge sees (strict / labeled), batch size and budget, split, and
the frozen evaluation set.

</dd></dl>

</details>

> ---

# Part C: harness evolution

<details>
<summary><b>C1 Test-time harness evolution on Situation Puzzles (10-13 05:03–07:17, 2 h 14)</b></summary>

<dl><dd>

The test-time harness evolution loop first developed on Circuit Decoding ([CD5](#)), applied to a
task whose hypothesis space cannot be enumerated. The questioner is the frozen
Qwen2.5-7B; the referee and the key-question judge are the official Llama-3.1-405B-INT4; and the
protocol is AR-Bench's own: 25 yes/no questions, then a written answer. A 30-puzzle stream (seed 0),
B=10, G=3, R=2, with **strict** signals: run status and referee answers only, no F1.

**Stream** (prequential = the incumbent on the new batch, before it adapts):

| batch | incoming → committed | F1 (char) | key-q recall | transductive F1 | pool |
|---|---|---|---|---|---|
| 0 | bare → b0r1g2 | 63.7 | 55 | 79 | 7 |
| 1 | b0r1g2 → b1r1g0 | 74.5 | 65 | 67 | 5 |
| 2 | b1r1g0 → b2r1g0 | 79.9 | 72 | 69 | 6 |

**Frozen evaluation on all 100 test puzzles**, 70 of which the stream never used:

| harness | F1, 100 | F1, unseen 70 | key-q recall, 100 | tok/ep |
|---|---|---|---|---|
| bare (AR-Bench's official zero-shot loop) | 60.4 | 61.4 | 59.7 | 670 |
| b0 committed | 76.7 | 75.7 | 59.0 | 4486 |
| b1 not selected (`b1r0g0`) | 73.0 | 73.8 | 64.3 | 4685 |
| b1 committed | 76.2 | 75.9 | 63.2 | 4944 |
| **b2 not selected (`b2r0g1`)** | 76.3 | **77.5** | 68.2 | **2112** |
| b2 committed (final) | 75.6 | 75.4 | **68.7** | 5263 |

Label-free evolution, starting from ten failure traces, lifts the frozen 7B from 60.4 to 73–77 on
puzzles it has never seen. That is above every zero-shot model in AR-Bench's own table, including
GPT-4o at 63, and the 7B still writes every question. Key-question recall only rises in the later
generations (59 → 68), so the first generation improves the answer rather than the questions. The
cost is 3–8× the output tokens; `b2r0g1`, which was not chosen, does as well at half that. **Read
this together with [A4](#)**, which shows that much of the F1 column is answer length.

**What the harness learned** (the batch-0 winner, written by the adversarial-audit proposer). Each
turn, the model gets a single fresh message containing the puzzle, the referee's answers sorted into
CONFIRMED / RULED OUT / NOT IN THE STORY, and a focus. It starts by testing the assumptions a reader
would take for granted, since a situation puzzle usually turns on one of them being false. It then
keeps a single working story that has to include every confirmed fact, and tests the unconfirmed
parts one at a time. After a run of questions with no Yes, it asks one broad question about an
aspect it has not yet covered. Every candidate question goes through format checks (one fact,
yes/no, no "or", not about beliefs, no invented names, not a near-repeat), and the final answer is
rewritten from the confirmed facts. Later generations added vetoes on repeats and on leads the
referee had already denied, a ban on entities refused twice, and a ban on referring to unconfirmed
guesses as if they were established.

⚠️ The judge is noisy on this task. At batch 1 it committed the candidate with the lowest F1 in the
pool (0.674, against the incumbent's 0.745), and at batch 2 it passed over the best one (0.802). On
ten-puzzle batches those gold differences are within noise, which is a finding in its own right;
[C2](#) quantifies it. Session audits: 18 of 21 clean.

<details>
<summary>🔗 Traceability (C1)</summary>

<dl><dd>

- Task adapters: [`meta/cd_tthe/tasks.py`](../meta/cd_tthe/tasks.py) ·
  [`sp_bridge.py`](../meta/cd_tthe/sp_bridge.py) (bottoms and key questions private) ·
  [`sp_harness_base.py`](../meta/cd_tthe/sp_harness_base.py) ·
  [`sp_signals.py`](../meta/cd_tthe/sp_signals.py) ·
  [`agents_sp/bare.py`](../meta/cd_tthe/agents_sp/bare.py) (the official zero-shot loop)
- Details: 📁 [`10_sp_tthe_pilot`](../experiments/10_sp_tthe_pilot/), including `judge_verdicts/`,
  `proposal_cards/`, `candidates/`, `eval_test100_scores.json`, `trajectory.png` and
  [`dialogues/`](../experiments/10_sp_tthe_pilot/dialogues/) (four puzzles, bare against evolved)
- Raw traces: Release `sp_tthe_pilot_raw.tar.zst` (17 MB)
- Cost: 21 proposer and judge sessions, about **$12** at subscription list price, 2 h 14 min

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>C2 Checking the judge inside the loop (10-13)</b></summary>

<dl><dd>

TurtleBench measures the referee. The judge that decides which harness is deployed next had no check
of its own. It does not need an external benchmark: every run stores the gold scores of the
candidate pool, which the judge never saw, next to the candidate it committed.
Script: [`runner/judge_pick_eval.py`](../runner/judge_pick_eval.py).

| run | task | committed a best-scoring candidate | mean regret | worst | mean rank |
|---|---|---|---|---|---|
| `strict6_b10g3r2` | CD | 2/3 | 0.011 | 0.033 | 2.00 |
| `labeled6_b10g3r2` | CD | 3/3 | 0.000 | 0.000 | 1.00 |
| `sp_strict1_b10g3r2` | SP | **1/3** | **0.063** | **0.114** | 3.67 |

Regret is measured in each task's own metric. The Circuit Decoding pools are saturated (six
candidates at 1.000, so any choice is correct), so the check tells us little there. Situation
Puzzles is where it matters: the judge gave up 0.075 and 0.114 F1 on its two bad calls, and at
batch 1 it committed the worst of five.

**This should be in the paper.** The label-free judge is part of the mechanism we claim, and on the
harder task it is measurably noisy. Part of the reason is the ten-puzzle batches, where those gold
gaps are within noise, and that is worth reporting too.

**External benchmarks that would fit (none run yet).** The referee's decision is three-way grounded
inference, which is well benchmarked outside puzzle games. FEVER uses SUPPORTED / REFUTED / NOT
ENOUGH INFO, the same three labels with NEI in the role of Unknown. ANLI and MNLI use entailment /
contradiction / neutral without the game framing, which separates reasoning from format. The
unanswerable half of SQuAD 2.0 isolates the narrower question of whether something can be decided
at all. JudgeBench and RewardBench 2 evaluate a different kind of judge, one that chooses between
candidate responses; that is closer in spirit but a different decision, so at most they would serve
as a sanity check. The first three are public and need no login.

</dd></dl>

</details>

<details>
<summary><b>C3 Test-time harness evolution on Guess Numbers (pending)</b></summary>

<dl><dd>

*Not yet run.* Conditions to fill in when it runs: task (Guess Numbers), questioner model, what the
proposer may change, signals the judge sees (strict / labeled), batch size and budget, split, and
the frozen evaluation set.

</dd></dl>

</details>

> ---

# Appendix: Circuit Decoding (earlier testbed)

<details>
<summary><b>Circuit Decoding: official definitions and baselines</b></summary>

<dl><dd>

**Official baselines (MTP Figure 1)**

| model | circuit-wise accuracy |
|---|---|
| Gemini-2.5-Pro | 39.8 |
| GPT-4o tier | 15.4 |
| open-weight tier | 9.3 |

MTP Table 1 gives one metric for Circuit Decoding, *normalized circuit-wise accuracy*, but never says
how it is normalized. The figures plot a "Success Rate (%)" whose definition for this task is also
never given. We therefore report two numbers: `circuit_wise_accuracy`, which is the paper's metric,
and `joint_truth_table_correct`, which is our literal reading of the paper's term for a solved
episode.


</dd></dl>

</details>

<details>
<summary><b>CD1 Stage-0: hypothesis space, oracle and synthetic policies (10-06, CPU)</b></summary>

<dl><dd>

| policy | solved | queries | ex-ante eff | zero-information queries |
|---|---|---|---|---|
| greedy max-entropy (the oracle) | **300/300** | 11.60 | 1.000 | 0.0% |
| greedy over 4 sampled queries | 250/300 | 13.92 | 0.775 | 18.1% |
| random fresh entry | 156/300 | 16.51 | 0.629 | 27.7% |
| sequential scan | 58/300 | 17.41 | 0.540 | 38.1% |
| uniform random (repeats allowed) | 35/300 | 17.63 | 0.480 | 44.1% |

The 18-query budget is enough: on average the oracle narrows the hypothesis space to a single answer
in 11.60 queries. Any later failure therefore comes from the policy rather than the budget, and
everything that follows depends on this.

<details>
<summary>🔗 Traceability (CD1)</summary>

<dl><dd>

- Details: 📁 [`experiments/01_stage0_oracle/`](../experiments/01_stage0_oracle/)
- Oracle and synthetic policies: `stage0/cd_oracle.py` (CPU only, no model calls)

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>CD2 Actionability ladder: seven hand-written harnesses (10-10/11)</b></summary>

<dl><dd>

| harness (what is appended after each answer) | cwa | joint | uniq@end (ours) | eff (ours) | tok/ep |
|---|---|---|---|---|---|
| bare — nothing | 12.9 | 0.5 | 23.2 | 0.553 | 2389 |
| facts (a) — the model's own observations restated | 16.2 | 0.5 | 29.0 | 0.570 | 2800 |
| symbolic — remaining configurations and determined outputs | 20.5 | 2.2 | 41.2 | 0.612 | 2648 |
| enum15 — symbolic plus the full list when ≤15 remain | 23.7 | 5.8 | 41.8 | 0.612 | 2892 |
| **T3-gated** — enum15 only when a trap proxy fires | **31.3** | **10.2** | 36.8 | 0.612 | 3104 |
| splits (c) — enum15 plus 0/1 split counts per open entry | 22.3 | 5.7 | **53.7** | 0.702 | 4316 |
| oracle hint (d) — enum15 plus the max-entropy query | 27.3 | 6.8 | **96.5** | 0.951 | 4522 |

The main observation on this track is that better information gathering does not produce better
final answers. With the oracle hint, uniq@end reaches 96.5, meaning the hypothesis space collapses
to a single answer in almost every episode, yet joint accuracy is only 6.8. The questions are
already good enough; what fails is turning the collapsed state into an answer. T3-gated gets a
higher joint score with less information, which points to the same place.

<details>
<summary>🔗 Traceability (CD2)</summary>

<dl><dd>

- Details: 📁 [`experiments/03_cd_ladder/`](../experiments/03_cd_ladder/)
- Scoring and the ladder figure: [`runner/cd_score.py`](../runner/cd_score.py) ·
  [`runner/cd_ladder_plot.py`](../runner/cd_ladder_plot.py)
- Raw traces: Release `cd_ladder_traces.tar.gz`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>CD3 EvoHarness-RL as an external baseline (10-11 06:28–07:36, 1 h 08)</b></summary>

<dl><dd>

| mode | run | cwa | joint | uniq@end (ours) |
|---|---|---|---|---|
| env_only (ReAct, no harness) | s0 / s1 | 4.3 / 4.9 | 0.0 / 0.0 | 29.3 / 26.0 |
| inline = EvoHarness-Base (the model calls track / commit / recall / note) | s0 / s1 | 5.7 / 6.9 | 0.7 / 1.7 | 30.0 / 31.7 |
| **always_on (a STATE panel every turn)** | s0 / s1 | **74.8 / 79.0** | **62.0 / 65.0** | **71.0 / 74.7** |

With the same state information available, showing a panel every turn works far better than asking
the model to keep track itself (joint 62–65 against 0.7–1.7). The inline mode is barely better than
ReAct, so giving a frozen 7B tools to keep its own notes does not work on this task. Together with
CD2, this suggests the harness should maintain the state for the model but should not do the
reasoning for it.

<details>
<summary>🔗 Traceability (CD3)</summary>

<dl><dd>

- Details: 📁 [`experiments/06_evoharness_cd_baseline/`](../experiments/06_evoharness_cd_baseline/)
- Our Circuit Decoding adapter for the official framework: `baselines/evoharness_cd/`
- Scoring: [`runner/cd_score_evoharness.py`](../runner/cd_score_evoharness.py)
- Raw traces: Release `evoharness_cd_trajectories.tar.gz`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>CD4 One-shot evolution under regime state_only</b></summary>

<dl><dd>

<details>
<summary>CD4 `so1`, regime state_only (10-10 21:22 – 10-11 ~01:00, ~3 h 38)</summary>

<dl><dd>

The rule: the harness may keep and display state and may refuse queries, but it may not compute
information gain, report split counts or recommend queries. `veto_only` reaches circuit-wise accuracy
**91.3** and joint **86.2**, with uniq@end 96.7, at the same token cost as bare.

This is our claim in its simplest form. Remove the part that computes on the model's behalf, keep the
constraints and the state, and the score still rises without using more tokens. CD2's T3-gated
harness and CD3's always_on mode point the same way.

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>CD5 Test-time harness evolution and label isolation</b></summary>

<dl><dd>

> In test-time harness evolution, instances arrive in batches. After each batch a proposer writes
> candidate harnesses, and a judge decides, without seeing gold, whether to commit one. We run two
> streams: **strict**, where the judge sees only whether runs completed, and **labeled**, where it
> also sees outcomes. If strict ends up about as good as labeled, that supports the label-free
> claim. The proposer and the judge are Claude Opus 5.5 running in Claude Code; the questioner is a
> frozen Qwen2.5-7B.

<details>
<summary>CD5 Stream, audit and frozen evaluation (10-12 19:01–22:33, 3 h 32)</summary>

<dl><dd>

**Stream `strict6`** (B=10, G=3, R=2):

| batch | incoming → committed | cwa | joint | transductive joint | pool | wall |
|---|---|---|---|---|---|---|
| 0 | bare → b0r1g1 | 7 | 0 | 100 | 5 | 2577 s |
| 1 | b0r1g1 → b1r1g0 | 97 | 90 | 90 | 6 | 1080 s |
| 2 | b1r1g0 → b2r0g1 | 100 | 100 | 100 | 6 | 2188 s |

**Stream `labeled6`** reaches 100/100 in batch 0 and commits nothing after that.

**Frozen evaluation on 200 held-out instances**, with no proposer and no judge: bare gets
circuit-wise accuracy 11.8 and joint 0.0; the strict stream's final harness `b2r0g1` gets
**99.5 / 98.5** (98.2 on the 170 unseen instances); `b1r1g0` gets 99.3 / 98.0; and the labeled
stream's final harness gets 100/100 at 215 tokens per episode.

**Audit.** 29 of 32 sessions were clean, and every judge session was clean. The judge rejected three
harnesses that scored 10/10 because they showed the model a menu or a ranking: the model has to write
every query itself. This is not a matter of style. If the harness picks the queries, the result no
longer tells us anything about the mechanism.

**strict ≈ labeled** is the direct evidence for the label-free claim: both streams end at the
ceiling, and the strict one needs one extra batch to get there.

</dd></dl>
</details>

<details>
<summary>🔗 Traceability (CD5)</summary>

<dl><dd>

- Loop: [`meta/cd_tthe/optimize.py`](../meta/cd_tthe/optimize.py) ·
  [`proposer.py`](../meta/cd_tthe/proposer.py) (prompts, `audit_session`, subscription-limit retry) ·
  [`solve_worker.py`](../meta/cd_tthe/solve_worker.py) (reveal levels) ·
  [`audit.py`](../meta/cd_tthe/audit.py) (static audit)
- Evaluation: [`meta/cd_tthe/evaluate.py`](../meta/cd_tthe/evaluate.py) ·
  [`trajectory.py`](../meta/cd_tthe/trajectory.py)
- Details: 📁 [`09_cd_tthe_isolated`](../experiments/09_cd_tthe_isolated/)
- Transcripts in each folder's `dialogues/`; verdicts in `judge_verdicts/`; audits in
  `session_audits.jsonl`
- Raw traces: Release `cd_tthe_isolated_raw.tar.zst`

</dd></dl>
</details>

</dd></dl>

</details>

> ---

# Operations: the two machines and their pitfalls

<details>
<summary><b>Capability boundaries and known pitfalls</b></summary>

<dl><dd>

| | MIT ORCD / Engaging | standalone H100 (`ssh h100`) |
|---|---|---|
| GPUs | H200 141 GB, at most 2 on `mit_normal_gpu` | 4×H100 80 GB, **shared with other users**, chosen at launch by free memory |
| Queue and preemption | queued; the 405B runs on a preemptable partition and was preempted twice on 10-13 | no queue, no preemption; a model that fits serves in 1–3 minutes |
| Software | current vLLM | `venv_cu128` (vLLM 0.19.1, torch 2.11+cu128); driver 570.195.03 is CUDA 12.8, so the **ceiling is vLLM 0.26.0** |
| Does not fit | — | Llama-3.1-405B-INT4 (needs two cards), Qwen3.8-Flash-Next (~126 GB resident), Qwen3-235B |

- **Engaging requires Duo.** You need both the public key and Duo, and no token can be stored, so
  when the ControlMaster connection drops it has to be re-established by hand.
- **Only `venv_cu128` works on the H100.** The other virtual environment has a CUDA 13 build of
  torch and fails with "NVIDIA driver is too old".
- **`ninja` is needed at inference time.** Models with Gated DeltaNet (Qwen3.8) compile their
  attention kernel on the first forward pass. Without ninja the server starts and passes `/health`,
  then returns 500 on every request as the engine dies.
- **A model's name does not tell you its architecture.** Qwen3.8-27B's Hugging Face `architectures`
  field says `Qwen3_5ForConditionalGeneration`, which vLLM 0.19.1 already supports; searching the
  registry for "3.8" gives the wrong answer. Check `config.json`.
- **Never give `pgrep -f` or `pkill -f` a pattern that appears in your own command line.** It
  matches the shell that runs it.
- **vLLM 0.19 silently ignores `guided_choice`**: the request succeeds and nothing is constrained.
- **Claude Code's five-hour subscription window ends sessions without warning**, so the proposer
  detects the limit, and `--resume-from BATCH ROUND` picks the stream back up.
- 2026-10-21: the H100's SSH host key changed. We have **not accepted it** and are waiting for
  confirmation through another channel that the machine was rebuilt.

Machine notes: 📁 [`h100/README.md`](../h100/README.md).

</dd></dl>

</details>
