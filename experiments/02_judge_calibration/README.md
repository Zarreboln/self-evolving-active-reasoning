# 02 — Judge calibration on TurtleBench (150 fixed cases, seed 7, 0-shot)

**Code**: `runner/turtle_judge_eval.py` (sample = 150 of the 1531 English cases across 32
stories; one independent call per case; label parsed with a leading word boundary).
**Why**: AR-Bench's Situation Puzzles need an answerer/judge; the official one is
Llama-3.1-405B. Our earlier SP numbers used the 7B as its own judge.

| judge | accuracy | per-class recall (Correct / Incorrect / Unknown) | note |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 50.5% | almost never says "No" (23.8%) | invalidates the earlier SP numbers |
| Llama-3.1-70B (2-shot, archived predictions) | 74.0% | | |
| GPT-4o (2-shot, archived) | 63.3% | | |
| Llama-3.1-405B bf16 (0-shot, archived predictions) | 82.7% | | the official AR-Bench judge |
| **Llama-3.1-405B-Instruct AWQ-INT4** (ours, 2×H200, PP=2) | **82.0%**, 83.3% on a 2026-10-13 re-run | 83.6 / 93.4 / 31.6, re-run 89.1 / 93.4 / 26.3 | chosen as the paper-grade judge; `judge/turtle150_405b_int4.json`. The two runs differ only by sampling: `turtle_judge_eval.py` decodes at temperature 0.7, so one run of 150 cases carries ~±1.5 points of noise. |
| Claude Opus 5 (one fresh context per case) | 88.0% | Unknown recall 47.4% | `labels_opus.json` |

Two pitfalls documented in `experiments.md` (2026-10-10): tensor-parallel serving of the INT4
checkpoint on H200 trips a FlashInfer JIT build (use pipeline parallelism), and the checkpoint's
`generation_config` lacks `<|eot_id|>` (pass `stop_token_ids` [128001, 128008, 128009]).

## Smaller-referee search (2026-10-13)

Can a smaller model replace the 405B as AR-Bench's referee? Every candidate ran the same
TurtleBench-150 set, served on Engaging H200s, scored by `runner/turtle_judge_eval.py`.

Accuracy, then per-class recall as Correct / Incorrect / Unknown (gold counts 55 / 76 / 19).
Thinking off decodes 16 tokens; thinking on uses `--reasoning-parser qwen3` and 2048 tokens.

| candidate | job | thinking off | recall C/I/U | thinking on | recall C/I/U |
|---|---|---|---|---|---|
| Llama-3.1-405B-Instruct AWQ-INT4 (incumbent) | 25698041 | **83.3** | 89.1 / 93.4 / 26.3 | n/a | |
| Qwen3.8-Flash-Next-FP8 | 25718457 / 25724076 | 79.3 | 90.9 / 81.6 / 36.8 | **80.7** | 90.9 / 75.0 / 73.7 |
| Qwen3.8-27B-FP8 | 25718456 / 25724077 | 64.0 | 74.5 / 59.2 / 52.6 | **78.7** | 85.5 / 77.6 / 63.2 |
| Qwen3-235B-A22B-Instruct-2507-FP8 | 25718459 | 74.0 | 83.6 / 69.7 / 63.2 | not run | |
| Qwen3-32B | 25698272 | 74.0 | 80.0 / 81.6 / 26.3 | not run | |
| Llama-3.3-70B-Instruct-FP8-dynamic | 25698273 | 73.3 | 76.4 / 84.2 / 21.1 | not run | |
| Qwen3.6-35B-A3B-FP8 | 25718458 | 72.7 | 87.3 / 69.7 / 42.1 | not run | |
| gpt-oss-120b | 25698271 | 62.7 | 61.8 / 65.8 / 52.6 | not run | |
| Jev 1.13 (TypeSafe, hosted) | n/a | 78.7 | 80.0 / 88.2 / 36.8 | n/a | |

Where the accuracy goes: every candidate is decent on Correct and Incorrect and bad on Unknown, the
class that decides whether a player's question is actually answerable from the story. The 405B's lead
is on Incorrect (93.4, nobody else above 85); its own Unknown recall is 26.3, worse than six of the
seven alternatives. Thinking buys exactly that class back for the Qwen3.8 models (Flash-Next 36.8 to
73.7) without overtaking the 405B overall.

Thinking lifts the two Qwen3.8 models (+1.4 and +14.7 points) at ~2048 extra tokens per case, but none
closes the gap to the 405B, and the 405B is also what AR-Bench itself used. The 405B stays the referee
for Situation Puzzles; the smaller models remain options only if judge throughput becomes the bottleneck.
Serving notes: FP8 on H200 needs `VLLM_USE_DEEP_GEMM=0` (DeepGEMM JIT fails against nvcc 13.4 headers),
and Qwen3.8-Flash-Next needs `max_num_seqs<=57` for its Mamba cache.

### Case-by-case against the incumbent

Every candidate scored on the same 150 cases, compared to the 405B's own predictions
(`turtle150_predictions/`, one JSON per candidate with `pred` / `raw` / `gold` per case id).
The 405B gets 125 right and 25 wrong.

| candidate | accuracy | agrees with 405B | fixes 405B's 25 errors | breaks 405B's 125 rights |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (think) | 80.7% | 79.3% | 13 | 17 |
| Qwen3.8-Flash-Next | 79.3% | 84.7% | 8 | 14 |
| Qwen3.8-27B (think) | 78.7% | 76.0% | 12 | 19 |
| Qwen3-32B | 74.0% | 79.3% | 7 | 21 |
| Qwen3-235B-A22B-2507 | 74.0% | 72.0% | 13 | 27 |
| Llama-3.3-70B | 73.3% | 84.0% | 3 | 18 |
| Qwen3.6-35B-A3B | 72.7% | 77.3% | 8 | 24 |
| Qwen3.8-27B | 64.0% | 65.3% | 9 | 38 |
| gpt-oss-120b | 62.7% | 64.7% | 8 | 39 |
| Jev 1.13 | 78.7% | 80.7% | 10 | 17 |

No candidate is a drop-in: the best one disagrees with the 405B on 31 of 150 cases, and every one of
them loses more of the 405B's correct answers than it repairs of its errors. Swapping the referee
would move the Situation Puzzles numbers for reasons unrelated to the harness, so the 405B stays.

Raw material: `turtle150_predictions/` holds the per-case predictions and each job's stdout
(model repo, vLLM args, serve time, accuracy, prediction distribution, per-class recall).


## A typed-decision model as referee: Jev 1.13 (2026-10-13)

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe's "System One"
model: it takes a state plus typed questions and returns a distribution over the declared options,
reading them off the model rather than sampling text. That is this task's native shape, so it was
worth one run. `runner/jev_judge_eval.py`, same 150 cases, same seed, the three judging rules
verbatim as the question's `instructions` and the three labels as its `criteria`. It cannot take the
official prompt byte-identically, because it has no system message -- that is the one caveat on the
comparison.

| | Jev 1.13 | Llama-3.1-405B-INT4 |
|---|---|---|
| accuracy | 78.7% | 83.3% |
| recall Correct / Incorrect / Unknown | 80.0 / 88.2 / **36.8** | 89.1 / 93.4 / 26.3 |
| wall for 150 cases | **12 s** (16 at a time) | ~15 s on 2×H200, after a 270 s load |
| cost | **$0.0041** (98,480 input tokens at $0.042/MTok) | an H200 pair for the length of the run |
| hosting | closed, hosted API | our own GPUs |

It loses 4.6 points to the 405B overall and is not a drop-in: it agrees on 80.7% of cases, repairs
10 of the 405B's 25 errors and breaks 17 of its 125 correct ones. But it beats the 405B on Unknown,
the class every chat judge is worst at, and it is the only candidate that returns a usable
distribution. Its confidence (`1 - H(p)/ln K`) is monotone in accuracy:

| confidence | n | accuracy |
|---|---|---|
| 0.0–0.2 | 12 | 50.0% |
| 0.2–0.4 | 26 | 61.5% |
| 0.4–0.6 | 37 | 73.0% |
| 0.6–0.8 | 30 | 86.7% |
| 0.8–1.0 | 45 | 95.6% |

So the verdicts can be filtered rather than trusted flat, which no other candidate here allows.
Taking Unknown whenever its probability clears a threshold trades the two off:

| threshold | accuracy | Unknown recall | times Unknown said |
|---|---|---|---|
| argmax (0.50 in practice) | 78.7% | 31.6% | 10 |
| 0.40 | 76.7% | 36.8% | 17 |
| 0.30 | 72.0% | 63.2% | 35 |
| 0.25 | 66.7% | 68.4% | 48 |

Not adopted as the SP referee: AR-Bench's referee is the 405B, swapping it would move the Situation
Puzzles numbers for reasons unrelated to the harness, and it is a closed hosted API, so every
story and guess would leave our infrastructure. Worth keeping for where its shape actually fits --
a cheap filter, or a second opinion whose confidence tells us which verdicts to re-check.
Predictions with probabilities: `turtle150_predictions/turtle150_jev.json`.
