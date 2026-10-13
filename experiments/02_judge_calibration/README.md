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
| **Llama-3.1-405B-Instruct AWQ-INT4** (ours, 2×H200, PP=2) | **82.0%** | 83.6 / 93.4 / 31.6 | chosen as the paper-grade judge; `judge/turtle150_405b_int4.json` |
| Claude Opus 5 (one fresh context per case) | 88.0% | Unknown recall 47.4% | `labels_opus.json` |

Two pitfalls documented in `experiments.md` (2026-10-10): tensor-parallel serving of the INT4
checkpoint on H200 trips a FlashInfer JIT build (use pipeline parallelism), and the checkpoint's
`generation_config` lacks `<|eot_id|>` (pass `stop_token_ids` [128001, 128008, 128009]).

## Smaller-referee search (2026-10-13)

Can a smaller model replace the 405B as AR-Bench's referee? Every candidate ran the same
TurtleBench-150 set, served on Engaging H200s, scored by `runner/turtle_judge_eval.py`.

| candidate | thinking off | thinking on |
|---|---|---|
| Llama-3.1-405B-Instruct AWQ-INT4 (incumbent) | **83.3** | n/a |
| Qwen3.8-Flash-Next-FP8 | 79.3 | 80.7 |
| Qwen3.8-27B-FP8 | 64.0 | 78.7 |
| Qwen3-235B-A22B-2507-FP8 | 74.0 | not run |
| Qwen3-32B | 74.0 | not run |
| Llama-3.3-70B-Instruct-FP8 | 73.3 | not run |
| Qwen3.6-35B-A3B-FP8 | 72.7 | not run |
| gpt-oss-120b | 62.7 (over-predicts "Unknown") | not run |

Thinking lifts the two Qwen3.8 models (+1.4 and +14.7 points) at ~2048 extra tokens per case, but none
closes the gap to the 405B, and the 405B is also what AR-Bench itself used. The 405B stays the referee
for Situation Puzzles; the smaller models remain options only if judge throughput becomes the bottleneck.
Serving notes: FP8 on H200 needs `VLLM_USE_DEEP_GEMM=0` (DeepGEMM JIT fails against nvcc 13.4 headers),
and Qwen3.8-Flash-Next needs `max_num_seqs<=57` for its Mamba cache.
