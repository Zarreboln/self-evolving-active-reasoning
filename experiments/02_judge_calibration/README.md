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
