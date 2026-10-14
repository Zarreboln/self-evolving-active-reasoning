# The standalone H100 box

`ssh h100` (passwordless). Four H100 80GB cards, but **only GPU 3 is ours**: every command must set
`CUDA_VISIBLE_DEVICES=3`. Code and data live under `/home/tmp_l`. No queue and no preemption, which
is what it is good for: a model that fits comes up in one to three minutes and a 150-case judge
evaluation takes seconds, so it suits the fast iteration that ORCD's preemptable partition does not.

## What is already set up (2026-10-14)

| | |
|---|---|
| Environment | `~/code/ar_harness/venv_cu128` — vLLM 0.19.1, torch 2.11+cu128, `ninja` |
| Data | `~/TurtleBench`, `~/code/ar_harness/AR-Bench/data/{dc,gn,sp}` |
| Weights cached | Qwen2.5-7B-Instruct, Qwen3-32B-FP8, Qwen3.8-27B-FP8 (`HF_HOME=/home/tmp_l/ckpt/hf_home`) |
| Scripts | `h100/typed_bench.sh` (serve a model, score it three ways on TurtleBench-150), `h100/judge_bench.sh`, `h100/chain.sh` |

```bash
# one model, three protocols: sampled T=0.7, sampled T=0, typed read
cd ~/code/ar_harness
REPO=Qwen/Qwen3.8-27B-FP8 TAG=qwen38_27b_fp8 PORT=8141 \
  EARGS="--no-think" TARGS="--disable-thinking" ./h100/typed_bench.sh
```

## Traps, all of them hit on the first day

- **Two venvs, one works.** `venv` holds vLLM 0.31 with torch cu130, and the driver here is
  570.195.03 = CUDA 12.8, so it fails with "NVIDIA driver is too old". Use `venv_cu128`.
- **Version ceiling.** cu128 torch tops out at 2.11, and vLLM 0.27+ pins torch 2.13 (CUDA 13).
  The newest installable vLLM here is **0.26.0**. Upgrading means a new venv, not touching this one.
- **`ninja` is required at inference time, not install time.** Models with Gated DeltaNet
  (Qwen3.8) JIT-compile their attention kernel on the first forward. Without `ninja` the server
  starts, passes `/health`, and then returns 500 on every request before the engine dies.
- **Model names lie about architectures.** Qwen3.8-27B's HF architecture is
  `Qwen3_5ForConditionalGeneration`; grepping the vLLM registry for "3.8" wrongly says unsupported.
  Check `config.json` → `architectures`, not the repo name.
- **Never `pgrep -f` / `pkill -f` a pattern that your own command line contains.** It matches the
  shell running it: once this deadlocked a waiter against itself, once it killed the ssh session.
- **Memory is the real gap against ORCD.** One 80GB card cannot hold Llama-3.1-405B-INT4 (needs two),
  Qwen3.8-Flash-Next (~126GB resident) or Qwen3-235B. Those stay on ORCD's H200s.

## Judge numbers measured here

See `experiments/02_judge_calibration/README.md`. Short version: the typed read never moves the
argmax, and Qwen3.8-27B-FP8 is the best-calibrated judge measured (ECE 0.055, macro recall 69.3).
