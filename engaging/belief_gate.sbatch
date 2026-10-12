#!/bin/bash
# Decisive experiment, 2026-10-06: does an EXTERNALLY computed belief state
# fix the GN dead-guess rate, or not?
#
# Measured so far, all three AR-Bench prompt conditions, dead-guess rate
# essentially flat regardless of prompt wording:
#   bare 95.4%  taught_incremental 95.7%  cot(ours,broken) 95.5%
#   proactive_cot (AR-Bench's own, 42x more reasoning tokens) 87.0% -- CoT
#   alone barely moves it. So the bottleneck is not "nowhere to think",
#   it is retrieval/use of the feasible set from its own transcript.
#
# "belief_injected" reuses the "bare" system prompt verbatim and changes
# exactly one thing: after each turn's numeric feedback, the harness appends
# a belief summary computed externally from the full history (confirmed
# digits in/out, confirmed positions, and -- once small enough -- the exact
# remaining candidate list) -- see runner/gn_runner_async.py belief_summary().
# No RL, no weight changes, just a different harness-constructed prompt.
#
# Reuses gate_gn.jsonl: the (condition, secret, seed) resume logic skips the
# 900 episodes already there and only runs the new 300 for belief_injected.
#
#   sbatch ~/code/ar_harness/engaging/belief_gate.sbatch
#
#SBATCH --job-name=gn_belief
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=node2906   # the only h100 node in this partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
CODE=$HOME/code/ar_harness
POOL=$HOME/orcd/pool/ar_harness
export HF_HOME=$POOL/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
# FlashInfer 0.7.0 bundles its own CCCL headers, which nvcc 13.4 rejects:
#   "CUDA compiler and CUDA toolkit headers are incompatible"
# raised while JIT-building cached_ops/sampling. Nothing here needs
# FlashInfer -- a 7B at 4k context on an L40S -- so take the prebuilt paths
# instead of compiling anything at startup.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# The system /usr/bin/python3.12 ships no Python.h (/usr/include/python3.12
# holds only pyconfig-64.h), so Triton cannot JIT cuda_utils.c and the vLLM
# engine core dies at startup. conda pythons carry their own headers.
PY=$CODE/conda_env/bin/python
VLLM=$CODE/conda_env/bin/vllm
# FlashInfer JIT-compiles kernels and looks for nvcc under CUDA_HOME, which
# defaults to /usr/local/cuda -- absent on these nodes, so the engine core
# dies with "Could not find nvcc". pip pulled a complete CUDA 13.4 tree
# (nvcc + headers + libs) inside the env; point CUDA_HOME at it rather than
# loading the cuda/13.0.1 module, so the job carries its own toolchain.
export CUDA_HOME=$CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
# conda_env/bin must be on PATH too: FlashInfer shells out to `ninja`, which
# pip put there. Calling $VLLM by absolute path does not put it on PATH, and
# the node has no system ninja -- hence "FileNotFoundError: 'ninja'".
export PATH=$CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# Preflight. Six startup failures so far cost two minutes each because the
# missing piece only surfaced deep inside engine-core init. Fail in one
# second instead, naming what is missing.
fail=0
for t in nvcc ninja g++ python vllm; do
  p=$(command -v $t 2>/dev/null) && echo "  ok   $t -> $p" || { echo "  MISSING $t"; fail=1; }
done
[ -f "$CUDA_HOME/include/cuda_runtime.h" ] && echo "  ok   cuda headers" \
  || { echo "  MISSING cuda_runtime.h under CUDA_HOME"; fail=1; }
"$PY" -c "import vllm, torch; print('  ok   vllm', vllm.__version__, 'torch', torch.__version__)" || fail=1
"$PY" -c "import torch; assert torch.cuda.is_available(); print('  ok   cuda visible:', torch.cuda.get_device_name(0))" || fail=1
[ $fail -eq 0 ] || { echo "PREFLIGHT FAILED"; exit 2; }
echo "  attn=$VLLM_ATTENTION_BACKEND flashinfer_sampler=$VLLM_USE_FLASHINFER_SAMPLER"
RESULTS=$HOME/results/ar_harness
mkdir -p "$RESULTS"

echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PORT=$((8000 + RANDOM % 1000))
# vLLM 0.31: `python -m vllm.entrypoints.openai.api_server` is deprecated
# (use `vllm serve`), and --disable-log-requests was replaced by
# --enable-log-requests, which defaults off. Flag names verified against
# the installed source -- `--help` cannot run on the login node (no GPU:
# "Failed to infer device type").
"$VLLM" serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "waiting for vLLM on :$PORT"
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died -- see vllm_${SLURM_JOB_ID}.log"; tail -30 "$RESULTS/vllm_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done

T0=$(date +%s)
# Concurrency is required, not an optimisation: serially this is ~15 h,
# past the wall limit. 32 episodes in flight lets vLLM batch them.
"$PY" "$CODE/runner/gn_runner_async.py" \
  --concurrency 32 \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --arbench "$CODE/AR-Bench" \
  --out "$RESULTS/gate_gn.jsonl" \
  --conditions bare,taught_incremental,cot,belief_injected \
  --n-instances 100 --seeds 3 --max-turns 25 --temperature 0.7
echo "runner exit=$? elapsed=$(( $(date +%s) - T0 ))s  end=$(date -Is)"

"$PY" "$CODE/runner/summarize.py" "$RESULTS/gate_gn.jsonl"
