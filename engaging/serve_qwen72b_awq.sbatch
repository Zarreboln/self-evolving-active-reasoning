#!/bin/bash
# Serve Qwen2.5-72B-Instruct-AWQ (int4, ~39GB on disk) as a bigger/stronger
# policy or judge model than the 7B we've been using. Single GPU, any type --
# the whole partition is backlogged right now (h200/h100/l40s all gave the
# same ~2-day sbatch --test-only estimate), so restricting to one type buys
# nothing. Max-model-len and gpu-memory-utilization are kept conservative
# (small KV cache budget) so this also fits if Slurm lands it on an l40s
# (44GB total -- weights alone leave only ~5GB of headroom at 0.92 util);
# h100/h200 just end up with spare room to spare.
#
# Motivation: AR-Bench's own judge-reliability ablation (paper Table 12)
# never validates anything below Llama-3.1-70B as a judge (70B=89.5%,
# 405B=96% accuracy); our SP belief_injected keypoint-hit judge, run on
# Qwen2.5-7B, degenerated (ran away into few-shot continuation, or emitted
# plain garbage) -- a model this size is exactly the gap the paper's own
# ablation warns about. 72B-AWQ is the practical next rung up without
# needing multi-GPU tensor-parallel.
#SBATCH --job-name=serve_qwen72b
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
CODE=$HOME/code/ar_harness
POOL=$HOME/orcd/pool/ar_harness
export HF_HOME=$POOL/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
PY=$CODE/conda_env/bin/python
VLLM=$CODE/conda_env/bin/vllm
export CUDA_HOME=$CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

RESULTS=$HOME/results/ar_harness
mkdir -p "$RESULTS"
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PORT=$((8000 + RANDOM % 1000))
"$VLLM" serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_qwen72b_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "waiting for vLLM on :$PORT"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s on port $PORT"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died -- see vllm_qwen72b_${SLURM_JOB_ID}.log"; tail -40 "$RESULTS/vllm_qwen72b_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done

echo "serving on port $PORT -- sbatch leaves the server up for $(squeue -h -j $SLURM_JOB_ID -o %L 2>/dev/null || echo 'the remaining time limit'); cancel the job to tear it down"
wait $VLLM_PID
