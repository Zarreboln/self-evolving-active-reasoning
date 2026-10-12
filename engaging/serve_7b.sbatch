#!/bin/bash
# Serve Qwen2.5-7B-Instruct on one L40S for the harness-evolution loop and
# publish the endpoint so the mac-side loop can tunnel to it:
#   $HOME/results/ar_harness/serve_7b.endpoint  ->  "<node>:<port>"
#
#   sbatch ~/code/ar_harness/engaging/serve_7b.sbatch
#
# mit_normal_gpu caps jobs at 6 h; resubmit when it ends (the loop waits on
# the endpoint file and retries, see meta/cd_evolve/tunnel.sh).
#
#SBATCH --job-name=serve7b
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
CODE=$HOME/code/ar_harness
POOL=$HOME/orcd/pool/ar_harness
export HF_HOME=$POOL/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
VLLM=$CODE/conda_env/bin/vllm
export CUDA_HOME=$CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
RESULTS=$HOME/results/ar_harness; mkdir -p "$RESULTS"
EP=$RESULTS/${EP_NAME:-serve_7b.endpoint}   # override with --export=ALL,EP_NAME=...
rm -f "$EP"
PORT=$((8000 + RANDOM % 1000))
echo "host=$(hostname) start=$(date -Is) port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
trap 'rm -f "$EP"' EXIT
"$VLLM" serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 48 \
  --gpu-memory-utilization 0.88 --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_serve7b_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died"; tail -30 "$RESULTS/vllm_serve7b_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
echo "$(hostname):$PORT:$SLURM_JOB_ID" > "$EP"
echo "endpoint published: $(cat $EP)"
wait $PID
