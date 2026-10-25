#!/bin/bash
# Serve the reasoner under test (AR-Bench's POLICY_* role) on one H100, and publish its endpoint.
#   sbatch --export=ALL,REPO=Qwen/Qwen2.5-7B-Instruct,TAG=qwen25_7b serve_7b.sbatch
#SBATCH --job-name=serve_policy
#SBATCH --partition=h100_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=mid
#SBATCH --output=%x_%j.out
set -uo pipefail
REPO=${REPO:-Qwen/Qwen2.5-7B-Instruct}; TAG=${TAG:-policy}
PORT=$(( 9000 + RANDOM % 900 ))
export PATH=$HOME/venv/bin:$PATH
# system nvcc is CUDA 12.0 but torch is cu130: every runtime JIT path must stay off
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$HOME/results" "$HOME/logs"
echo "host=$(hostname) repo=$REPO port=$PORT start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
$HOME/venv/bin/vllm serve "$REPO" --port "$PORT" --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
  > "$HOME/logs/vllm_${TAG}_$SLURM_JOB_ID.log" 2>&1 &
VP=$!
for i in $(seq 1 120); do
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 10
done
if ! curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "serve failed"; tail -25 "$HOME/logs/vllm_${TAG}_$SLURM_JOB_ID.log"; kill $VP; exit 1
fi
echo "$(hostname):$PORT" > "$HOME/results/serve_${TAG}.endpoint"
echo "up after ${SECONDS}s -> $(cat "$HOME/results/serve_${TAG}.endpoint")"
wait $VP
