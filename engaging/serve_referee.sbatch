#!/bin/bash
# Serve a referee / answerer model from the scratch HF cache and publish its endpoint:
#   sbatch -p mit_preemptable --gres=gpu:h200:1 --export=ALL,REPO=openai/gpt-oss-120b,TAG=gptoss120b,VARGS="--max-model-len 8192" engaging/serve_referee.sbatch
#   -> $HOME/results/ar_harness/serve_referee_<TAG>.endpoint = "<node>:<port>"
#SBATCH --job-name=referee
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
REPO=${REPO:?hf repo id}; TAG=${TAG:?short tag}; VARGS=${VARGS:-}
CODE=$HOME/code/ar_harness
export HF_HOME=/orcd/scratch/orcd/012/$USER/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
VLLM=$CODE/conda_env/bin/vllm
export CUDA_HOME=$CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
RESULTS=$HOME/results/ar_harness; mkdir -p "$RESULTS/judge"
EP=$RESULTS/serve_referee_${TAG}.endpoint; rm -f "$EP"; trap 'rm -f "$EP"' EXIT
PORT=$((9000 + RANDOM % 900))
echo "host=$(hostname) start=$(date -Is) port=$PORT repo=$REPO vargs=[$VARGS]"
"$VLLM" serve "$REPO" --served-model-name "$REPO" $VARGS \
  --max-num-seqs 128 --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/judge/vllm_referee_${TAG}_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died"; tail -60 "$RESULTS/judge/vllm_referee_${TAG}_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
echo "$(hostname):$PORT:$SLURM_JOB_ID" > "$EP"; echo "endpoint published: $(cat $EP)"
wait $PID
