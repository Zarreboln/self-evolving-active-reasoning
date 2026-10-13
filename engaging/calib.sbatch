#!/bin/bash
# Serve one candidate referee model from the scratch HF cache and run the TurtleBench-150
# calibration against it (runner/turtle_judge_eval.py), then exit.
#   sbatch --gres=gpu:h200:1 --export=ALL,REPO=openai/gpt-oss-120b,TAG=gptoss120b,VARGS="--max-model-len 8192",EARGS="--max-tokens 2048" engaging/calib.sbatch
#SBATCH --job-name=calib
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
REPO=${REPO:?hf repo id}; TAG=${TAG:?short tag}; VARGS=${VARGS:-}; EARGS=${EARGS:-}
CODE=$HOME/code/ar_harness
export HF_HOME=/orcd/scratch/orcd/012/$USER/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
PY=$CODE/conda_env/bin/python
VLLM=$CODE/conda_env/bin/vllm
export CUDA_HOME=$CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
RESULTS=$HOME/results/ar_harness/judge; mkdir -p "$RESULTS"
PORT=$((9000 + RANDOM % 900))
echo "host=$(hostname) start=$(date -Is) port=$PORT repo=$REPO tag=$TAG vargs=[$VARGS] eargs=[$EARGS]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$VLLM" serve "$REPO" --served-model-name "$REPO" $VARGS \
  --max-num-seqs 64 --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_calib_${TAG}_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died"; tail -60 "$RESULTS/vllm_calib_${TAG}_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
T0=$(date +%s)
"$PY" "$CODE/runner/turtle_judge_eval.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$REPO" \
  --turtlebench "$HOME/TurtleBench" --concurrency 32 $EARGS --out "$RESULTS/turtle150_${TAG}.json"
echo "calibration done in $(( $(date +%s) - T0 ))s end=$(date -Is)"
kill $PID 2>/dev/null; wait $PID 2>/dev/null; exit 0
