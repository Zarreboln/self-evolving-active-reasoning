#!/bin/bash
# Serve the AR-Bench judge/respondent model -- Llama-3.1-405B-Instruct AWQ-INT4
# (hugging-quants), the paper's official 405B judge in 4-bit -- on two H200s,
# calibrate it on the fixed TurtleBench 150 (runner/turtle_judge_eval.py), then
# keep serving and publish the endpoint for SP/DC jobs:
#   $HOME/results/ar_harness/serve_judge.endpoint -> "<node>:<port>"
#
#   sbatch ~/code/ar_harness/engaging/judge405.sbatch
#
#SBATCH --job-name=judge405
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
CODE=$HOME/code/ar_harness
MODEL_DIR=/orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct-INT4
NAME=meta-llama/Llama-3.1-405B-Instruct-INT4
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
PY=$CODE/conda_env/bin/python
VLLM=$CODE/conda_env/bin/vllm
export CUDA_HOME=$CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
RESULTS=$HOME/results/ar_harness; mkdir -p "$RESULTS/judge"
EP=$RESULTS/serve_judge.endpoint; rm -f "$EP"; trap 'rm -f "$EP"' EXIT
PORT=$((9000 + RANDOM % 900))
echo "host=$(hostname) start=$(date -Is) port=$PORT model=$MODEL_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
n=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l); echo "safetensors shards: $n"
# Pipeline parallelism instead of tensor parallelism: with TP=2 on H200 nodes vLLM
# JIT-builds FlashInfer's trtllm_mnnvl all-reduce during CUDA-graph capture and that
# build fails against nvcc 13.4 ("CUDA compiler and CUDA toolkit headers are
# incompatible"; jobs 25539226 and 25539697, --disable-custom-all-reduce did not
# help). PP=2 only needs point-to-point sends, no all-reduce kernel. Slower per
# request, irrelevant for a judge that answers one short label.
"$VLLM" serve "$MODEL_DIR" --served-model-name "$NAME" --pipeline-parallel-size 2 \
  --dtype float16 \
  --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/judge/vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died"; tail -40 "$RESULTS/judge/vllm_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
T0=$(date +%s)
"$PY" "$CODE/runner/turtle_judge_eval.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" \
  --turtlebench "$HOME/TurtleBench" --concurrency 32 --out "$RESULTS/judge/turtle150_405b_int4.json"
echo "calibration done in $(( $(date +%s) - T0 ))s"
echo "$(hostname):$PORT" > "$EP"; echo "endpoint published: $(cat $EP)"
wait $PID
