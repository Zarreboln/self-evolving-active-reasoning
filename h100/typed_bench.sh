#!/bin/bash
# Serve one model on H100 GPU 3 and score it on TurtleBench-150 twice:
# once by sampling a word (turtle_judge_eval.py) and once by reading the three
# label probabilities off the logits (typed_judge_eval.py).
set -uo pipefail
REPO=${REPO:?}; TAG=${TAG:?}; PORT=${PORT:-8123}
CODE=$HOME/code/ar_harness; VENV=$CODE/venv_cu128
export PATH=$CODE/venv_cu128/bin:$PATH
export CUDA_VISIBLE_DEVICES=3 HF_HOME=/home/tmp_l/ckpt/hf_home VLLM_USE_DEEP_GEMM=0 VLLM_LOGGING_LEVEL=WARNING
L=$HOME/logs; mkdir -p "$L" "$HOME/results/judge"
echo "host=$(hostname) repo=$REPO start=$(date -Is) port=$PORT"
$VENV/bin/vllm serve "$REPO" --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 --enable-prefix-caching ${VARGS:-} > "$L/vllm_$TAG.log" 2>&1 &
VP=$!
for i in $(seq 1 120); do
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 15
done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve failed"; tail -25 "$L/vllm_$TAG.log"; kill $VP; exit 1; }
echo "up after $SECONDS s"
echo "=== sampled word (the usual referee protocol) ==="
$VENV/bin/python $CODE/runner/turtle_judge_eval.py --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$REPO" --turtlebench $HOME/TurtleBench --concurrency 16 ${EARGS:-} \
  --out "$HOME/results/judge/turtle150_${TAG}_sampled.json"
echo
echo "=== sampled word, temperature 0 (matched to the typed read) ==="
$VENV/bin/python $CODE/runner/turtle_judge_eval.py --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$REPO" --turtlebench $HOME/TurtleBench --concurrency 16 --temperature 0 ${EARGS:-} \
  --out "$HOME/results/judge/turtle150_${TAG}_sampled_t0.json"
echo
echo "=== typed read (label probabilities off the logits) ==="
$VENV/bin/python $CODE/runner/typed_judge_eval.py --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$REPO" --turtlebench $HOME/TurtleBench --concurrency 16 ${TARGS:-} \
  --out "$HOME/results/judge/turtle150_${TAG}_typed.json"
kill $VP 2>/dev/null; echo "done $(date -Is), total ${SECONDS}s"
