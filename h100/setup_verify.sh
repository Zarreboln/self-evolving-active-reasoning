#!/bin/bash
# H100 vLLM environment: fixed 2026-10-06. The venv at ~/code/ar_harness/venv
# has torch 2.13.0+cu130, which this node's driver (570.195.03, CUDA 12.8)
# cannot run ("The NVIDIA driver on your system is too old"). torch wheels
# released after a certain point default to cu130; vLLM 0.20.0+ pulls one of
# those. vLLM 0.19.1 is the newest release still pinned to torch==2.10.0,
# whose CUDA deps (nvidia-cublas-cu12==12.8.4.1, nvidia-cuda-nvrtc-cu12==
# 12.8.93) match this driver exactly.
#
# Unlike Engaging/ORCD, this node's SYSTEM CUDA toolkit already matches:
#   /usr/bin/nvcc -> release 12.8, V12.8.93 (identical to the pip-pinned
#   version), /usr/lib/cuda has bin/include/lib64/nvvm complete. No conda,
#   no separate CUDA_HOME package juggling needed -- just point at /usr/lib/cuda.
#
# GPU policy: CUDA_VISIBLE_DEVICES=3 always (the only card we're allowed to
# use on this shared box). At verification time GPU 3 had another user's job
# holding ~25GB of 81GB; --gpu-memory-utilization is set low (0.3) so this
# fits alongside it without crowding it out, and this script tears the server
# down immediately after one health check + one completion.
set -euo pipefail
CODE=$HOME/code/ar_harness
VENV=$CODE/venv_cu128
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/lib/cuda
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0   # same fix as Engaging, defensive here too

echo "=== preflight ==="
for t in nvcc ninja g++ python vllm; do
  p=$(command -v $t 2>/dev/null) && echo "  ok   $t -> $p" || { echo "  MISSING $t"; exit 2; }
done
"$CUDA_HOME/bin/nvcc" --version | tail -1
"$VENV/bin/python" -c "import vllm, torch; print('  ok  ', 'vllm', vllm.__version__, 'torch', torch.__version__)"
"$VENV/bin/python" -c "import torch; print('  ok   GPU3:', torch.cuda.get_device_name(0), 'free', round(torch.cuda.mem_get_info()[0]/2**30,1), 'GiB')"

PORT=18842
"$VENV/bin/vllm" serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.3 \
  --port "$PORT" > /tmp/h100_verify_vllm.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT

echo "=== waiting for vLLM (pid $PID) ==="
ok=0
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died -- see /tmp/h100_verify_vllm.log"; tail -40 /tmp/h100_verify_vllm.log; exit 1; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "timed out"; tail -40 /tmp/h100_verify_vllm.log; exit 1; }

echo "=== one completion ==="
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role":"user","content":"Say OK."}],
  "max_tokens": 10
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(' ', d['choices'][0]['message']['content'])"

echo "H100 VERIFIED WORKING"
