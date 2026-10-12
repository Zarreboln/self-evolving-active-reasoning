#!/bin/bash
# Engaging one-time setup. Run from the LOGIN node (I/O only, no computation)
# or submit to mit_data_transfer if the login node objects to the download size.
#
#   bash ~/code/ar_harness/engaging/setup.sh
#
# Layout follows the storage rules: code + results in /home (200 GB, backed up),
# the 15 GB model in ~/orcd/pool (1 TB, no backup, regenerable).
set -euo pipefail

CODE=$HOME/code/ar_harness
POOL=$HOME/orcd/pool/ar_harness
export HF_HOME=$POOL/hf_cache

mkdir -p "$CODE" "$POOL" "$HF_HOME" "$HOME/logs"
cd "$CODE"

echo "=== internet check ==="
curl -sS -m 10 -o /dev/null -w "huggingface.co %{http_code}\n" \
  https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct \
  || { echo "NO OUTBOUND INTERNET from this node -- run setup on the login node"; exit 2; }

echo "=== AR-Bench ==="
[ -d AR-Bench ] || git clone --depth 1 https://github.com/tmlr-group/AR-Bench.git

# Login-node python3 is 3.6.8 -- far too old for vLLM wheels ("from versions:
# none"). /usr/bin/python3.12 exists; module miniforge/26.7.2-0 is the fallback.
PY=$(command -v python3.12 || command -v python3.11 || echo python3)
echo "=== venv ($PY -> $($PY -V 2>&1)) ==="
"$PY" -c 'import sys; assert sys.version_info >= (3,10), f"need >=3.10, got {sys.version}"'
[ -d venv ] || "$PY" -m venv venv
./venv/bin/python -m pip install -q --upgrade pip wheel
./venv/bin/python -m pip install -q vllm
./venv/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"

echo "=== model -> $HF_HOME (15.24 GB) ==="
./venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                      allow_patterns=["*.safetensors", "*.json", "*.txt"])
print("LOCAL_PATH", p)
PY

echo "=== footprint ==="
du -sh "$CODE/venv" "$HF_HOME" 2>/dev/null
echo "setup OK"
