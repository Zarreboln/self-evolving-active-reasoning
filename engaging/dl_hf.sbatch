#!/bin/bash
# Download one HF repo into the shared vLLM cache on a data-transfer node.
#   sbatch engaging/dl_hf.sbatch Qwen/Qwen2.5-32B-Instruct-AWQ
#SBATCH --job-name=dlhf
#SBATCH --partition=mit_data_transfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
REPO=${1:?hf repo id}
export HF_HOME=$HOME/orcd/pool/ar_harness/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
HF=$HOME/code/ar_harness/conda_env/bin/hf
echo "host=$(hostname) start=$(date -Is) repo=$REPO HF_HOME=$HF_HOME"
"$HF" download "$REPO" --max-workers 4 2>&1 | tr '\r' '\n' | grep -v "Still waiting to acquire lock" | tail -5
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
du -sh "$HF_HOME/hub/models--${REPO//\//--}"
