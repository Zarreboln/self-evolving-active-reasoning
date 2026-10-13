#!/bin/bash
# Download one HF repo into the scratch HF cache (pool NFS locks stall hf downloads).
#   sbatch engaging/dl_hf_scratch.sbatch openai/gpt-oss-120b
#SBATCH --job-name=dlhf
#SBATCH --partition=mit_data_transfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
REPO=${1:?hf repo id}
export HF_HOME=/orcd/scratch/orcd/012/$USER/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
HF=$HOME/code/ar_harness/conda_env/bin/hf
mkdir -p "$HF_HOME"
echo "host=$(hostname) start=$(date -Is) repo=$REPO HF_HOME=$HF_HOME"
"$HF" download "$REPO" --max-workers 8 2>&1 | tr '\r' '\n' | grep -v "Still waiting to acquire lock" | tail -5
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
du -sh "$HF_HOME/hub/models--${REPO//\//--}"
