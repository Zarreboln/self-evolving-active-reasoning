#!/bin/bash
# Resume one Llama-3.1-405B-Instruct variant download on a data-transfer node
# instead of the login node (the 2026-10-09 attempt, 3x8 workers on login008,
# saturated the login node and got killed with ~3-5 GB landed per variant).
#
#   sbatch engaging/dl_405b.sbatch fp8     # 487 GB -> /orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct-FP8
#   sbatch engaging/dl_405b.sbatch int4    # 219 GB -> /orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct-INT4
#   sbatch engaging/dl_405b.sbatch bf16    # 812 GB -> /orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct
#
# hf download resumes the *.incomplete blobs already in <dir>/.cache/huggingface.
# 2026-10-10: FP8/INT4 moved from /orcd/pool to /orcd/scratch -- on pool the resumed
# downloads sat on stale NFS file locks and wrote nothing for 15 min, while the bf16
# one on scratch pulled 800 GB in the same window.
#
#SBATCH --job-name=dl405b
#SBATCH --partition=mit_data_transfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%a_%j.out
#SBATCH --error=%x_%a_%j.err
set -uo pipefail
V=${1:?fp8|int4|bf16}
case "$V" in
  fp8)  REPO=meta-llama/Llama-3.1-405B-Instruct-FP8;            DIR=/orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct-FP8 ;;
  int4) REPO=hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4; DIR=/orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct-INT4 ;;
  bf16) REPO=meta-llama/Llama-3.1-405B-Instruct;                DIR=/orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct ;;
  *) echo "unknown variant $V"; exit 2 ;;
esac
HF=$HOME/code/ar_harness/conda_env/bin/hf
[ -x "$HF" ] || HF=$(command -v hf || command -v huggingface-cli)
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "host=$(hostname) start=$(date -Is) repo=$REPO dir=$DIR hf=$HF"
du -sh "$DIR" 2>/dev/null
"$HF" download "$REPO" --local-dir "$DIR" --max-workers 4 --exclude "original/*" 2>&1 | tr '\r' '\n' | grep -v "Still waiting to acquire lock"
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
du -sh "$DIR"
