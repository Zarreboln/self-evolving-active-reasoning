#!/bin/bash
# Move the partial Llama-3.1-405B-Instruct bf16 download from scratch (per-user
# quota ~1 TB, hit on 2026-10-10 20:07) to pool, freeing scratch for the INT4
# judge model. rsync --remove-source-files frees space file by file.
#SBATCH --job-name=mvbf16
#SBATCH --partition=mit_data_transfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
SRC=/orcd/scratch/orcd/012/$USER/models/Llama-3.1-405B-Instruct
DST=/orcd/pool/007/$USER/models/Llama-3.1-405B-Instruct
mkdir -p "$DST"
echo "host=$(hostname) start=$(date -Is)"; du -sh "$SRC"
rsync -a --remove-source-files --info=progress2 "$SRC/" "$DST/" 2>&1 | tail -3
find "$SRC" -type d -empty -delete 2>/dev/null
echo "end=$(date -Is)"; du -sh "$SRC" "$DST" 2>/dev/null
