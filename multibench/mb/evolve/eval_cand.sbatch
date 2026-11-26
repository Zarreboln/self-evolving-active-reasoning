#!/bin/bash
# Experiment B: score one candidate harness on one split of one cell (mb/evolve/broker.py), then
# export what the proposer may read. One GPU, one vLLM server; lowprio + requeue, and the broker
# resumes from the episodes on disk. Partition, QOS and GPU type come from the caller
# (remote.SBATCH_ARGS); cluster paths from ~/.mb_site.sh if present.
#   CAND=dir OUT=dir DEST=dir [CELL=wordle:basic5 SPLIT=search N=30 SMOKE=1] sbatch eval_cand.sbatch
#SBATCH --job-name=mb_evo_b
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
[ -f "$HOME/.mb_site.sh" ] && source "$HOME/.mb_site.sh"
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-$HOME/hf_cache} VLLM_LOGGING_LEVEL=WARNING
echo "host=$(hostname) start=$(date -Is) restart=${SLURM_RESTART_COUNT:-0} cand=$CAND split=${SPLIT:-search}"
PORT=$(( 9000 + RANDOM % 800 ))
${VLLM_BIN:-$HOME/venv/bin/vllm} serve "$REPO" --port "$PORT" --max-model-len 65536 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > "$HOME/logs/vllm_mb_evo_b_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 10; done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve failed"; kill $SERVER; exit 1; }
echo "server up after ${SECONDS}s"

cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_MODEL="$REPO" POLICY_BASE_URLS="http://127.0.0.1:$PORT/v1"
# CELL may list several tasks (comma-separated): one sub-run per task, all against this one server,
# then a combined OUT/summary.json (or OUT/rejected.json if any task rejected the candidate).
IFS=, read -ra CELLS <<< "${CELL:-wordle:basic5}"
W=$(( 48 / ${#CELLS[@]} )); [ $W -lt 8 ] && W=8
for c in "${CELLS[@]}"; do
  sub="$OUT"; dst="${DEST:-}"
  if [ ${#CELLS[@]} -gt 1 ]; then sub="$OUT/cells/${c/:/_}"; [ -n "$dst" ] && dst="$DEST/cells/${c/:/_}"; fi
  cand="$CAND"; [ -n "${PER_CELL:-}" ] && cand="$CAND/${c/:/_}"
  ( python -m mb.evolve.broker run --cand "$cand" --cell "$c" --split "${SPLIT:-search}" \
      --n "${N:-30}" --out "$sub" ${SMOKE:+--smoke} --workers $W
    [ -n "$dst" ] && python -m mb.evolve.broker export --out "$sub" --dest "$dst" ) &
done
wait
[ ${#CELLS[@]} -gt 1 ] && python - "$OUT" <<'PY'
import json, os, statistics, sys
out = sys.argv[1]
cells = sorted(os.listdir(os.path.join(out, "cells")))
rej, summ = [], {}
for c in cells:
    d = os.path.join(out, "cells", c)
    if os.path.exists(os.path.join(d, "rejected.json")):
        rej += [f"{c}: {r}" for r in json.load(open(os.path.join(d, "rejected.json")))["rejected"]]
    elif os.path.exists(os.path.join(d, "summary.json")):
        summ[c] = json.load(open(os.path.join(d, "summary.json")))
if rej:
    json.dump({"rejected": rej}, open(os.path.join(out, "rejected.json"), "w"), indent=1)
elif len(summ) == len(cells):
    json.dump({"cells": summ, "infra_errors": sum(s["infra_errors"] for s in summ.values()),
               "mean_score": statistics.mean(s["mean_score"] for s in summ.values())},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
PY
echo "end=$(date -Is)"
touch "$OUT/DONE"
kill $SERVER
