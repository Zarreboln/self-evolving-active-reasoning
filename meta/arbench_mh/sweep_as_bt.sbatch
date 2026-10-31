#!/bin/bash
# Serve the policy, run the whole AS/BT sweep, exit. One job, no idle server.
#
# The grant went on servers, not evals: jobs 1972/1978/2123/2397/2632 burned ~89 GPU-hours and
# four of the five ended in TIMEOUT, i.e. they sat idle until the wall clock killed them, while
# the evals themselves take minutes. So the server's lifetime is this script, and nothing else.
#
# Everything runs on the SELECTION split. The question -- did evolution move AS or BT? -- needs
# the two harnesses on the same split, not on the sealed one, and test has been opened enough.
#
#   sbatch sweep_as_bt.sbatch
#SBATCH --job-name=sweep_as_bt
#SBATCH --partition=h100_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --qos=lowprio
#SBATCH --output=%x_%j.out
set -uo pipefail
export PATH=$HOME/venv/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
PORT=$(( 9000 + RANDOM % 900 ))
OUT=$HOME/results/as_bt; mkdir -p "$OUT" "$HOME/logs"
echo "host=$(hostname) port=$PORT start=$(date -Is)"

$HOME/venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct --port "$PORT" --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
  > "$HOME/logs/vllm_asbt_$SLURM_JOB_ID.log" 2>&1 &
VP=$!
trap 'kill $VP 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 10; done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve failed"; tail -25 "$HOME/logs/vllm_asbt_$SLURM_JOB_ID.log"; exit 1; }
echo "up after ${SECONDS}s"

export POLICY_BASE_URL="http://127.0.0.1:$PORT/v1"
cd $HOME/meta/arbench_mh

# gn: the published zero-shot loop is the "before", packed_group_echo the "after".
# dc: template_zero_shot is the "before"; both frontier members are the "after".
run () {  # task harness ablate
  local tag=$1_$2$( [ "$3" = 1 ] && echo _ablate )
  ABLATE_FEEDBACK=$3 python benchmark.py --task "$1" --harness "$2" \
    --split selection --workers 16 --out "$OUT/$tag.json" 2>&1 | sed "s/^/[$tag] /"
}
for h in template_zero_shot packed_group_echo; do run gn "$h" 0; run gn "$h" 1; done
for h in template_zero_shot prior_verify_stop lean_verify; do run dc "$h" 0; run dc "$h" 1; done

echo "=== AS/BT (gn only; dc has no computable consistent set)"
python $HOME/meta/tools/as_bt.py "$OUT"/gn_*.json
echo "=== feedback ablation: accuracy with every answer replaced by 'unknown'"
python - "$OUT" <<'PY'
import json, sys, glob, os, re
rows = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    k = os.path.basename(f)[:-5]
    r = json.load(open(f))
    rows[k] = 100.0 * sum(1 for x in r if x.get("correctness")) / len(r)
for k in sorted(k for k in rows if not k.endswith("_ablate")):
    a = rows.get(k + "_ablate")
    print("  %-34s real=%5.1f  ablated=%s  delta=%s"
          % (k, rows[k], "%5.1f" % a if a is not None else "  n/a",
             "%+5.1f" % (rows[k] - a) if a is not None else "  n/a"))
PY
echo "done $(date -Is)  elapsed=${SECONDS}s"
