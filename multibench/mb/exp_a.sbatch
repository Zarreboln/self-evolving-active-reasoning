#!/bin/bash
# Experiment A: five hand-written harness components (mb/harness/strategies.py) x ten subtasks,
# N search instances per cell, paired (same instances for every strategy). Qwen3.5-9B plays the
# policy and DC's suspects; Jev plays SP / 20Q / Guess Who. Before A, the Jev rerun of the screen's
# seed baselines for the three benches whose responder changed from Qwen to Jev.
# Two GPUs, one server each; lowprio + requeue, and mb.screen resumes from the episodes on disk.
#   sbatch ~/code/multibench/mb/exp_a.sbatch            # N=30
#   N=1 OUTA=~/results/mb/exp_a_smoke JEV=0 sbatch -t 1:00:00 ~/code/multibench/mb/exp_a.sbatch
#SBATCH --job-name=mb_exp_a
#SBATCH --partition=h100_1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --qos=lowprio
#SBATCH --nodelist=ml-16-node-001
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
echo "host=$(hostname) start=$(date -Is) restart=${SLURM_RESTART_COUNT:-0}"
BASE=$(( 9000 + RANDOM % 800 ))
SERVERS=(); URLS=()
for g in 0 1; do
  PORT=$((BASE + g))
  CUDA_VISIBLE_DEVICES=$g $HOME/venv/bin/vllm serve "$REPO" --port "$PORT" --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$HOME/logs/vllm_mb_exp_a_${SLURM_JOB_ID}_g$g.log" 2>&1 &
  SERVERS+=($!); URLS+=("http://127.0.0.1:$PORT/v1")
done
for u in "${URLS[@]}"; do
  for i in $(seq 1 120); do curl -sf -m 5 "${u%/v1}/health" >/dev/null && break; sleep 10; done
  curl -sf -m 5 "${u%/v1}/health" >/dev/null || { echo "serve failed: $u"; kill "${SERVERS[@]}"; exit 1; }
done
echo "servers up after ${SECONDS}s"

cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_MODEL="$REPO" POLICY_BASE_URLS=$(IFS=,; echo "${URLS[*]}")
N=${N:-30}; OUTA=${OUTA:-$HOME/results/mb/exp_a}
A_CELLS=gn:all,dc:all,mastermind:basic,hangman:basic,twentyq:basic-things,wordle:basic5,blicket:n5,sciworld:test-conductivity,alfworld:pick_and_place_simple,babyai:GoToLocal
JEV_CELLS=twentyq:basic-places,twentyq:basic-people,twentyq:basic-things,twentyq:hard-places,twentyq:hard-people,twentyq:hard-things,sp:all,guesswho:all
if [ "${JEV:-1}" = 1 ]; then
  python -m mb.screen --out $HOME/results/mb/screen_v1_jev --only "$JEV_CELLS" --n 20 --workers 64
  python -m mb.screen --out $HOME/results/mb/screen_v1_jev --only "$JEV_CELLS" --n 20 --merge
fi
python -m mb.screen --out "$OUTA" --strategies seed,belief,plan,check,brief --only "$A_CELLS" --n "$N" --workers 64
python -m mb.screen --out "$OUTA" --strategies seed,belief,plan,check,brief --only "$A_CELLS" --n "$N" --merge
echo "end=$(date -Is)"
kill "${SERVERS[@]}"
