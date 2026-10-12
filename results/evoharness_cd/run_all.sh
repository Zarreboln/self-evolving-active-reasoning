#!/bin/bash
cd ~/code/ar_harness/evoharness_rl_official
export AR_HARNESS_ROOT=$HOME/code/ar_harness
for seed in 0 1; do for mode in env_only always_on inline; do
  OUT=$HOME/results/ar_harness/evoharness_cd/${mode}_s${seed}
  echo "=== $mode seed $seed start=$(date -Is)"
  ../conda_env/bin/python -m evoharness eval --config configs/eval/circuitdecoding.yaml     policy.model=Qwen/Qwen2.5-7B-Instruct policy.base_url=http://node3600:8611/v1     harness.mode=$mode split=all out_dir=$OUT workers=32 2>&1 | grep -v -E "INFO .* won=" | tail -25
  echo "=== $mode seed $seed end=$(date -Is)"
done; done
echo ALL DONE
