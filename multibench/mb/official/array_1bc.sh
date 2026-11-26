#!/bin/bash
# Steps 1b-1c full test split as a job array of one-GPU tasks (via serve.sbatch), each benchmark's default
# agent. Task t runs one (runner, shard) pair; runners shard by a stable hash and resume from disk.
#   CMD="bash ~/code/multibench/mb/official/array_1bc.sh" sbatch --array=0-8 --job-name=off_1bc serve.sbatch
set -u
O=/data/team-bath/tmp_l/official
BAL="env PYTHONPATH=$HOME/code/multibench $HOME/venvs/balrog/bin/python"
T=(
  "python -m mb.official.run_textarena --out $O/textarena --agent standard --split test --n 100000 --shard 0/3"
  "python -m mb.official.run_textarena --out $O/textarena --agent standard --split test --n 100000 --shard 1/3"
  "python -m mb.official.run_textarena --out $O/textarena --agent standard --split test --n 100000 --shard 2/3"
  "$BAL -m mb.official.run_balrog --out $O/balrog --agent naive --level GoToLocal --split test --n 200 --shard 0/2"
  "$BAL -m mb.official.run_balrog --out $O/balrog --agent naive --level GoToLocal --split test --n 200 --shard 1/2"
  "python -m mb.official.run_react_alfworld --out $O/alfworld --agent react --split test"
  "python -m mb.official.run_swiftsage --out $O/sciworld --agent react --split test --n 100000 --shard 0/3"
  "python -m mb.official.run_swiftsage --out $O/sciworld --agent react --split test --n 100000 --shard 1/3"
  "python -m mb.official.run_swiftsage --out $O/sciworld --agent react --split test --n 100000 --shard 2/3"
)
echo "task ${SLURM_ARRAY_TASK_ID}: ${T[$SLURM_ARRAY_TASK_ID]}"
eval "${T[$SLURM_ARRAY_TASK_ID]}"
