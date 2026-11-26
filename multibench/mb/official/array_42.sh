#!/bin/bash
# Official default agents on the 42-candidate list (the part whose runners exist): one task per array
# job (one GPU, via serve.sbatch). Each job runs the search split (20 episodes, for choosing the
# evolution tasks) and then the full test split (for reporting). Tasks whose test run is already done
# only get the search run. Runners resume from disk.
#   CMD="bash ~/code/multibench/mb/official/array_42.sh" sbatch --array=0-34 --job-name=off_42 serve.sbatch
set -u
O=/data/team-bath/tmp_l/official        # test results (as before)
S=/data/team-bath/tmp_l/official_search  # search-split screen
BAL="env PYTHONPATH=$HOME/code/multibench $HOME/venvs/balrog/bin/python"
TA="python -m mb.official.run_textarena --agent standard"
BA="$BAL -m mb.official.run_balrog --agent naive"
AL="python -m mb.official.run_react_alfworld --agent react"
SW="python -m mb.official.run_swiftsage --agent react"
AR() { OUT=$1 SPLIT=$2 TASKS=$3 SHARDS=$4 METHODS=zero_shot bash $HOME/code/multibench/mb/official/arbench_inline.sh; }
J=(
  # done on test: search only
  "AR $S/arbench search gn,dc 5"
  "$TA --out $S/textarena --split search --n 20 --cells wordle:basic5,mastermind:basic,hangman:basic,twentyq:basic-things"
  "$BA --out $S/balrog --level GoToLocal --split search --n 20"
  "$AL --out $S/alfworld --subtask pick_and_place_simple --split search --n 20"
  "$SW --out $S/sciworld --task test-conductivity --split search --n 20"
)
for c in wordle:hard5 wordle:basic7 mastermind:hard mastermind:extreme hangman:hard minesweeper:basic minesweeper:medium minesweeper:hard twentyq:basic-places twentyq:basic-people twentyq:hard-people; do
  J+=("$TA --out $S/textarena --split search --n 20 --cells $c && $TA --out $O/textarena --split test --n 100000 --cells $c")
done
for l in PickupLoc Open Unlock SynthLoc GoToSeq; do
  J+=("$BA --out $S/balrog --level $l --split search --n 20 && $BA --out $O/balrog --level $l --split test --n 200")
done
for t in look_at_obj_in_light pick_clean_then_place_in_recep pick_heat_then_place_in_recep pick_two_obj_and_place; do
  J+=("$AL --out $S/alfworld --subtask $t --split search --n 20 && $AL --out $O/alfworld --subtask $t --split test")
done
for t in measure-melting-point-unknown-substance power-component grow-plant lifespan-longest-lived-then-shortest-lived inclined-plane-determine-angle mendelian-genetics-unknown-plant; do
  J+=("$SW --out $S/sciworld --task $t --split search --n 20 && $SW --out $O/sciworld --task $t --split test --n 100000")
done
if [ "${1:-}" = "--count" ]; then echo ${#J[@]}; exit 0; fi
echo "task ${SLURM_ARRAY_TASK_ID}: ${J[$SLURM_ARRAY_TASK_ID]}"
eval "${J[$SLURM_ARRAY_TASK_ID]}"
