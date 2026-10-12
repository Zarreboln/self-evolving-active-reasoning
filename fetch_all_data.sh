#!/bin/bash
# Fetch every dataset this project uses, into $ROOT. Idempotent.
#
#   AR-Bench            github.com/tmlr-group/AR-Bench   (code + data/{gn,sp,dc})
#   Multi-Turn Puzzles  hf.co/datasets/arianhosseini/mt_puzzles  (CC BY 4.0,
#                       data only -- no code was ever released)
set -uo pipefail
ROOT=${1:-$HOME/code/ar_harness}
mkdir -p "$ROOT/mt_puzzles"
cd "$ROOT"

echo "=== AR-Bench ==="
if [ -d AR-Bench/.git ]; then
  git -C AR-Bench pull --ff-only -q 2>/dev/null && echo "  updated" || echo "  已存在"
else
  git clone --depth 1 -q https://github.com/tmlr-group/AR-Bench.git && echo "  cloned"
fi
for f in gn sp dc; do
  printf "  data/%s/test.json: " "$f"
  [ -f "AR-Bench/data/$f/test.json" ] \
    && python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))),'条')" \
         "AR-Bench/data/$f/test.json" 2>/dev/null || echo "缺"
done

echo "=== Multi-Turn Puzzles ==="
BASE=https://huggingface.co/datasets/arianhosseini/mt_puzzles/resolve/main
declare -A TAG=( [circuit_decoding]=cd [word_guess]=wg [twenty_questions]=tq
                 [word_chaining]=wc [movie_recommendation]=mr )
for fam in "${!TAG[@]}"; do
  t=${TAG[$fam]}
  for f in instruction.txt last_turn_prompt.txt test.jsonl; do
    out="mt_puzzles/${t}_${f}"
    [ -s "$out" ] && continue
    code=$(curl -sL -w "%{http_code}" -o "$out" "$BASE/$fam/$f")
    [ "$code" = "200" ] || rm -f "$out"      # several families have no last_turn_prompt
  done
  n=$(wc -l < "mt_puzzles/${t}_test.jsonl" 2>/dev/null || echo 0)
  printf "  %-22s %s 条  (%s)\n" "$fam" "$n" \
    "$(ls mt_puzzles/${t}_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
done
curl -sL -o mt_puzzles/README.md "$BASE/README.md" 2>/dev/null

echo "=== 总计 ==="
du -sh AR-Bench mt_puzzles 2>/dev/null
