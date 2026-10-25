#!/bin/bash
# Pick the queue back up after its driver was killed while gn/uot was mid-flight.
#
# The driver died but its 16 shard workers were reparented to init and kept running, so no episode
# was lost -- what was lost is the merge at the end of the combo and the move to the next one. This
# waits for the orphans to exit, merges what they wrote, and restarts the driver, which skips every
# combo already at 100 and continues with gn/tot, then sp, then dc.
set -uo pipefail
PY=$HOME/venv/bin/python
while [ "$(ps -u tmp_l -o cmd | grep -c '[m]ethod uot')" -gt 0 ]; do sleep 60; done
echo "gn/uot workers finished $(date -Is)"
$PY "$HOME/code/shard_tool.py" merge "$HOME/results/shards/gn_uot" "$HOME/results/gn_uot.json"
cd "$HOME/code" || exit 1
exec bash run_all_sharded.sh
