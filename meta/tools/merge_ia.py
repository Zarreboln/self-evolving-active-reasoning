"""Put the re-measured information-acquisition column into the curves file the figure reads.

The first column changed on every task. Belief tracking did not: it is the same measurement
everywhere and is left exactly as it was.

  gn  unchanged in substance -- the posterior mass 1/|H_t| -- but recomputed by the same tool as
      the other two, which reproduces the published curve point for point and so certifies that
      the tool is not introducing a difference of its own.
  dc  distinct suspects interrogated, of five  ->  key questions covered, of six. The old proxy
      saturates: both harnesses sit at 100 from turn 12 on, so the turn-25 number was a ceiling,
      not a measurement.
  sp  decisive answers as a share of the budget  ->  key questions covered, of six. The old proxy
      divides by a fixed 25 whatever the episode's length, so a harness that stops early scores
      low for stopping rather than for learning less, and the evolved harness stops early.

Both readings of the old IA are kept under `ia_old_*`, and the hit rate -- what share of the
questions asked were worth asking -- is kept alongside, so neither has to be recomputed to be
argued about.

    python meta/tools/merge_ia.py notes/arbench_mh/ia_bt_curves.json gn_cov_sweep.json dc_... sp_...
"""
import json
import sys

PAIRS = {"gn_template_zero_shot": "gn_template_zero_shot", "gn_packed_group_echo": "gn_packed_group_echo",
         "dc_template_zero_shot": "dc_template_zero_shot", "dc_lean_verify": "dc_lean_verify",
         "sp_baseline_zero_shot": "sp_baseline_zero_shot", "sp_atomic_audit": "sp_atomic_audit"}

cur = json.load(open(sys.argv[1]))
new = {}
for p in sys.argv[2:]:
    new.update(json.load(open(p)))

for k, src in PAIRS.items():
    if src not in new or k not in cur:
        continue
    assert cur[k]["turns"] == new[src]["turns"], k
    cur[k].setdefault("ia_old", cur[k]["ia"])
    cur[k]["ia"] = new[src]["coverage"]
    cur[k]["hit_rate"] = new[src]["hit_rate"]
    cur[k]["ia_n"] = new[src]["n"]
    print("%-24s turn25: %5.1f -> %5.1f   hit rate %5.1f"
          % (k, cur[k]["ia_old"][-1], cur[k]["ia"][-1], new[src]["hit_rate"][-1]))

json.dump(cur, open(sys.argv[1], "w"))
