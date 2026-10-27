# Candidates discarded on 2026-10-27

Kept as evidence, out of `agents/` so an rsync back to the cluster cannot redeploy them.

All five were written by proposer sessions that ran before two fixes landed: the held-out split
moved to `~/sealed/`, and UoT was removed from the tree the proposer can read. An audit of the
session transcripts found dc's proposer reading `test.json` on its eighth command and gn's reading
AR-Bench's UoT prompts.

| file | task | why discarded |
|---|---|---|
| `survivor_menu.py` | gn | maintains the full consistent answer set and cites UoT for it by name; the mechanism arrived by reading, not by search |
| `plan_queue.py` | gn | same session, same exposure |
| `contradiction_hunt.py` | dc | written while the held-out split was readable |
| `prior_gated_verify.py` | dc | same |
| `witness_dossier.py` | dc | same |

None had been evaluated: no run had completed an iteration, and `evolution_summary.jsonl` was
empty for all three tasks. The three tasks restarted from seeds only.

Full session transcripts: `~/Desktop/ar_harness_sealed/proposer_sessions_2026-10-27/`
(off the cluster, so no later proposer can read them).
