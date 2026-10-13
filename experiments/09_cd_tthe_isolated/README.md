# 09 — Test-time harness evolution with label isolation (2026-10-12 19:01–22:33 EDT)

Same protocol as [08](../08_cd_tthe_pilot/README.md) (TTHE port, B = 10, G = 3, R = 2, 30 held-out
instances, stream seed 0, Qwen2.5-7B-Instruct questioner, Claude Opus 5.5 proposer and judge inside
Claude Code), rerun after the pilot's label leaks were closed: gold records and oracle-derived signals
live in a controller-private store outside the repository, the run directory holds public traces only,
memory directories are emptied per session, the sandbox contains nothing but the current run, and every
proposer/judge session is audited automatically (paths resolved against the shell's working directory;
a violating proposer's candidate or judge's verdict is discarded). Runs `strict6_b10g3r2` (run status
only) and `labeled6_b10g3r2` (plus each episode's outcome). Both were interrupted at 19:50 by the
Claude Max session limit and resumed from their last valid point (`--resume-from`) at 21:37 on a fresh
7B server; the void rows were discarded first. Metrics: official circuit-wise accuracy (cwa) and joint
truth table correct (joint), see [`../OFFICIAL_DEFINITIONS.md`](../OFFICIAL_DEFINITIONS.md).

## Streams (prequential = incoming harness on the new batch before adaptation)

**strict** (label-free)

| batch | incoming → committed | cwa | joint | transductive joint | pool | wall |
|---|---|---|---|---|---|---|
| 0 | bare → b0r1g1 | 7 | 0 | 100 | 5 | 2577 s |
| 1 | b0r1g1 → b1r1g0 | 97 | 90 | 90 | 6 | 1080 s |
| 2 | b1r1g0 → b2r0g1 | 100 | 100 | 100 | 6 | 2188 s |

**labeled**

| batch | incoming → committed | cwa | joint | transductive joint | pool | wall |
|---|---|---|---|---|---|---|
| 0 | bare → b0r1g2 | 10 | 0 | 100 | 7 | 50 s |
| 1 | b0r1g2 → b1r0g2 | 100 | 100 | 100 | 6 | 1176 s |
| 2 | b1r0g2 → b1r0g2 | 100 | 100 | 100 | 6 | 1892 s |

Stream totals (30 instances): strict prequential cwa 67.8 / joint 63.3,
transductive 98.9 / 96.7; labeled prequential
70.0 / 66.7, transductive 100.0 / 100.0.
Audits: strict 29/32 sessions clean (3 proposer candidates discarded: one stray `cat >` that wrote an empty
file to the Desktop — real; two false positives of the audit, since fixed — an over-broad `solver_cache`
name rule and an untracked `cd agents` before `python3 ../audit.py`), every judge session clean; labeled
36/36 clean. Sessions and subscription-priced cost: strict 23 sessions $7.99
(+0 limit-hit stubs), labeled 31 sessions $7.88 (+0).

**What the strict judge did.** Batch 0: built its own label-free checker (enumerate rule-allowed circuit
triples, filter by the answers, check uniqueness and that the guess is the unique solution), found the
menu harness (r0g2) and the deduction-board harness (r1g1) tied at 10/10 and chose the board because
the model took the menu's top-ranked query in 113 of 126 turns ("the query-choosing the interface
forbids"). Batch 1: rejected three 10/10 candidates for the same reason ("closest to 50 % now" hints
followed ~95 % of the time; one harness picked the best-splitting of five model-listed queries) and
committed the 9/10 board variant that skips cells the board already knows. Batch 2: six identical,
answer-consistent guess sets; chose the variant that used fewer queries. Verdicts in `judge_verdicts/`.

## Frozen evaluation of the finished harnesses: 200 held-out instances, no proposer, no judge

`meta/cd_tthe/evaluate.py`, 48 workers, 22:34–23:4x. "unseen" = the 170 instances the streams never
used. CIs are bootstrap 95 %.

| harness | cwa | joint | joint (unseen 170) | uniq@end (ours) | queries/ep | tokens/ep | crashes |
|---|---|---|---|---|---|---|---|
| no harness (official protocol) | 11.8 [9.2,14.5] | 0.0 [0.0,0.0] | 0.0 | 26.5 | 17.2 | 2675 | 0 |
| strict b2 (final): same family, fewer queries | 99.5 [98.8,100.0] | 98.5 [96.5,100.0] | 98.2 | 99.0 | 12.0 | 1568 | 0 |
| strict b1: board + skip cells the board knows + rule-allowed tables at the end + no early-guess loop | 99.3 [98.7,99.8] | 98.0 [96.0,99.5] | 98.2 | 95.5 | 13.0 | 2150 | 0 |
| strict b0: deduction board (observed / rule-forced / open outputs), no ranking, no vetoes | 97.0 [95.5,98.3] | 92.0 [88.0,96.0] | 91.8 | 87.5 | 13.8 | 2519 | 1 |
| labeled b1 (final): same + final answer in three 8-bit fields | 100.0 [100.0,100.0] | 100.0 [100.0,100.0] | 100.0 | 100.0 | 11.5 | 215 | 0 |
| labeled b0: joint-space enumerator + reject queries that split worse than the best | 100.0 [100.0,100.0] | 100.0 [100.0,100.0] | 100.0 | 100.0 | 11.5 | 207 | 0 |
| pilot strict b0 (experiment 08): candidate tables + menu of 8 | 99.5 [98.5,100.0] | 99.5 [98.5,100.0] | 99.4 | 99.0 | 12.6 | 653 | 1 |
| strict b0, not chosen: candidate tables + split menu (top-1 taken 113/126 turns) | 97.5 [96.0,98.8] | 93.5 [90.0,97.0] | 92.4 | 99.5 | 12.0 | 406 | 1 |
| strict b1, rejected as query-choosing: 'closest to 50 %' hints | 99.5 [98.5,100.0] | 99.5 [98.5,100.0] | 99.4 | 99.5 | 11.4 | 1284 | 0 |
| strict b1, rejected: harness picks the best-splitting of 5 model-listed queries | 99.0 [98.2,99.7] | 97.0 [94.5,99.0] | 96.5 | 97.0 | 12.8 | 1195 | 0 |
| strict b1, rejected: 'closest to 50 %' hints (child of b1r0g2) | 99.5 [98.5,100.0] | 99.5 [98.5,100.0] | 99.4 | 99.5 | 11.4 | 1196 | 1 |

**Reading.** Every evolved harness is within two points of the oracle ceiling on 200 instances; the
10-instance batches of the stream understated them (b0r1g1 scored 9/10 on batch 1 and 92.0 on 200).
The state-only family that the strict judge enforced (board, no ranking, the model writes every query)
reaches 98.5 joint / 99.5 cwa with 12 queries per episode; the harnesses the judge rejected for steering
the queries reach 99.5, and the labeled arm's enumerator with query vetoes reaches 100.0 at 215 output
tokens per episode. The gap between "model chooses from a visible belief state" and "harness chooses"
is about one point on this task. bare: 11.8 cwa, 0 joint.

Files: `result_*.json`, `opt_log_*.jsonl`, `branch_lineage_*.jsonl`, `session_audits_*.jsonl`,
`trajectory_*.png`, `judge_verdicts/`, `proposal_cards/`, `candidates/`, `eval_heldout200_scores.json`.
Raw traces, session transcripts, controller records and per-episode evaluation records: release asset
`cd_tthe_isolated_raw.tar.zst`.
