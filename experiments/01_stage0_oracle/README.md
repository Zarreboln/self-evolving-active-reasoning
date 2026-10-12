# 01 — Stage-0: hypothesis space, oracle and synthetic policies (2026-10-06, CPU only)

**Code**: `stage0/cd_oracle.py`, `stage0/survey_families.py`. **Data**: `mt_puzzles/cd_test.jsonl`.

The gate inventory the agent is told implies a finite set of joint truth tables: each circuit
reduces 3 inputs to 1 output with exactly 2 binary gates, NOT gates are unconstrained in count,
no circuit is constant. Enumerating wirings per circuit and the Cartesian product across the
three circuits gives |H| = 252 / 2886 / 3924 for the three released inventories
({3 AND,3 OR,0 NOT} / {1 AND,5 OR,2 NOT} / {3 AND,3 OR,1 NOT}). All 300 released truth tables
are inside their inventory's space (300/300; row order checked on all 900 circuits).

Synthetic policies on the 300 instances, 18-query budget:

| policy | solved | queries | ex-ante eff | zero-info queries |
|---|---|---|---|---|
| greedy max-entropy (the oracle) | 300/300 | 11.60 | 1.000 | 0.0% |
| greedy over 4 sampled queries | 250/300 | 13.92 | 0.775 | 18.1% |
| random fresh entry | 156/300 | 16.51 | 0.629 | 27.7% |
| sequential scan | 58/300 | 17.41 | 0.540 | 38.1% |
| uniform random (repeats allowed) | 35/300 | 17.63 | 0.480 | 44.1% |

Takeaway: the task has large headroom with a dense, computable per-query signal (bits over the
feasible set), which is why it is the main testbed; GuessNumbers has no "legal but
uninformative" move and saturated at 99% with a hand-written harness (see `experiments.md`,
2026-10-06).
