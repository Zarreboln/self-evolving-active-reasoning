# 03 — Actionability ladder: seven hand-written harnesses (2026-10-10/11)

**Question**: how much of the gap between a 7B model and the oracle closes when the harness
hands over (a) its own observations, (b) the state, (c) a decision table, (d) the decision?

**Setup**: 300 instances × 2 seeds per harness, Qwen2.5-7B-Instruct on one L40S (vLLM),
`runner/cd_harness_run.py`, harness files in `meta/cd_evolve/agents/`. Job
`engaging/cd_ladder.sbatch` (12k context, 20:36–00:28) and a 16k rerun of the three rows that
overflowed (`cd_ladder16k`, 01:xx–03:20). Errored episodes (context overflow) count as failures.

| harness (what is appended after each answer) | circuit-wise accuracy | joint truth table correct | err% | uniq@end (ours) | 95% CI | eff (ours) | zero% | oracle% | rep% | tok/ep |
|---|---|---|---|---|---|---|---|---|---|---|
| bare — nothing | 12.9 | 0.5 | 0.0 | 23.2 | [19.8,26.8] | 0.553 | 36.7 | 25.4 | 3.2 | 2389 |
| facts (a) — the model's own observations restated | 16.2 | 0.5 | 0.3 | 29.0 | [25.3,32.5] | 0.570 | 34.8 | 26.4 | 3.4 | 2800 |
| symbolic — count of remaining configurations + determined outputs | 20.5 | 2.2 | 1.3 | 41.2 | [37.3,45.0] | 0.612 | 30.1 | 27.9 | 3.9 | 2648 |
| enum15 — symbolic + full list when ≤ 15 remain (the hand-written harness of 10-06; 16k) | 23.7 | 5.8 | 0.2 | 41.8 | [38.0,45.8] | 0.612 | 30.4 | 28.7 | 4.0 | 2892 |
| T3-gated — enum15 only when a trap proxy fires | 31.3 | 10.2 | 1.3 | 36.8 | [32.8,40.7] | 0.612 | 30.4 | 26.7 | 5.2 | 3104 |
| splits (c) — enum15 + 0/1 split counts per open entry (16k; still 10% overflow) | 22.3 | 5.7 | 10.2 | 53.7 | [49.8,57.8] | 0.702 | 20.8 | 32.2 | 6.7 | 4316 |
| oracle hint (d) — enum15 + the max-entropy query (16k) | 27.3 | 6.8 | 3.0 | 96.5 | [95.0,97.8] | 0.951 | 3.4 | 91.1 | 0.2 | 4522 |

Files: `scores_12k.json` (all seven at 12k), `scores_16k.json` (enum15/splits/oracle_hint at
16k), `scores_merged.json` + `ladder.png` (valid rows). Raw traces: release asset
`cd_ladder_traces.tar.zst` (`results/cd_ladder/`, `results/cd_ladder16k/`).

**Takeaways**
* State (symbolic/enum15/T3-gated) is worth +14–18 points and the three variants are within
  each other's CIs; the enumeration adds nothing to information gathering, only to `joint truth table correct`.
* Decision tables (+30) and the decision itself (+73, eff 0.95) are used almost verbatim, but
  `oracle hint` still gets the joint truth table right in only 6.8%: it copies the query, not the answer.
* Restating observations (a) does almost nothing: the failure is not forgetting answers.
* The proposer-written harnesses in 04/05 beat every ladder level with fewer tokens.
