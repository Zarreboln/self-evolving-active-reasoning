# 05 — Evolution run `so1`: regime *state_only* (2026-10-10 21:22–2026-10-11 ~01:00)

Same loop as 04 with the upgraded runner (hooks `filter_action` / `compose_messages`), the
**v2 split** (search 100 × 2 seeds, held-out 200 × 3 seeds), **RRSI-style selection** (noise
band δ = 0.080 from the bare bootstrap; cost rule ΔC ≤ 0.1 + 2.0·ΔS; within-band only with ≥5%
token saving) and the **critic** (regex precheck + `claude -p` review; verdicts in `critic/`).
**Regime state_only**: the harness may present/reorganize state and history and veto repeats or
determined entries, but must not compute information gain, split counts or recommend queries.
16k context on the second (preemptable) server.

Proposer session: 1346 s, subscription-priced $6.69. All three candidates passed the critic.

**Search split (100 × 2)** — `search/*.json`

| system | circuit-wise accuracy | joint truth table correct | uniq@end (ours) | eff (ours) | rep% | ill% | tok/ep | selection |
|---|---|---|---|---|---|---|---|---|
| bare (seed) | 12.6 | 0.0 | 19.5 | 0.550 | 2.2 | 0.4 | 2200 | — |
| enum15 (seed) | 25.4 | 6.0 | 38.0 | 0.604 | 4.4 | 0.3 | 2904 | — |
| veto_determined — enum15 + re-ask when the query is determined/repeated | 82.6 | 76.5 | 93.5 | 0.880 | 0 | 1.0 | 4437 | admissible (+0.555, cost +27%) |
| **veto_only** — nothing per turn; re-ask only when about to waste a query, with the list of open outputs; remaining tables before the final guess | **89.1** | **83.5** | **97.5** | 0.899 | 0 | 0.1 | 2882 | frontier (within band, −49% tokens) |
| answer_grid — enum15 re-laid as a per-circuit grid | 90.3 | 82.5 | 86.5 | 0.746 | 1.4 | 2.3 | 3966 | below the noise floor |

**Held-out (200 × 3 = 600 episodes per system)** — `heldout_scores.json`, `heldout_table.txt`

| system | circuit-wise accuracy | joint truth table correct | err% | uniq@end (ours) | 95% CI | eff (ours) | tok/ep | vs enum15 (paired): uniq@end / joint truth table correct |
|---|---|---|---|---|---|---|---|---|
| bare | 13.2 | 0.3 | 0.0 | 25.2 | [21.7,28.7] | 0.556 | 2347 | −12.5 / −6.2 |
| enum15 | 24.4 | 6.5 | 0.5 | 37.7 | [34.0,41.7] | 0.614 | 3046 | — |
| answer_grid | 91.2 | 84.7 | 0.0 | 90.0 | [87.3,92.3] | 0.763 | 4071 | +52.3 / +78.2 |
| veto_determined | 82.4 | 75.8 | 1.5 | 95.3 | [93.7,96.8] | 0.880 | 4448 | +57.7 / +69.3 |
| **veto_only** | **91.3** | **86.2** | 0.0 | **96.7** | [95.2,98.0] | 0.890 | 2699 | **+59.0 [+55.2,+63.0] / +79.7 [+76.3,+82.8]** |

Candidate code: `meta/cd_evolve/agents/{veto_only,veto_determined,answer_grid}.py`.
Raw traces: release asset `cd_evolve_so1_traces.tar.zst`.

**Takeaways**: without computing any information quantity or recommending any query — only
refusing queries whose answer every remaining configuration already agrees on — the 7B goes
from 25% to 97% uniq@end and from 0% to 86% joint truth table correct at bare's token cost. On this task the model's
failure is belief tracking (asking what it already knows), not planning. answer_grid shows the
presentation axis alone is worth +52 points (it was rejected on the search split by the noise
floor and still scored 90% held-out: the cost of conservative selection, not a contradiction).
