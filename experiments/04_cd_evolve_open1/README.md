# 04 — Evolution run `open1`: regime *open* (2026-10-10 20:30–23:10)

**Loop**: `meta/cd_evolve/evolve.py` (Meta-Harness protocol: one fresh `claude -p --model opus`
proposer session per iteration with the skill in `meta/cd_evolve/.claude/skills/meta-harness-cd/`,
3 candidates, validated, benchmarked on the search split, frontier on uniq@end).
**Regime open**: the harness may compute anything from the public state, including
recommending queries. **Split v1**: search 50 instances × 2 seeds, held-out 250 × 3 seeds,
scored once after freezing. Task model Qwen2.5-7B-Instruct, 12k context.

Proposer session: 866 s, 81.8k output tokens, 12.9M cache-read tokens (subscription-priced
$3.90). Transcript: `meta/cd_evolve/logs/open1/claude_sessions/`.

**Search split (50 × 2)** — `search/*.json`, `evolution_summary.jsonl`

| system | uniq@end | 95% CI | exact | eff | zero% | oracle% | rep% | ill% | tok/ep |
|---|---|---|---|---|---|---|---|---|---|
| bare (seed) | 26.0 | [18,35] | 0.0 | 0.558 | 36.9 | 27.3 | 1.8 | 0.3 | 2084 |
| enum15 (seed) | 53.0 | [43,63] | 5.0 | 0.629 | 29.1 | 30.6 | 3.5 | 0.2 | 2879 |
| copy_query — the whole message is one copyable `<query>` (max-entropy) | **100.0** | [100,100] | 81.0 | 0.962 | 2.8 | 94.5 | 0.0 | 8.3 | 972 |
| copy_open — copy_query + the list of still-open outputs | 99.0 | [97,100] | 74.0 | 0.962 | 2.8 | 94.5 | 0.0 | 8.6 | 1231 |
| open_entries — enum15's negative list turned into a positive list | 32.0 | [24,41] | 47.0 | 0.577 | 34.6 | 30.0 | 0.7 | 0.3 | 1726 |

**Held-out (250 × 3 = 750 episodes per system)** — `heldout_scores.json`, `heldout_table.txt`

| system | uniq@end | 95% CI | exact | eff | tok/ep | vs enum15 (paired): uniq / exact |
|---|---|---|---|---|---|---|
| bare | 22.5 | [19.6,25.6] | 0.3 | 0.550 | 2222 | −12.5 / −6.8 |
| enum15 | 35.1 | [31.7,38.3] | 7.1 | 0.609 | 2960 | — |
| copy_query | **99.7** | [99.3,100] | **80.4** | 0.962 | 1129 | +64.7 [+61.2,+68.1] / +73.3 |
| copy_open | 99.7 | [99.3,100] | 68.4 | 0.962 | 1181 | +64.7 / +61.3 |
| open_entries | 34.5 | [31.2,37.9] | 42.3 | 0.582 | 1672 | −0.5 [−4.8,+3.9] / +35.2 |

Candidate code: `meta/cd_evolve/agents/{copy_query,copy_open,open_entries}.py` (docstrings carry
the proposer's diagnosis and hypothesis). Raw traces: release asset `cd_evolve_open1_traces.tar.zst`.

**Takeaways**: in the open regime one iteration reaches the ceiling by turning the model into a
copier (oracle agreement 94.5%); the proposer measured from the seed traces that 29% of enum15's
queries ask already-determined outputs (76% once two configurations remain) and that a canonical
scan skipping determined entries would reach 96% — the diagnosis that run `so1` exploits.
enum15's search-split 53% vs held-out 35% shows the sampling noise of a 50-instance split
(hence split v2 in 05). Note: this run used the old scorer convention (uniq@end selection) and
12k context; 2 of 100 enum15 search episodes overflowed.
