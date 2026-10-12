# 06 — EvoHarness-RL as a baseline on CircuitDecoding (2026-10-11 06:28–07:36)

**What**: the official EvoHarness-RL framework (arXiv 2608.05446) run on our task through the
adapter in `baselines/evoharness_cd/`. Their Belief is backed by our exact feasible-set tracker;
their three coordination modes are compared on the same 7B, 300 instances, 2 runs each
(temperature 0.7, 1200 tokens, meta-actions free up to 18 per episode, accumulated context).
Their own summaries: `summary_*.json`; our metrics replayed from their trajectories:
`replay_scores.txt` (`runner/cd_score_evoharness.py`).

| mode | run | uniq@end | 95% CI | exact | ckt | eff | zero% | oracle% | rep% | meta/ep |
|---|---|---|---|---|---|---|---|---|---|---|
| env_only (ReAct, no harness) | s0 / s1 | 29.3 / 26.0 | [24,34] / [21,31] | 0.0 / 0.0 | 4.3 / 4.9 | 0.577 / 0.569 | 33.5 / 33.9 | 20.2 / 19.3 | 4.9 / 4.1 | 0 |
| always_on (STATE panel every turn: count, determined, open outputs, tables ≤15, plan) | s0 / s1 | **71.0 / 74.7** | [66,76] / [70,79] | **62.0 / 65.0** | 74.8 / 79.0 | 0.760 / 0.763 | 14.7 / 13.8 | 38.4 / 38.1 | 0.6 / 0.4 | 0 |
| inline = EvoHarness-Base (model calls track / commit / recall / note itself) | s0 / s1 | 30.0 / 31.7 | [25,35] / [27,37] | 0.7 / 1.7 | 5.7 / 6.9 | 0.577 / 0.586 | 33.3 / 32.7 | 21.7 / 20.8 | 3.8 / 4.6 | track 0.07 / 0.09 |

Raw trajectories: release asset `evoharness_cd_trajectories.tar.zst`.

**Takeaways**
* ReAct ≈ our bare (23–25%).
* EvoHarness-Base (inline) ≈ ReAct: the frozen 7B calls `track` 0.08 times per episode, so the
  harness is effectively unused — consistent with their paper (frozen Qwen3-8B: 47.8 → 56.4 on
  ALFWorld; the gains come from SFT + GRPO on *when* to call). Pull-style access fails for a
  frozen small model; the trainable EvoHarness-RL changes weights and is outside our frozen-model
  setting.
* always_on (push) reaches 71–75% with the same information our `enum15` renders (38–42%):
  the STATE block, the explicit list of still-open outputs, the short `<think>`/`<action>`
  format and the re-rendered objective make the presentation worth a factor of two. Our
  proposer's state-only candidates (answer_grid 90%, veto_only 97%) sit above it.
