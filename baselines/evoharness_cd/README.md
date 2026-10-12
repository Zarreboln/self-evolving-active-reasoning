# EvoHarness-RL on CircuitDecoding (baseline adapter)

Our `Domain` adapter that plugs Multi-Turn Puzzles CircuitDecoding into the
EvoHarness-RL framework (arXiv 2608.05446, https://github.com/EvoHarnessRL/EvoHarness-RL).

* `env.py` — the environment with the paper's exact wording (`A(0, 0, 1) = 1` /
  `Please send your next query. N queries remaining.`), 18 queries + final guess.
* `belief.py` — their `Belief` interface backed by our exact feasible-set tracker
  (`stage0/cd_oracle.py`): `track [remaining]`, `track [A]`, and the always-on panel.
* `prompts.py`, `domain.py` — the three coordination modes (`env_only` = ReAct,
  `always_on` = panels every turn, `inline` = EvoHarness-Base with meta-actions),
  accumulated context (the paper's protocol).
* `circuitdecoding.yaml` — eval config; `cd_smoke.py` — scripted-policy smoke test.

Install: copy the `.py` files to `evoharness_rl_official/evoharness/envs/circuitdecoding/`,
register `"circuitdecoding": "evoharness.envs.circuitdecoding:CircuitDecodingDomain"` in
`evoharness/envs/__init__.py`, copy the yaml to `configs/eval/`, `pip install -e .`, then

    AR_HARNESS_ROOT=<this repo> python -m evoharness eval --config configs/eval/circuitdecoding.yaml \
        policy.model=Qwen/Qwen2.5-7B-Instruct policy.base_url=http://host:port/v1 harness.mode=always_on split=all

Score the trajectories with our metrics: `python runner/cd_score_evoharness.py <out_dir>`.
Results: `experiments/06_evoharness_cd_baseline/`.
