# Third-party code used by this project (not vendored)

Clone each next to this README; scripts expect these directory names.

| Directory | Repository | Commit / version | Used for |
|---|---|---|---|
| `AR-Bench/` | https://github.com/TongLi3701/AR-Bench (official AR-Bench evaluator, prompts, data) | as of 2026-10 | GN / SP / DC tasks, judge prompts |
| `meta_harness_official/` | https://github.com/stanford-iris-lab/meta-harness | 8123cca | the evolution-loop protocol we mirror; `claude_wrapper.py` copied verbatim (MIT) into `meta/cd_evolve/` |
| `meta_harness_repo/` | https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact | 57fefdb | reference only |
| `rrsi_official/` | https://github.com/google-research/rrsi | be50316 | selection-side rules (noise band, cost rule, critic) re-implemented in `meta/cd_evolve/evolve.py` |
| `evoharness_rl_official/` | https://github.com/EvoHarnessRL/EvoHarness-RL | 1c1e63a | baseline; our CircuitDecoding adapter lives in `baselines/evoharness_cd/` and is copied into `evoharness/envs/circuitdecoding/` of that checkout |
| `tthe_official/` | https://github.com/junnie00/TTHE | 69a614d | test-time harness evolution protocol (batches, G branches x R rounds, three proposer roles, agentic judge with rollback) ported to CircuitDecoding in `meta/cd_tthe/` |

Data: `mt_puzzles/` holds the CircuitDecoding split of Multi-Turn Puzzles
(https://huggingface.co/datasets/arianhosseini/mt_puzzles, CC-BY-4.0). TurtleBench
(https://github.com/mazzzystar/TurtleBench) is cloned on the cluster for judge calibration.
