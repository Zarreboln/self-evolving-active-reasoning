# Third-Party Code

## Submodules (pinned to exact commits)

```bash
git clone --recurse-submodules https://github.com/Zarreboln/self-evolving-active-reasoning.git
```

| Directory | Repository | License | Our changes |
|---|---|---|---|
| `AR-Bench/` | tmlr-group/AR-Bench | no license file at the top level, so the code is not copied here, only fetched as a submodule | none |
| `meta_harness_official/` | stanford-iris-lab/meta-harness | MIT | none |
| `third_party_b/ace/` | ace-agent/ace @ 82709de (its `ace-appworld` submodule @ 9f3e921) | Apache-2.0 | none; loaded at run time by `multibench/mb/baselines/ace/official.py` |

## Code copied into this repository

| Location | Source | License |
|---|---|---|
| `multibench/mb/vendor/balrog_babyai/` | the text wrapper for BabyAI from balrog-ai/BALROG | MIT (license file in the folder) |

## Used at run time but not in this repository

Running the official baselines also uses code from these projects (installed on the server or downloaded
at run time): TextArena, BALROG, ALFWorld, ScienceWorld, ysymyth/ReAct (prompt files), and the
`science_world` branch of SwiftSage (prompt files and `eval_utils.py`). How each is used and what was
changed is described at the top of each script in `multibench/mb/official/` and in
`notes/official_initial_agents.md`.

The other submodules used by the earlier MAS.S62 active-reasoning project (BALAR, TTHE, RRSI,
EvoHarness-RL, meta-harness-tbench2-artifact) are kept under the tag `mas-s62-ar-report`.
| `third_party_b/harness2` | google-research/harness2 @ 6868817 (submodule) | Apache-2.0 | Harness² baseline: official prompts and library code, imported by `multibench/mb/baselines/harness2/` |
