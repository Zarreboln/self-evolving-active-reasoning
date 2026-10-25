# Third-party code, pinned as submodules

Every repository this project runs against or mirrors is a submodule at an exact commit, so a
reproduction gets the same code we ran and not whatever upstream has moved to:

```bash
git clone --recurse-submodules https://github.com/Zarreboln/self-evolving-active-reasoning.git
# or, in an existing clone
git submodule update --init
```

They are submodules rather than copies for two reasons. A pinned commit is what reproduction needs,
and it is lost the moment a tree is copied in. And `git -C <dir> status` then answers "did we modify
their code?" directly — which is how the two facts in the last column below were established.

| Directory | Repository | Pinned commit | Licence | Our changes to it |
|---|---|---|---|---|
| `AR-Bench/` | tmlr-group/AR-Bench | `9971322` | **none at top level** | **none** — clean tree |
| `balar_official/` | AymenEcharghaoui/BALAR | `499d9a9` | MIT | none — clean tree |
| `tthe_official/` | junnie00/TTHE | `69a614d` | MIT | none — clean tree |
| `meta_harness_official/` | stanford-iris-lab/meta-harness | `8123cca` | MIT | none — clean tree |
| `meta_harness_repo/` | stanford-iris-lab/meta-harness-tbench2-artifact | `57fefdb` | none | none — clean tree |
| `rrsi_official/` | google-research/rrsi | `be50316` | present (Google) | none — clean tree |
| `evoharness_rl_official/` | EvoHarnessRL/EvoHarness-RL | `1c1e63a` | MIT | **4 paths**: `evoharness/envs/__init__.py` edited to register the env, plus `configs/eval/circuitdecoding.yaml`, `evoharness/envs/circuitdecoding/` and `scripts/cd_smoke.py` added — the same files kept in `baselines/evoharness_cd/` |

AR-Bench ships no licence file, so its tree is not redistributed here; the submodule fetches it from
upstream instead. That is also why nothing in this repository patches it: the Jev referee shim
(`runner/jev_referee_shim.py`) speaks the OpenAI API so the official evaluators run unmodified, which
`git -C AR-Bench status` can confirm at any time.

## What we re-implemented rather than imported, and why

| Ours | Mirrors | Why not their code directly |
|---|---|---|
| `meta/cd_tthe/` | `tthe_official/` | TTHE ships one package per domain (`ase`, `swe`, `livecodebench`, `ds1000`), each a copy of the same files; ours is that pattern for Circuit Decoding and Situation Puzzles, which do not exist upstream |
| `meta/cd_evolve/` | `meta_harness_official/` + `rrsi_official/` | the Meta-Harness loop and RRSI's selection-side rules, for our tasks; `claude_wrapper.py` is copied verbatim (MIT) with its provenance in its first line |
| `baselines/evoharness_cd/` | `evoharness_rl_official/` | an env written to their plugin interface; it has to live inside their tree to run, hence the four changed paths above |
| `runner/md_evaluator.py` | `balar_official/` (`icraft-md`) | runs AR-Bench's prompt baselines on BALAR's dataset. It is **not** BALAR: BALAR's method is `run.py` + `agent.py` and reproducing *that* means running their code, not ours |

## Data

`balar_official/data/arbench/{sp,dc}/test.json` is **byte-identical** to `AR-Bench/data/{sp,dc}/test.json`
(md5 `8416d136…` and `0b476dba…`, 100 instances each), so the two repositories agree on the test sets.
BALAR additionally ships the `train.json` splits, which we do not use, and `data/mediQ/all_craft_md.jsonl`
(140 instances), which is the only iCraft-MD source — MediQ proper is not included, so the MediQ Expert
baseline would have to come from `stellalisy/mediQ`. BALAR has no Guess Number prompts: it does not
cover that task.

`mt_puzzles/` holds the CircuitDecoding split of Multi-Turn Puzzles
(https://huggingface.co/datasets/arianhosseini/mt_puzzles, CC-BY-4.0) and is committed here, not a
submodule. TurtleBench (https://github.com/mazzzystar/TurtleBench) is cloned on the cluster for judge
calibration.
