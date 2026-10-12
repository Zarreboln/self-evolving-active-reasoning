# Policies: harnesses around the fixed task model

Hand-written seeds and proposer-written candidates live in `meta/cd_evolve/agents/`
(one file each, interface documented in `meta/cd_evolve/.claude/skills/meta-harness-cd/SKILL.md`):

| file | origin | what it adds per turn |
|---|---|---|
| `bare.py` | seed | nothing (the environment line only) |
| `facts.py` | seed, ladder (a) | the model's own observations restated |
| `symbolic.py` | seed | count of remaining configurations + determined outputs |
| `enum15.py` | seed (hand-written belief summary, 2026-10-06) | symbolic + full list when <= 15 remain |
| `t3_gate.py` | seed | enum15 only when a trap proxy fires |
| `splits.py` | seed, ladder (c) | enum15 + 0/1 split counts per undetermined entry |
| `oracle_hint.py` | seed, ladder (d) | enum15 + the max-entropy query |
| `copy_query.py`, `copy_open.py`, `open_entries.py` | proposer, run open1 | see `experiments/04_cd_evolve_open1/` |
| `veto_only.py`, `veto_determined.py`, `answer_grid.py` | proposer, run so1 (state_only) | see `experiments/05_cd_evolve_so1/` |

The evolution loop that produced the proposer-written files is `meta/cd_evolve/evolve.py`.
