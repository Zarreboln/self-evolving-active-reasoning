# Self-Evolving Harnesses for Active Reasoning

MAS.S62 *Self-Evolving AI* (Fall 2026) course project.
**Zining Liu** (ziningl@mit.edu) — Massachusetts Institute of Technology

## The question

An active-reasoning agent must *ask* for the information a problem withholds, one question at
a time, and the dominant failure of small models is not a wrong final answer but a stalled
dialogue: questions that stop narrowing what the answer could be. Here the model's weights are
fixed and everything around it — what it is shown, what it is asked again, what is refused —
is a **harness** that a stronger coding agent evolves against measured traces
(the Meta-Harness protocol, arXiv 2603.28052, with RRSI-style regularized selection,
arXiv 2609.24972). The questioner is a frozen Qwen2.5-7B-Instruct; the proposer is Claude.

Testbed: **CircuitDecoding** from Multi-Turn Puzzles (arXiv 2508.10142). Three hidden boolean
circuits built from an announced gate inventory, 18 one-bit queries, then the joint truth table.
The inventory implies an exact, enumerable hypothesis space (252–3924 joint tables), so the
feasible set after every answer, the expected information of every possible query and a
max-entropy oracle are all computable without the truth — the harness can be given *any* state,
and what the model does with it can be measured per turn.

## Results so far (300 instances unless noted; held-out numbers are frozen one-shot evaluations)

| system | who wrote it | uniq@end | exact | eff |
|---|---|---|---|---|
| bare (no harness) | — | 23–25 | 0.3 | 0.55 |
| EvoHarness-RL ReAct / EvoHarness-Base (inline) | official framework | 26–29 / 30–32 | 0 / 1 | 0.57 |
| enum15 (state summary every turn) | hand-written | 38–42 | 6 | 0.61 |
| EvoHarness-RL always-on panels | official framework | 71–75 | 62–65 | 0.76 |
| oracle hint (the max-entropy query every turn) | hand-written | 96.5 | 6.8 | 0.95 |
| answer_grid (state_only regime, held-out 200×3) | proposer | 90.0 | 84.7 | 0.76 |
| **veto_only** (state_only: refuse queries whose answer is already determined) | proposer | **96.7** | **86.2** | 0.89 |
| copy_query (open regime: one copyable query, held-out 250×3) | proposer | 99.7 | 80.4 | 0.96 |

`uniq@end` = the 18 queries pinned the answer down (information gathering, luck-free);
`exact` = the task's success metric; `eff` = ex-ante bits of the chosen query over the best
available query (1.0 = oracle, 0.63 ≈ a random unasked entry). Details, commands and CIs in
[`experiments/`](experiments/).

What the numbers say: the 7B's failure on this task is belief tracking, not planning — it keeps
asking outputs its own answers already determine. A 30-line harness that only refuses such
queries, computing no information quantity, lifts it from 25% to 97% at the same token cost; a
harness that hands it the decision reaches the ceiling but leaves the final answer to copying.
Pull-style harnesses (the model must call `track`) go unused by a frozen 7B.

## Layout

| path | content |
|---|---|
| `experiments/` | one folder per experiment: setup, command, results, takeaway, score files |
| `stage0/` | hypothesis-space oracle (`cd_oracle.py`), AR-Bench counterparts, family survey |
| `runner/` | episode runners (`cd_harness_run.py`, `sp_runner_async.py`), scorers, mock task models, judge calibration |
| `meta/cd_evolve/` | the evolution loop (`evolve.py`), proposer skill prompt, `agents/` (seed + proposer-written harnesses), run logs and proposer transcripts |
| `baselines/evoharness_cd/` | our CircuitDecoding adapter for EvoHarness-RL |
| `engaging/` | Slurm scripts (vLLM servers, ladder, judge, downloads) for the MIT Engaging cluster |
| `envs/`, `signal/`, `policy/` | pointers into the above, by role |
| `experiments.md`, `RELATED_WORK.md`, `meta/journal.md` | dated experiment log, literature notes, process journal |
| `THIRD_PARTY.md` | pinned external repositories (AR-Bench, Meta-Harness, RRSI, EvoHarness-RL, MastermindEval) |

Raw per-episode traces (~330 MB) are attached to GitHub Releases, one tarball per experiment.

## Reproduce (sketch)

```bash
# serve the task model (any OpenAI-compatible endpoint), then
python runner/cd_harness_run.py --harness meta/cd_evolve/agents/veto_only.py \
    --split heldout --split-file splits/cd_split_v2.json --seeds 3 --base-url http://host:port/v1 \
    --out results/cd_heldout/so1/veto_only.jsonl
python runner/cd_score.py results/cd_heldout/so1/*.jsonl --split heldout --split-file splits/cd_split_v2.json
# one evolution iteration (needs the claude CLI for the proposer and a served 7B on the cluster)
cd meta/cd_evolve && python evolve.py --run-name demo --regime state_only --iterations 1 \
    --split-file ../../splits/cd_split_v2.json
```

The Situation Puzzles track (AR-Bench) uses Llama-3.1-405B-Instruct AWQ-INT4 as respondent and
judge (82.0% on TurtleBench-150, equal to the official bf16 judge's 82.7%); see
`experiments/02_judge_calibration/` and `experiments/07_sp_bare_405judge/`.
