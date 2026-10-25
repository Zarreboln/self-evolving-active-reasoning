# Domain spec — Meta-Harness on AR-Bench Situation Puzzles

Produced by following `meta_harness_official/ONBOARDING.md`. It is an implementation plan, not an
implementation, and **it authorizes no spending or job launches**.

Decisions are marked **[confirmed]** when the user or a measurement settled them, **[default]** when
proposed here, and **[unknown]** with a next step when unresolved.

## 1. Why this document exists

The harness-evolution arm is being rebuilt to run **Meta-Harness's own loop**
(`meta_harness_official/reference_examples/text_classification/meta_harness.py`, pinned at `8123cca`)
instead of our re-implementation in `meta/cd_tthe/`. **[confirmed]** Two reasons were given: a
re-implementation can diverge silently from the method it claims to run, and the seed should be a
real baseline rather than our own `bare` loop.

Their loop turns out to be almost domain-agnostic. It imports only `claude_wrapper` and two helpers
from `benchmark`, evaluates by shelling out to `benchmark.py`, and treats candidates as files in
`agents/` that it import-checks. Exactly two lines are domain-specific:

| Line | Content | Our value |
|---|---|---|
| `meta_harness.py:203` | `from text_classification.agents.{name} import *` | `from arbench_mh.agents.{name} import *` |
| `meta_harness.py:29` | `BASELINE_FILES = {"__init__.py", "no_memory.py", …}` | our seed file names |

Everything else — iteration accounting, `--resume`, candidate validation, the Pareto frontier, the
proposer session, `finalize_run` — is used verbatim. That two-line diff is recorded in
`meta/arbench_mh/LOOP_DIFF.md` so the claim "we ran their loop" stays checkable.

## 2. Task and success criterion

AR-Bench Situation Puzzles. A questioner is shown the puzzle's **surface** and asks up to 25
yes/no questions to a referee that also knows the hidden **bottom**; the referee answers only
`Yes` / `No` / `Unknown`. After the last question the questioner writes the explanation.

Success is AR-Bench's official metric: **character-level F1** of the written explanation against the
hidden bottom. **[confirmed]** — the user has ruled that official metrics are not to be replaced
(`feedback_benchmark_metrics_official_only`).

One caveat must travel with every number this produces: experiment 11 measured that this F1
correlates 0.806 with answer *length*, and a smoke-test prediction naming the wrong cause of death
still scored 0.637. The metric is kept because it is the official one; `uniq@end`, key-question
recall and question-repetition are reported beside it as **(ours)** diagnostics, never in the same
column.

Example (abbreviated, `AR-Bench/data/sp/test.json`): surface — a man is found dead in a locked
study, no sign of struggle; bottom — a concealed heart condition and a delayed ambulance. A good
questioner spends its budget separating "someone else was present" from "nobody was", not
rewording one hypothesis.

## 3. What the harness may change, and what is fixed

**Fixed.** Nothing in this list may be touched by a candidate, and `benchmark.py` enforces it:

| Fixed | Value |
|---|---|
| base model | `Qwen/Qwen2.5-7B-Instruct`, frozen, temperature 0.7, top_p 0.7 **[confirmed]** |
| serving | one vLLM instance on `bath` (`--max-model-len 8192`), shared by all candidates |
| referee | **Jev** (`jev-latest`) through `runner/jev_referee_shim.py` **[confirmed]** |
| question budget | 25 **[confirmed]** |
| scorer | AR-Bench's `f1_score` on characters, called from AR-Bench's own module |
| test split | `AR-Bench/data/sp/test.json`, 100 instances, untouched until §6 |

The referee is fixed to Jev because that is what the baselines now use. The previous runs
(experiment 10) used Llama-3.1-405B, which is no longer reachable — Engaging is closed to us and the
old H100 is retired — so our row and the baseline rows could not go in one table. The referee is the
environment: it decides what the questioner learns and it scores key-question recall, so a different
referee is a different task.

**Editable.** Everything between the fixed pieces: what the questioner is told each turn, what state
the harness keeps across turns, how the referee's replies are grouped and re-presented, any filter
applied to a proposed question before it is sent, how the final explanation is composed from what
was confirmed.

**Interface.** A candidate is a module in `agents/` defining one class that subclasses
`SPHarness` (`meta/cd_tthe/sp_harness_base.py`) and overrides `solve()`:

```python
class Harness(SPHarness):
    def solve(self) -> str:          # returns the final explanation
        env = self.env               # env.system_prompt, env.turn_prompt(t), env.final_prompt
        ...                          # env.ask(question) -> "Yes" | "No" | "Unknown"
        ...                          # self.llm(messages) -> the frozen model
```

Reusing the existing interface **[default]** means every harness already in
`meta/cd_tthe/agents_sp/` loads under the new loop, so the old runs can be re-scored under Jev
without being rewritten.

## 4. The seed

`agents/baseline_few_shot_inst.py` — AR-Bench's `few_shot_inst` method, the strongest of its
prompt baselines, wrapped in the interface above with its official prompt text unchanged.
**[confirmed]** The search therefore starts from something that already appears in a published
table and has to beat it, rather than from a `bare` loop that anything beats.

`agents/baseline_bare.py` (AR-Bench's zero-shot loop) is kept as a second baseline so the table has
a floor, but it is not the seed.

**[unknown]** Which number the seed scores on the search split. Next step: evaluate both baselines on
the search split before the first proposer call; that measurement is also the acceptance check in §8.

## 5. Splits, and what the proposer may see

AR-Bench ships train splits that the previous runs did not use: `sp/train.json` holds **400**
puzzles against the test set's 100. (BALAR's copies of both are byte-identical to AR-Bench's.)

| Split | Source | Size | Who may see it |
|---|---|---|---|
| search | `sp/train.json` | 50 **[default]** | proposer: traces and, in the labeled arm, scores |
| selection | `sp/train.json`, disjoint | 50 **[default]** | the loop's Pareto frontier only; never the proposer |
| test | `sp/test.json` | 100 | nobody until one candidate is selected |

This is the fix for a real defect in experiment 10, whose stream ran over 30 of the **test** puzzles
and then evaluated on all 100 — it reported the 70 unseen separately, which was honest, but the
search had still touched the test set. With 400 train puzzles there is no reason to.

Size 50 follows Meta-Harness's Appendix D (50–100 examples). Their tip — "filter for examples that
the baseline gets wrong" — is available but **[default] off for the first run**: filtering changes
what the search optimizes, and we want one clean run before adding that variable.

**Two arms, because Meta-Harness and our claim disagree about labels.** Meta-Harness gives the
proposer candidate scores. Our claim is label-free (`project_ar_claim_scope_data_free`): the correct
word is label-free / transductive, and the evidence for it is that strict ≈ labeled. So both arms
run **[confirmed by the existing claim structure]**:

- `labeled` — Meta-Harness as designed: the proposer sees F1 on the search split.
- `strict` — the proposer sees run status and the referee's replies only. No F1, no bottom, no gold.

The strict arm keeps the label-isolation machinery already built for `meta/cd_tthe`: a filesystem
boundary plus a session audit that discards any candidate whose proposer session read outside its
folder (`meta/cd_tthe/audit.py`). The difference between the two arms is the measurement that
licenses the label-free claim; neither alone does.

## 6. Selection and final testing

The loop picks on the **selection** split's Pareto frontier over their two axes, unchanged in
meaning: `accuracy` and `memory_context_chars`. For this domain `accuracy` is the official character
F1, and `memory_context_chars` is the mean size of the context the harness feeds the frozen model per
episode — the quantity their text-classification domain measures as the memory it carries. A harness
that wins by pasting more into every turn is therefore visible on the frontier rather than hidden. One candidate is named before any test number is computed. Then, once:
all 100 test puzzles, no proposer, no judge, the frozen evaluator, under Jev.

Baselines are evaluated on the same splits, the same referee and the same budget, or the table has
no denominator. That is the whole reason the baseline queue exists.

## 7. Budget — an estimate, not an authorization

Per candidate, on a 50-puzzle split: 50 × 25 = 1250 frozen-model calls and 1250 Jev calls.
Sharded 16 ways that is a few minutes of one H100; Jev bills input only at $0.042/MTok and ~600
tokens a call, so about **$0.03** per candidate evaluation.

At their defaults — k = 2–3 candidates per iteration, 20–40 iterations:

| | estimate |
|---|---|
| candidate evaluations | 60–120 |
| Jev | $2–4 |
| GPU | 5–10 h on one card |
| proposer sessions | 20–40 Opus 5.5 sessions; experiment 10 measured $11.92 for 12, so **$20–40** subscription-priced |

Stopping **[default]**: 30 iterations, or no Pareto improvement on the selection split for 8
consecutive iterations, whichever comes first.

## 8. Smallest first implementation, and its acceptance check

Build only what the loop requires, SP only, one arm (`labeled`), one iteration:

1. `meta/arbench_mh/benchmark.py` — same CLI as theirs
   (`--logs-dir`, `--results-dir`, plus `--systems`, `--datasets sp`), writing the result and
   frontier files in the shapes their `load_results` and `print_frontier` read.
2. `meta/arbench_mh/agents/` — `baseline_few_shot_inst.py`, `baseline_bare.py`, `__init__.py`.
3. `meta/arbench_mh/.claude/skills/meta-harness/SKILL.md` — their skill, with the domain swapped.
4. `meta/arbench_mh/config.yaml` — one dataset (`sp`), one model, the two baselines.
5. `meta_harness.py` and `claude_wrapper.py` copied in with the two-line diff of §1 recorded.

**Acceptance check.** `python meta_harness.py --iterations 1` finishes, and:

- both baselines produce a score on the search split, and `baseline_few_shot_inst` reproduces the
  number the AR-Bench baseline queue produced for `few_shot_inst` within sampling noise;
- the proposer writes `pending_eval.json` and at least one candidate passes the import check;
- that candidate is evaluated and appears in `frontier_val.json` under `_pareto`;
- no test file was opened — checked by the session audit, not by inspection.

## 9. Open items

| Item | Next step | What depends on it |
|---|---|---|
| ~~result shapes~~ **[resolved]** | `<results-dir>/<dataset>/<memory>/<model>[_seed<N>]/val.json`, exactly three path components under the results dir. Each file is a dict with a numeric `accuracy` (checked by `_has_usable_result`), `seed`, and the fields `load_results` sums: `runtime_seconds`, `llm_calls`, `llm_input_tokens`, `llm_output_tokens`, `llm_total_tokens`. A group is dropped unless **every** seed in `SEEDS` is present, so one seed must mean `seeds: [42]` in config. | done |
| ~~where the loop runs~~ **[resolved]** | **Locally.** `bath` has neither `uv` nor the `claude` CLI, so the proposer cannot run there; the Mac has `uv` 0.7.16. The loop and proposer run locally and the frozen model and referee are reached over the network — the same split experiments 09/10 used. No loop edit needed. | done |
| DC **[unknown]** | same spec with `dc/train.json` (401 instances), referee replaced by the Kimi suspect role | the second row of the table |
| re-scoring the old harnesses under Jev | `evaluate.py --task sp` with the answerer pointed at the shim; ~$0.4 | whether experiment 10's numbers can be quoted at all |
