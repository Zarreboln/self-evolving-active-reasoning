# GN: acquired evidence versus the final answer

A post-hoc reconstruction of the existing GN run found that recorded feedback uniquely identified the answer in **164 of 300 traditional-baseline trajectories**, yet **162 of those trajectories ended with an incorrect final answer**. The three methods share the **same 100 test cases**; these are not 300 independent puzzles.

This identifies a concrete failure to turn sufficient recorded evidence into a correct terminal response under the tested protocol. It does not establish what the model internally knew, a cause such as forgetting, or the benefit of a repair.

| Method | Trajectories | Native final correct | History uniquely identifies answer by turn 25 | Unique history, wrong final | Unique history without earlier correct guess | Final contradicts feedback |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 100 | 0 | 83 | 83 | 82 | 100 |
| Few-shot | 100 | 0 | 52 | 52 | 50 | 100 |
| Few-shot + instruction | 100 | 2 | 29 | 27 | 26 | 98 |
| Total trajectories | 300 | 2 | 164 | 162 | 158 | 298 |

The other **136 histories remained ambiguous**. An external deterministic reconstruction applied the native feedback rules to the 5,040 permitted candidates, preserving leading zeros. It used the recorded guesses and public feedback, not hidden answers, to construct feasible sets. No empty feasible sets or singleton/correctness identity mismatches were reported.

## What happened in the four earlier-hit failures?

Four trajectories explicitly guessed correctly during questioning but gave a wrong final answer. In all four, the correct response and unambiguous success feedback were still present in the final input. The final response ended normally and contained one distinct wrong number; the parser agreed with the recorded prediction. Three reused a previously refuted guess. These four failures are therefore not explained by response truncation or the parser dropping a correct final number. This observation does not determine the internal cognitive cause.

The result is broader than retaining an explicit hit: **158 of the 164 uniquely identifying histories never contained an earlier correct guess**. The available constraints could identify the answer even though the model had never proposed it.

## Repetition and the fixed protocol

Across the 7,500 question turns, **1,318 repeated a previously refuted guess**. There were **1,348 repeated guesses in total**, including 30 repeats of correct guesses. These quantities must not be interchanged.

The original traditional-baseline loop continues for 25 answered questions and then requests a separate terminal prediction. It does not stop after success. This delta does not separate repeats or zero-reduction questions before versus after the history became unique. It therefore does not label all such turns as poor information acquisition. A question outside the current feasible set can also be a legitimate informative probe.

## Original results and limits

Original native success counts remain **0, 0, 2, 5, 2 out of 100** for zero-shot, few-shot, few-shot + instruction, ToT, and proactive CoT, respectively. This diagnostic covers only the three traditional methods. It does not rescore the five-method experiment. ToT's five successes have no comparable actor-final prediction, and proactive CoT's actor-final prediction is unavailable; selected-path question counts omit search-branch costs.

The tested model was **Qwen2.5-7B-Instruct**, replacing the official script's Qwen2.5-32B-Instruct. The underlying run also discloses explicit output/context caps, concurrent serving and the ToT parser adaptation. Findings apply to that tested setup; they are not an exact reproduction of the original 32B system.

**164/300 is identifiability coverage, not corrected model accuracy.** No test trajectory was used for training or tuning. This analysis did not improve a model, policy or output. It establishes neither self-evolution gains nor improved questioning efficiency. A prospective intervention using separate training/development data and a matched, untouched evaluation would be needed to test whether preserving and using evidence improves outcomes.

## Files and provenance

- `aggregate-results.json`: anonymous counts, definitions, limits and source checksums.
- `by-method.csv`: the three method-level aggregates, including repeat distinctions.
- `manifest.json`: file checksums and provenance for this publication delta.

Preparation only read existing summary reports and checked their arithmetic and consistency. It did not rerun trace diagnostics, generate model responses or submit a GPU job. The earlier execution audit supports recorded-run integrity. Research CP72 reviewed the reported interpretation; it was not an independent raw-data audit.

No case text, secret values, guessed numbers, per-case identifiers, raw responses, credentials, teammate messages or recording scripts are included.
