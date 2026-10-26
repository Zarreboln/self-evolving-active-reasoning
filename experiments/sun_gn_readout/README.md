# GN: public evidence and final-answer readout

This contribution separates **collecting useful feedback** from **turning that feedback into a correct final answer**. It includes a small public-only constraint adapter, synthetic tests, paired-statistics reproduction, and the experiments that motivated it.

The repository already implements exact GN inference in [stage0/gn_oracle.py](../../stage0/gn_oracle.py), and [runner/gn_runner_async.py](../../runner/gn_runner_async.py) already supports belief summaries. **Exact solving and candidate-state prompting are not claimed as new methods.** The additions here are a strict public-record interface, repeated-digit guess handling, explicit abstention, a controlled final-readout comparison, and its measured limitations.

## Evidence so far

All model experiments use frozen Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.7 and top_p 0.7. Each episode has 25 public-feedback interactions; only the final readout differs within the later paired studies. Weights are not trained.

### 1. Context evolution: completed and audited

One 20-step adaptation trajectory produced 19 format-accepted curator outputs but only **one actual note change**, at step 2. Evaluation used 100 paired cases and four conditions:

| Condition | Mean Q | Exact answers /100 | Record uniquely determines answer | Final outputs truncated |
|---|---:|---:|---:|---:|
| B0: no note | 0.886680 | 1 | 92 | 65 |
| B1: official static example | 0.766758 | 0 | 33 | 69 |
| S: initial note M0 | 0.808280 | 0 | 71 | 31 |
| E: final note M20 | 0.852756 | 0 | 70 | 74 |

Q averages normalized log candidate-set reduction over the 25 interaction steps; it is **not accuracy or a measurement of internal model belief**. Primary E−S Q difference: +0.044476, paired percentile 95% CI [0.020238, 0.068738], 66 wins /34 losses. E did not outperform B0 descriptively, and no final-accuracy gain was observed. Invalid curator notes were rejected while preserving the previous bytes; early-format checks became diagnostic records. These development-time amendments are part of the executed version, not an unchanged original frozen protocol.

### 2. Limited-feedback readout: completed and audited

A separate 50-case train-derived internal holdout was fixed before requests, excluding prior designated labels and declared exposed four-digit strings. It is **not a new official-test replication**. A common B0 history was followed by four final variants:

| Variant | Final procedure | EM /50 | Record-consistent /50 | Truncated /50 |
|---|---|---:|---:|---:|
| R0 | Original final prompt, 128 tokens | 0 | 0 | 32 |
| R1 | Same prompt, 512 tokens | 0 | 0 | 5 |
| R2 | Shared strong-prompt draft + neutral self-check; 256+256 caps | 0 | 0 | 0 |
| R3 | Same draft + first public-feedback contradiction; same caps | 0 | 0 | 0 |

R2 and R3 share the exact draft and revision seed. The checker does not enumerate candidates or see labels. Raw second outputs are scored, with no best-of selection or repeated correction. The original parser returns the last four-digit span, including for truncated outputs; missing spans count as wrong. There is no favorable reparsing.

Primary R3−R2: 0 percentage points, 0 wins /0 losses /50 ties. The paired bootstrap is degenerate [0,0]; supplementary simultaneous exact-binomial bounds are [-8.391,+8.391] percentage points. This is not an equivalence claim. R1 reduces truncation without observed EM improvement; it does not show that all other budgets or formats would fail. All 50 drafts and both revisions emitted exactly 8 output tokens despite larger caps. R3 changed 41/50 drafts and satisfied its cited constraint in 7/50 cases (R2: 1/50), yet none satisfied the whole record.

The allocation completed in 295 GPU seconds with 1,550 requests (50 discarded development readouts +1,500 evaluation requests), 2,093,206 input and 29,630 output tokens. Experimental API cost was $0; GPU time is reported separately.

### 3. Symbolic replay: post hoc, not a model result

The public-only solver recovered 44/50 answers correctly, abstained on six, and emitted no wrong answer. Candidate counts were 44 singletons, five pairs and one triple. Local replay took approximately 0.765 CPU wall seconds, without a model call. This was computed **after observing the null readout result**; it motivates a new test and does not rescue R3−R2.

### 4. Prospective answer-supplied readout: pending

A further disjoint internal N50 has been fixed and one bounded allocation submitted. No new score is claimed here. Baseline and assisted finals share the same new 25-turn history, 128-token cap and seed; execution order alternates. The assisted final receives an exact candidate-state block before the unchanged final instruction. The direct unique-only solver is a separate baseline.

When a record has one candidate, the tool has computed the answer. Any gain must be called **answer-supplied/tool-assisted readout**, not learned reasoning, memory evolution or solver novelty. It also changes input length and presentation. The model's raw output is never replaced by the solver's answer. See [readout_settings.json](readout_settings.json) for fixed settings. The main comparison is all-case paired EM; candidate strata, parser interactions, truncation and costs are secondary descriptive analyses. The study reserves at most 1,350 requests /172,800 output tokens /10 million input tokens /45 GPU minutes, with no retry or reduced-N substitution.

## Use the adapter

The hidden number has four distinct ASCII digits; leading zero is allowed. A guess may repeat digits. Each public record has exactly these fields:

```json
[{"turn": 1, "guess": "0011", "same": 1, "different": 1}]
```

`resolve_public(records)` enumerates all 5,040 legal hypotheses, returns an answer only if exactly one survives, and otherwise reports ambiguity or inconsistency. It accepts no secret/label fields. `public_state(records, 64)` returns the complete sorted candidate list up to 64; larger sets get an explicit count-only representation, never a selected subset. Inconsistent records raise an error for model-state delivery. Dataset/exposure exclusions never prune the solver universe.

`readout_pair.branch_readout` is a portable extraction of the evaluated final-branch routine. It accepts public history, frozen prompt text and a caller-provided boundary that records requests and enforces budgets. It neither loads a model nor submits jobs. The private cluster deployment is not included, so this folder alone is not an end-to-end reproduction of the original allocations.

## Verify without a GPU

From this directory, using standard-library Python:

```sh
python -B -m unittest discover -v
python -B verify_public_results.py
```

Nine synthetic adapter/statistics tests pass. The allocation runner additionally passed a 50-case synthetic auditor reconstruction and rejection of corrupted state; a submission-directory naming mismatch was caught before GPU submission and fixed without changing scientific code. These code checks are not real model performance evidence.

[results.json](results.json) contains aggregates and anonymized paired outcome rows so the released EM statistics can be recomputed. Raw prompts, records, private filesystem paths, authorization documents, target labels and original case identifiers are excluded. This limits independent reconstruction of raw model outputs; hashes bind retained originals but do not replace access to them. Audit code independently reconstructed results, but was executed by the same assistant. Research reviewers reviewed reported evidence, not an independent raw-data replication.

This is one model, one decoding setup and outcome-informed sequential follow-up. Restricting the internal holdout by prior exposures changes its eligible distribution. Null and positive results remain separate, and all tried variants remain visible. No general model-incapacity claim, selective winner-only report, or unadjusted family-wide confirmatory claim is warranted.
