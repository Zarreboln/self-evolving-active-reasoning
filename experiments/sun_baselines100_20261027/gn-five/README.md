# Five GN inference baselines, 100 official test cases each

Five adapted ARBench inference baselines completed all **500 method-case episodes** and **12,805 model calls**. The independent recorded-response replay passed; root separately checked scheduler completion, cleanup and serving-model startup logs. These are completed baselines, **not evidence of self-evolution benefit**. UoT, SFT/DPO and the new SP comparisons are not included in this package.

The native/harness success counts are low:

| Method | Native successes / 100 | Any logged hit / 100 | Mean selected-path questions | Model calls | Truncated outputs |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 1 | 25.00 | 2,600 | 0 |
| Few-shot | 0 | 2 | 25.00 | 2,600 | 0 |
| Few-shot + instruction | 2 | 3 | 25.00 | 2,600 | 3 |
| ToT, parser-repaired | 5 | 5 | 24.10 | 2,505 | 0 |
| Proactive CoT | 2 | 2 | 24.91 | 2,500 | 10 |

## What this establishes

This supplies five independently executed 100-case baseline rows rather than citing someone else's scores or extrapolating from the earlier 20-case pilot. Under this particular 7B configuration, prompt/search complexity alone did not produce reliable native task solving. It does not establish why the methods fail or whether a larger model, training or state evolution would improve them.

ToT has five native successes versus zero for zero-shot. The paired exact two-sided McNemar p-value is 0.0625 before any multiplicity correction. This single run does not establish a reliable ranking. The 95% Wilson interval for ToT's native success rate is approximately 2.15%–11.18%; zero-shot's upper bound is approximately 3.70%. All descriptive intervals and paired discordant counts are in `analysis.json`.

## Endpoint and comparability limits

- Native/harness success is distinct from a final actor-generated answer. ToT's five successful rows have actor-final EM unavailable; Proactive CoT does not provide a uniformly identifiable actor-final endpoint. Do not impute actor success from harness success.
- Plain prompting runs 25 answered questions and then a final prediction. Its 26 logged guesses therefore do not mean 26 environment questions.
- ToT also scores candidate branches against the oracle. Selected-path question counts omit those branch evaluations. The table is not a complete search-oracle cost comparison, and lower counts at such low success rates are not reliable questioning-efficiency gains.
- Qwen2.5-7B-Instruct substitutes for the official script's 32B model. The explicit 2,048 output-token cap, 16,384 context limit, disclosed ToT parser repair and concurrent serving are adaptations. This is not an exact reproduction of the paper's reported setting.
- One seed schedule and the official 100-case test set are used. Earlier pilot outcomes had been seen. There are no new held-out learning replications here. Prior 512-token pilot scores must not be pooled with these 2,048-token baseline scores.
- All 13 truncated responses and three Proactive CoT random fallbacks remain recorded. No cell was silently dropped or rerun.

## Reproduce the public arithmetic

Run `python verify_results.py` in this directory. It checks file hashes, all five-by-100 numeric cells and the reported sums/means. `numeric-cases.csv` contains only anonymous case indices and numerical outcomes/costs. `audit-result.json` and `audit-execution.json` are byte-identical downloaded audit outputs. `provenance.json` records the separate root observations and source hashes.

This public check does not rerun a model or repeat the private transcript audit. Private cases, answers, prompts, transcripts, credentials and teammate/presentation drafts are excluded. The original raw result retains its historical pre-audit status; the independent audit is a separate immutable artifact.
