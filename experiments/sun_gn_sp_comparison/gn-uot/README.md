# GN UoT: complete 100-case integrity result

The GN native UoT baseline now has a complete, independently replayed **100/100-case** result. Two MIT shards supplied 91 completed cases. They ended with nine execution failures, so two disjoint cloud suffixes generated only those nine missing cases. No successful MIT case was repeated.

The zero-model combined audit verified all four source journals independently, replayed every selected episode from its recorded responses and recovered the original global order. It made no model, API or GPU calls.

| Endpoint | Result |
|---|---:|
| Official / harness success | 100/100 |
| Hit at any guess | 100/100 |
| Actor-final exact match | unavailable for 100/100 |
| Native model requests | 12,803 |
| Selected-path environment queries | 774 total; mean 7.74 |

The 100/100 endpoint is the native harness result: the search resolved every case to the correct candidate. It is **not** evidence that the language-model actor itself produced 100 correct final answers, because actor-final exact match is unavailable for all cases. The selected-path query count also omits search-branch cost, so it is not a query-efficiency or speedup claim.

## Scope

- Model: Qwen2.5-7B-Instruct, a disclosed adaptation from the official 32B configuration.
- UoT depth: 1.
- Serving: 2,048 output-token cap and 16,384-token context.
- Proposal sampling: temperature/top-p 0.7/0.7; simulation sampling 0.1/0.9.
- Recovery: the two suffixes were planned after the MIT failures were observed; this is not a preplanned single-run execution.

This result closes baseline coverage and establishes execution integrity. It does not establish self-evolution, general model improvement, serial/concurrent equivalence or a scientific speedup.

Run `python -B verify_results.py results.json` to check the published arithmetic and claim guards. The public verifier does not authenticate private execution records; the retained private audit hash and four source journal/result hashes bind that evidence without publishing cases, targets, prompts, transcripts, credentials or infrastructure paths.
