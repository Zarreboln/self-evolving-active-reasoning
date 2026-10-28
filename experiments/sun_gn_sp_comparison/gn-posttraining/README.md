# GN post-training results

This delta reports the completed GN training-data collection, the SFT run and its fixed 100-case evaluation, plus the preserved DPO evaluation failure. It publishes aggregate counts and integrity hashes only; no private cases, targets, prompts, transcripts, learned adapter weights or credentials are included.

## Training data

The fixed 500-case training collection completed with **37,458 calls**, producing **7,299 SFT rows** and **3,354 DPO preference pairs** from **12,486 eligible states**. The collection audit found zero excluded states and zero invalid responses. Only **3/500** collection episodes solved the task; 497 exhausted the state budget. Corpus inspection found zero dropped rows and zero token truncations, with maximum encoded length 1,372 under the 8,192-token training limit.

These counts establish corpus completeness under the declared transformation. They are not evidence that the resulting adapters improve task performance.

## SFT

SFT completed three epochs over 7,299 rows with **1,371/1,371 optimizer steps**. The saved trainer state reported loss 0.1732545150. A fixed zero-shot evaluation then completed all **100/100** official test cases with **2,600 model responses**, zero truncated responses and an independent replay-integrity PASS.

| Model condition | Official GN success |
|---|---:|
| Matched base zero-shot baseline | 0/100 |
| SFT adapter | 1/100 |

The single discordant success is a one-case difference. It is not reliable evidence of improvement or self-evolution. The SFT run also recorded three hit-at-any cases, but the official endpoint is the final answer, not an intermediate hit.

## DPO

The DPO adapter passed structural artifact checks, but its fixed evaluation did not produce a valid completed case. Eight cases were attempted; one produced an invalid-length final guess and the shared no-retry boundary halted the other seven. The run preserved 92 completed model calls and zero response truncations.

This is an evaluation-format failure, not a DPO effect estimate. No claim about DPO benefit, harm or self-evolution is supported by this attempt.

## Reproduce the public checks

From this directory, run:

```text
python -B verify_results.py results.json
```

The verifier checks the public aggregate schema, arithmetic and claim guards. It does not rerun models or independently audit private execution evidence.

## Scope and provenance

- Model family: Qwen2.5-7B-Instruct, a disclosed adaptation from the upstream 32B configuration.
- Evaluation: the same official 100-case zero-shot protocol used by the published base baseline.
- Training and test splits were kept separate. Test trajectories were not used for training or tuning.
- The SFT evaluation audit hash and collection/token-inspection hashes are recorded in `results.json`.
- UoT completion and the full SP baseline are separate unfinished work and are not included here.
