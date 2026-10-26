# What the completed GN matrix shows

All 30 development and 140 final evaluation slots are now recorded, with 12 learning updates. The final set contains the same 20 cases for each of seven arms. Development reuses five cases at updates 0, 3 and 6; each learner has one realized trajectory. Final memories were fixed at update 6, not chosen by development performance.

Native final success was **0/20 for DC-Cumulative and 1/20 for supervised ACE**. The inference baselines were ZeroShot 0/20, FewShot 0/20, FewShotInst 1/20, ToT with the declared parser repair 2/20, and ProactiveCoT 1/20. Both learners remained at **0/5 development exact match at all three checkpoints**. There is no observed development exact-match gain; these small final counts do not establish a reliable method ranking or a learning benefit.

A changed memory artifact or different full-text output is not sufficient evidence of learning. The useful negative here is that completing the matrix did not reveal development exact-match improvement. It also removes the earlier 43-cell coverage gap without treating missing observations as failures. This experiment does not isolate whether an update's behavior was caused by its prompt, output cap, extraction, merge, model, or other implementation choices.

Native success, harness final exact match, final actor exact match, and any-guess hits are different endpoints. For example, ToT has two native successes but 0/18 defined final-actor successes; ProactiveCoT has 1/19 defined final-actor successes. Undefined actor outcomes remain missing. These metrics differ from the previously reported **31% singleton-identification/query study** in [PR 2](https://github.com/Zarreboln/self-evolving-active-reasoning/pull/2); that result is not a directly comparable baseline for this matrix.

## Execution and interpretation limits

Completion used a bounded suffix after partial outcomes were visible, not a fresh blinded replication. The original allocation remains failed. Its 4,032 actual responses were replayed from the saved record, followed by 1,096 new responses: 5,128 actual calls counted once. The [original partial package](../gn/README.md) remains historical evidence.

Two earlier audit attempts also remain failed records. A separate corrected audit passed after its checks were aligned with the saved pending-response event and numeric wire representations. Original requests, responses and scores were not replaced. The accepted private audit has status `PASS_COMBINED_SUFFIX_INTEGRITY` and SHA256 `4d7f365446319c89cb1e4766f74b96755cc6adaf2bbcb8d8665639316d7688ea`. This is an integrity result, not a scientific-success claim.

ACE used supervised training targets while DC used public trajectories. The pilot uses Qwen2.5-7B, a 512-token actor cap, and declared method/parser adaptations. Descriptive paired intervals are available in [results.json](results.json); no claim is made here about every pairwise interval. Neither text changes nor this matrix establish the earlier proposed readout/merge mechanism.

UoT, SFT and DPO remain unrun. Qualified SP feedback, replicated learning trajectories, and a prospectively frozen matched random-schedule comparison of query efficiency remain open requirements. The existing SP result remains a small exploratory study with an unqualified same-family referee.

The [public verifier](verify_results.py) reproduces anonymous arithmetic and file hashes; it does not reproduce private execution provenance. The 13 copied numeric/figure files are unchanged. This interpretation is deliberately outside their numeric manifest.
