# What the completed GN matrix shows

All 30 development and 140 final evaluation slots are now recorded, with 12 learning updates. The final set contains the same 20 cases for each of seven arms. Development reuses five cases at updates 0, 3 and 6; each learner has one realized trajectory. Final memories were fixed at update 6, not chosen by development performance.

Native final success was **0/20 for DC-Cumulative and 1/20 for supervised ACE**. The inference baselines were ZeroShot 0/20, FewShot 0/20, FewShotInst 1/20, ToT with the declared parser repair 2/20, and ProactiveCoT 1/20. Both learners remained at **0/5 development exact match at all three checkpoints**. There is no observed development exact-match gain; these small final counts do not establish a reliable method ranking or a learning benefit.

A changed memory artifact or different full-text output is not sufficient evidence of learning. The useful negative here is that completing the matrix did not reveal development exact-match improvement. It also removes the earlier 43-cell coverage gap without treating missing observations as failures. This experiment does not isolate whether an update's behavior was caused by its prompt, output cap, extraction, merge, model, or other implementation choices.

Native success, harness final exact match, final actor exact match, and any-guess hits are different endpoints. For example, ToT has two native successes but 0/18 defined final-actor successes; ProactiveCoT has 1/19 defined final-actor successes. Undefined actor outcomes remain missing. These metrics differ from the previously reported **31% singleton-identification/query study** in [PR 2](https://github.com/Zarreboln/self-evolving-active-reasoning/pull/2); that result is not a directly comparable baseline for this matrix.

## Execution and interpretation limits

Completion used a bounded suffix after partial outcomes were visible, not a fresh blinded replication. The original allocation remains failed. Its 4,032 actual responses were replayed from the saved record, followed by 1,096 new responses: 5,128 actual calls counted once. The [original partial package](../gn/README.md) remains historical evidence.

Two earlier audit attempts also remain failed records. A separate corrected audit passed after its checks were aligned with the saved pending-response event and numeric wire representations. Original requests, responses and scores were not replaced. The accepted private audit has status `PASS_COMBINED_SUFFIX_INTEGRITY` and SHA256 `4d7f365446319c89cb1e4766f74b96755cc6adaf2bbcb8d8665639316d7688ea`. This is an integrity result, not a scientific-success claim.

ACE used supervised training targets while DC used public trajectories. The pilot uses Qwen2.5-7B, a 512-token actor cap, and declared method/parser adaptations. Descriptive paired intervals are available in [results.json](results.json); no claim is made here about every pairwise interval. Neither text changes nor this matrix establish the earlier proposed readout/merge mechanism.

Applicable UoT, SFT and DPO rows remain unrun in this contribution; SP baseline breadth and qualified feedback also remain gaps. The existing SP result is a small exploratory study with an unqualified same-family referee. Additional learning replications and a prospectively frozen matched random-schedule comparison are desirable future controls, not newly imposed requirements for delivering the completed evidence.

## How this fits the team's research question

The team's [main project claim](https://github.com/Zarreboln/self-evolving-active-reasoning/blob/main/README.md) allows state maintenance and vetoes while leaving question selection to the frozen actor. E0/E1 supply literal queries, so their **31.02% observed query-cost reduction is a separate policy-search comparator, outside that strict state-only condition**. A model printing a supplied query does not show that it selected the query. The result neither supports nor refutes the strict claim; it is not a matched ablation or demonstrated upper bound for it. Its singleton stopping rule and absence of a final-answer model call also differ.

| Evidence delivered | Role in the project | Boundary |
|---|---|---|
| Complete seven-arm GN matrix | Existing-method comparison and failure analysis | 20 cases per arm, adapted methods and unmatched supervision; native and actor-final endpoints remain separate |
| SP DC/ACE pilot | Exploratory evolution curves and metric/referee diagnostics | Five final cases, same-family referee qualification unmet; character F1 is not semantic accuracy |
| Selected GN schedules | A separate question-policy search comparator | Supplies concrete queries; no direct evidence for strict state-only evolution |
| Earlier C2 writer probe | Separates fixed-history answer writing from evidence acquisition | Uses first-generation histories; does not evaluate the team's later full SP harness |

Two context-method adaptations, descriptive evolution curves, failure analysis, proposal material and slides have been delivered. This does not imply all applicable baselines are reproduced or that a reliable strict-compatible evolution gain has been established. The team's own CD and 405B-referee SP evidence remains theirs; it cannot fill unmatched cells in these experiments.

A strict-compatible **future hypothesis**, not a result, is that a lossless numbered ledger of the same public query-feedback facts reduces exact repeated queries relative to chronological history. Hold actor, budget, decoding and veto/refinement rules fixed; supply no candidate queries, utility rankings or model-facing information gain. Report repeated queries and native final performance, disclose token-length differences, and keep offline information diagnostics separate. A hand-designed ledger would test state presentation, not by itself demonstrate evolution.

The [public verifier](verify_results.py) reproduces anonymous arithmetic and file hashes; it does not reproduce private execution provenance. The 13 copied numeric/figure files are unchanged. This interpretation is deliberately outside their numeric manifest.
