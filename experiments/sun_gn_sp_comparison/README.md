# GN and SP method-adapter pilots

Two bounded experiments compare native task algorithms with Dynamic Cheatsheet and ACE memory updates. The outcomes have different evidential limits.

| Experiment | Observed evidence | What it supports |
|---|---|---|
| GN, Qwen2.5-7B-Instruct | Full run failed; 30/30 development slots and 97/140 final slots completed; 43 final slots missing | Descriptive development curves and a posthoc common 13-case final prefix, with partial integrity/scoring verification only |
| SP, Kimi K2.6 | All 33 episodes and six updates completed across three execution segments; 27 evaluation rows; final N=5 per arm | Mechanically audited character/word-overlap descriptions under an **UNQUALIFIED same-family referee** |

Neither experiment establishes a general method ranking, a full paper/benchmark reproduction, or semantic reasoning improvement. There is one realized learning trajectory per method in each task. The GN and SP scores are different metrics and must not be pooled.

- [GN results, curve and limitations](gn/README.md): DC and supervised ACE both have 0/5 final-actor exact match at development updates 0/3/6. The 43 missing final slots are not scored as failures.
- [SP results and curve](sp/README.md): final mean character F1 is 0.6962 / 0.8353 / 0.8200 for ZeroShot / DC / ACE-Public-Feedback-SP. These are overlap scores, not accuracy. Final N=5; development reuses two cases.
- [SP fixed-subset feedback review](sp/feedback-source-review.json): execution-team AI labels 23 supported, one contradicted and six unresolved out of 30 prespecified replies. This does not supply human qualification or a population error rate.
- [Methods and provenance](METHODS.md), [upstream sources and licensing boundary](THIRD_PARTY.md), and [source-reference scope](source/README.md).

## Later completed GN work

- [Five inference baselines on the full 100-case test split](../sun_baselines100_20261027/gn-five/README.md): ZeroShot 0/100, FewShot 0/100, FewShotInst 2/100, ToT 5/100 and ProactiveCoT 2/100. The complete run and independent integrity audit are already part of this PR.
- [Acquired-evidence versus final-answer diagnostic](gn-information-use/README.md): recorded feedback uniquely identified the answer in 164/300 traditional-method trajectories, yet 162 of those ended wrong. Four trajectories guessed correctly earlier and then changed to a wrong final answer. This is post-hoc mechanism evidence, not corrected accuracy or a repair result.
- [GN post-training results](gn-posttraining/README.md): SFT completed 1,371 optimizer steps and scored 1/100 versus the matched base model's 0/100. That one-case difference is not reliable improvement. The DPO adapter was trained, but its fixed evaluation failed the output-format boundary and provides no effect estimate.

## Recompute the published numbers

Only Python's standard library is needed. From this directory:

```text
python -B gn/verify_results.py --data gn/anonymous-numerics.json --results gn/results.json
python -B sp/numeric/verify_results.py sp/numeric
python -B sp/verify_supplements.py
```

These commands check the anonymous numeric inputs and summaries, not private execution provenance or model behavior. They make no model or network calls. Whole-artifact hashes bind private original records without exporting those records. Native scoring was checked separately inside the execution team; this is not external replication. SVG figures are included; no binary image is needed.

## A testable next step, not an executed result

Freeze a finite candidate bank of complete ACE memory snapshots from the learning run. Compare the initial snapshot, the latest snapshot and a snapshot selected only on a separate validation set, all on the same fresh heldout cases, with the same monitor and explicit total cost accounting. This tests snapshot selection without treating the observed post-reflection/post-update pattern as proof that merging caused the difference. No such follow-up is included in these results.

What remains: the full audited UoT baseline, complete SP baseline coverage, qualified SP feedback or a separately justified evaluation condition, a valid DPO evaluation, SP post-training, reliable semantic outcome assessment, and replicated learning trajectories. The observed truncation and update stagnation motivate targeted checks; their causes are not isolated by these runs. The historical selected-policy GN studies are separate experiments, linked from the team log rather than duplicated here.

Private case identities, targets, prompts, interaction transcripts, learned text, credentials, cluster paths and coordination records are excluded. The retained SP `publication_ready=false` and `qualification=UNMET` flags describe the recorded scientific limitations; assembling this disclosure does not alter those flags. This package contains no execution or publication authorization.
