# GN restricted pilot: completed across two execution segments

The original allocation remains failed. The combined result adds only the previously unissued suffix after an exact cached replay of 4,032 actual responses. The suffix was planned after partial outcomes were visible. Integrity verification is not a scientific success claim.

All 30 development and 140 final slots are recorded. The same five development cases recur at updates 0, 3 and 6; there is one realized learning trajectory per method. Final memory is fixed at update 6, not selected on development results.

| Arm | Native success | Harness final EM | Actor final EM | Hit at any guess |
| --- | --- | --- | --- | --- |
| ZeroShot | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing |
| FewShot | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing |
| FewShotInst | 1/20 = 5.0%; 0/20 missing | 1/20 = 5.0%; 0/20 missing | 1/20 = 5.0%; 0/20 missing | 1/20 = 5.0%; 0/20 missing |
| ToT_parser_repair01 | 2/20 = 10.0%; 0/20 missing | 2/20 = 10.0%; 0/20 missing | 0/18 = 0.0%; 2/20 missing | 2/20 = 10.0%; 0/20 missing |
| ProactiveCoT | 1/20 = 5.0%; 0/20 missing | 1/20 = 5.0%; 0/20 missing | 1/19 = 5.3%; 1/20 missing | 1/20 = 5.0%; 0/20 missing |
| DC_Cumulative | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing | 0/20 = 0.0%; 0/20 missing |
| ACE_Supervised | 1/20 = 5.0%; 0/20 missing | 1/20 = 5.0%; 0/20 missing | 1/20 = 5.0%; 0/20 missing | 2/20 = 10.0%; 0/20 missing |

Native/harness success and final actor exact match are distinct. ToT uses target-informed branch selection; its early selected candidate does not define a final actor answer. PCoT random fallback is not an actor answer. The complete audit can identify valid final PCoT actor responses from raw text; the older partial package conservatively left its PCoT actor metric unassessed. Null remains missing, never failure.

These endpoints do not measure public-feedback singleton identification or query efficiency. This is a compute-controlled Qwen2.5-7B pilot with a 512-token actor cap and declared parser repairs, not a full paper/CLI reproduction. UoT, SFT and DPO remain unrun. ACE receives supervised training targets; DC receives public trajectories. No general method ranking or causal mechanism conclusion follows.

Usage: 5128 cumulative actual calls = 4,032 original responses + 1096 new suffix responses. Counts include all training, development and final phases. The cached prefix is not charged as new inference. Scheduler time is verified separately, not by this numeric export.

Run `python -B verify_results.py .` from this directory to recompute every public table and paired contrast. Intervals are posthoc, descriptive and based only on defined pairs. The verifier requires no private cases, vendor code, model or network; it does not authenticate execution provenance.

The anonymous case order preserves the audited bootstrap order. Whole-artifact hashes bind the private evidence without publishing source IDs, targets, prompts, transcripts, learned text, credentials, local/cluster paths or owner records. Original partial results remain unchanged as historical evidence.
