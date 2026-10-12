# Official definitions we are bound to (no invention)

Everything below is quoted or transcribed from the official sources; items the sources do not
specify are marked **unspecified** and must be reported as our choice, never as the benchmark's.

## Multi-Turn Puzzles — CircuitDecoding (arXiv 2508.10142; data `arianhosseini/mt_puzzles`, CC-BY-4.0)

**Task (Table 1, verbatim):** "Probe the C different boolean circuits for N turns. Predict the joint
truth table of all the circuits at N+1 turn." Evaluation set: C x 2^k = 24, 300 unique circuit
configurations (Sec. 2). Max turns 19 = 18 interaction turns + 1 final prediction turn (Table 2;
Sec. 3.2: "18+1").

**Official text we use unchanged:** `mt_puzzles/cd_instruction.txt` (the system prompt; placeholders
filled per instance) and `mt_puzzles/cd_last_turn_prompt.txt` (the final-guess prompt), both shipped
with the dataset. The environment is `CircuitDecoderRuleBasedUserSimulator` (named in the data,
code not released); its wording is taken from the paper's appendix transcript:
`A(0, 0, 0) = 1` / `Please send your next query. 17 queries remaining.`, and after the last query
`C(0, 0, 0) = 0` / `Please provide the truth tables for all circuits in the format requested.`

**Metrics (Table 1, verbatim):** "Normalized circuit-wise accuracy." and "Normalized % of
trajectories with successful end state." Figures 1 and 5 report one number per task labelled
"Success Rate (%)". The paper and the dataset card do not define the normalization
(**unspecified**); the data carries a per-instance field `avg_reveals` (mean 17.9 over the 300
instances) whose role is not documented. Our literal implementations, reported under these names:
- `circuit-wise accuracy` = fraction of the 3 circuits whose full 8-row truth table is correct
  (`ckt` in our score files);
- `success` = all 24 bits of the final guess correct (`exact` in our score files).
Everything else we compute (uniq@end, ex-ante efficiency, zero-information rate, oracle agreement,
repeat rate) is **ours**, used for diagnosis only, never as the headline.

**Official baselines (model alone under the official protocol, API default sampling, no harness):**
Figure 1, Circuit Dec. success rate: Gemini-2.5-Pro-Exp-0325 **39.8**, OpenAI-GPT-4o-1120 **15.4**,
Claude-3.5-sonnet-1022 **9.3**. Figure 5 (Sec. 3.2): Gemini-2.0-Pro-Exp 32.2, Gemini-2.0-Flash 7.2,
Gemini-2.0-Flash-lite 4.1. For our task model the baseline is the same protocol with no harness:
`bare` (Qwen2.5-7B-Instruct, temperature 0.7 — the paper used API defaults; **our choice**).

**Unspecified by the official release (our choices, always stated):** what the environment does
when a reply contains no valid query (we re-ask once with a fixed message and the turn counts);
whether CD has an early-stopping condition (the paper mentions early stopping only for other
tasks; we run all 18 + 1 turns); decoding parameters for open models; the metric normalization.

## AR-Bench (arXiv 2506.08295), for the Situation-Puzzles / Detective-Cases line

**Outcome metrics (Sec. 3, verbatim):** "the accuracy of identifying the true murderer in DC",
"the F1 score in the character level to assess similarity with the ground truth in SP", "the exact
match rate" for GN. **Process metrics:** based on the key questions defined during dataset
construction (per-task). **Judge / respondent:** Llama-3.1-405B (the paper reports 96% judge
accuracy on its own check); we use Llama-3.1-405B-Instruct AWQ-INT4 (82.0% on TurtleBench-150 vs
the bf16 model's 82.7% on the same items) and state the substitution. **Official methods** (the
paper's baselines): zero-shot, few-shot, proactive CoT, ToT, planning, plus fine-tuned variants.
