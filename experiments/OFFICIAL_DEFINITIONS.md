# Official definitions we are bound to (no invention)

Everything below is quoted or transcribed from the official sources. Items the sources do not
specify are marked **unspecified** and are reported as our choice, never as the benchmark's.
Naming rule (2026-10-12): every table and score file uses the official names; our own
diagnostics are labelled as ours and never used as the headline.

## Multi-Turn Puzzles — Circuit Decoding (arXiv 2508.10142; data `arianhosseini/mt_puzzles`, CC-BY-4.0)

**Task name.** The paper writes "Circuit Decoding" ("Circuit Dec." in tables and figures); the
dataset files are prefixed `cd_`; the environment class named in the data is
`CircuitDecoderRuleBasedUserSimulator`.

**Task (Table 1, verbatim):** "Probe the C different boolean circuits for N turns. Predict the joint
truth table of all the circuits at N+1 turn." Sec. 2.1: "we keep C × 2^k = 24 and 300 unique circuit
configurations are constructed." Table 2: 300 problems, max 19 turns = "18 + 1". Appendix A.2 gives
the construction rules (fixed totals of AND/OR/NOT gates distributed over the C circuits; each
circuit has at least ⌈log2 k⌉ AND+OR gates; layer-by-layer construction).

**Official text we use unchanged:** `mt_puzzles/cd_instruction.txt` (system prompt; placeholders
filled per instance) and `mt_puzzles/cd_last_turn_prompt.txt` (final-guess prompt), both shipped
with the dataset. The environment code is not released; its wording is taken from the paper's
appendix transcript: `A(0, 0, 0) = 1` / `Please send your next query. 17 queries remaining.`, and
after the 18th query the answer plus the last-turn prompt.

**Metric (Table 1, verbatim, the only one listed for this task):** "Normalized circuit-wise
accuracy." Neither the paper nor the dataset card defines the normalization (**unspecified**; the
data carries a per-instance `avg_reveals` field, mean 17.9, whose role is not documented). Our
literal implementation, reported under the official name:
- **circuit-wise accuracy** = fraction of the C = 3 circuits whose full 8-row truth table is
  correct in the final guess (JSON key `circuit_wise_accuracy`), unnormalized.

Figures 1 and 5 plot one number per task labelled "Success Rate (%)" (Circuit Dec.: Gemini-2.5-Pro
39.8; Gemini-2.0 Pro 32.2 / Flash 7.2 / Flash-lite 4.1). The text does not say how that number is
derived for Circuit Decoding (**unspecified**); for Word Chaining it is defined as the fraction of
games completed without loss. We therefore also report
- **joint truth table correct** = fraction of episodes whose whole joint truth table (all 24 bits)
  is right (JSON key `joint_truth_table_correct`): the strictest literal reading of a successful
  episode, using the paper's own noun ("joint truth table"). This is our reading, not an official
  definition, and comparisons with the figures' "Success Rate" are therefore approximate.

Earlier versions of this sheet listed "Normalized % of trajectories with successful end state" as
a second Circuit Decoding metric. That is wrong: in Table 1 that metric belongs to Word Chaining.

Everything else we compute — uniq@end (feasible set collapsed to one hypothesis after the 18
queries), ex-ante efficiency (eff), zero-information rate, oracle agreement, repeat rate, bit-wise
accuracy — is **ours**, used for diagnosis only.

**Official baselines (model alone under the official protocol, no harness):** Fig. 1 "Success Rate
(%)", Circuit Dec.: Gemini-2.5-Pro-Exp-0325 **39.8**, OpenAI-GPT-4o-1120 **15.4**,
Claude-3.5-sonnet-1022 **9.3**. Fig. 5 (Sec. 3.2): Gemini-2.0-Pro-Exp 32.2, Gemini-2.0-Flash 7.2,
Gemini-2.0-Flash-lite 4.1. Table 2: average turns played on Circuit Dec. Gemini 2.5 10.5, Claude
3.5 19, GPT-4o 18.4. For our task model the baseline is the same protocol with no harness: `bare`
(Qwen2.5-7B-Instruct; sampling temperature 0.7 is **our choice**, the paper used API defaults).

**Unspecified by the official release (our choices, always stated):** what the environment does
when a reply contains no valid query (we re-ask once with a fixed message and the turn counts);
whether Circuit Decoding has an early-stopping condition (Table 2's note mentions early stopping
for Twenty Questions and Word Chaining only; we run all 18 + 1 turns); decoding parameters for
open models; the metric normalization; how the figures' "Success Rate" is computed for this task.

## AR-Bench (arXiv 2506.08295), for the Situation Puzzles / Detective Cases line

**Outcome metrics (Sec. 3, verbatim):** "the accuracy of identifying the true murderer in DC",
"the F1 score in the character level to assess similarity with the ground truth in SP", "the exact
match rate" for GN. **Process metrics:** based on the key questions defined during dataset
construction (per-task). **Judge / respondent:** Llama-3.1-405B (the paper reports 96% judge
accuracy on its own check); we use Llama-3.1-405B-Instruct AWQ-INT4 (82.0% on TurtleBench-150 vs
the bf16 model's 82.7% on the same items) and state the substitution. **Official methods** (the
paper's baselines): zero-shot, few-shot, proactive CoT, ToT, planning, plus fine-tuned variants.
