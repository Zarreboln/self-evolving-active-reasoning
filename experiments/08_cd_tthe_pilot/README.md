# 08 — Test-time harness evolution on Circuit Decoding: pilot streams (2026-10-12 16:14–17:51 EDT)

**Question**: with a frozen Qwen2.5-7B-Instruct as the questioner, can a harness grow *inside* the
deployment stream, with the proposer (Claude Opus) starting from nothing and the judge (Claude Opus)
never seeing a label? TTHE protocol (arXiv 2607.08124) ported to Circuit Decoding in `meta/cd_tthe/`:
batches of B = 10 held-out instances, G = 3 branches × R = 2 rounds of proposals per batch (roles:
conservative repair / independent exploration / adversarial audit), then one judge call that picks the
harness for the next batch (the incumbent is always in the pool = rollback gate). Prequential = the
incoming harness on the new batch *before* any adaptation; transductive = the committed harness on
the batch it was adapted on. 30 instances = 3 batches, stream seed 0, same instances in both arms.
Models: questioner Qwen2.5-7B-Instruct (vLLM, temperature 0.7, seed 0); proposer and judge are
`claude -p --model opus` via Claude Code 2.1.295, which the session metadata and billing resolve to
`claude-opus-5-5` (effort high; a fresh session per candidate and per judge call).
Arm `scratch`: no oracle library; the proposer sees only the system prompt, the harness interface
and raw traces. Two signal regimes: **strict** (run status only) and **labeled** (strict + each
episode's outcome). Metrics follow the paper (see [`../OFFICIAL_DEFINITIONS.md`](../OFFICIAL_DEFINITIONS.md)).

## Per-batch results (prequential on the incoming harness; 10 instances per batch)

| arm | batch | incoming → committed | circuit-wise accuracy | joint truth table correct | uniq@end (ours) | queries/ep (incoming) | wall |
|---|---|---|---|---|---|---|---|
| strict | 0 | bare → b0r0g2 | 10 | 0 | 20 | 17.3 | 2639 s |
| strict | 1 | b0r0g2 → b0r0g2 (kept) | **100** | **100** | 100 | 11.1 | 1390 s |
| strict | 2 | b0r0g2 → b2r0g2 | **100** | **100** | 100 | 12.9 | 1898 s |
| labeled | 0 | bare → b0r1g2 | 7 | 0 | 30 | 17.7 | 2987 s |
| labeled | 1 | b0r1g2 → b1r1g2 | **100** | **100** | 100 | 10.5 | 1001 s |
| labeled | 2 | b1r1g2 → b2r0g0 | **100** | **100** | 100 | 12.1 | 1731 s |

Stream totals (30 instances): strict prequential circuit-wise accuracy 70.0 / joint truth table
correct 66.7; labeled 68.9 / 66.7; transductive 100 / 100 in both. The official Fig. 1 "Success
Rate" of the strongest API model on this task is 39.8 (Gemini-2.5-Pro); bare 7B here is 0–0.5.
Costs (Claude Max subscription-priced): strict 14 proposer sessions $14.25 + 3 judge $1.63, 99 min;
labeled $13.17 + $0.86, 95 min. Files: `result_<arm>.json`, `opt_log_<arm>.jsonl`,
`trajectory_<arm>.png`, `judge_verdicts/`, `proposal_cards/`; candidates in
`meta/cd_tthe/agents/cand_pilot_*`; raw traces, session transcripts and per-episode records in the
release asset `cd_tthe_pilots_raw.tar.zst`.

## What grew, and how it was judged

* **Strict, batch 0 (the zero-hint result).** The round-0 *adversarial audit* proposer, given the
  bare traces and the system prompt, derived the hypothesis space from the task rules ("each circuit
  turns 3 inputs into 1 output, so it has exactly two 2-input gates plus NOTs; the inventory is split
  without replacement"), wrote an enumerator, and built a harness that shows the model the surviving
  candidates and a menu of up to 8 queries that still split them; the model writes every query
  (vetoed zero-information or repeated queries are re-asked a bounded number of times, then the
  model's own choice is sent) and picks each circuit's final table from numbered candidates.
  Pool on batch 0 (uniq@end / joint correct): bare .20/.00, r0g0 .30/.00, r0g1 .40/.00,
  **r0g2 1.00/1.00**, r1g0 .70/.00, r1g1 .80/.00, r1g2 1.00/1.00. The judge, seeing parsed%,
  queries, tokens, crashes, traces and source only, picked r0g2 and rejected r1g2 (menu of 3, other
  choices refused until the last try: "close to the harness choosing the queries").
  Trace audit on batch 1 (10 fresh instances): every sent query appears in a model reply (10/10
  episodes), the model wrote the final `<guess>` itself in 9/10, 11.1 queries per episode, 10/10
  joint truth tables correct.
* **Labeled, batch 0.** Three candidates were perfect (r0g2, r1g0, r1g2); r1g1 (a formula checker)
  timed out in 9/10 episodes. The judge chose r1g2 (same enumerator, suggestions ranked by binary
  entropy over the joint space) and said the tie-break was its judgement, not a measurement.
* **Batches 1–2 (saturated).** Every harness in both pools scored 10/10. Strict's judge kept the
  incumbent at batch 1 ("no candidate clearly better") and at batch 2 switched to b2r0g2 after
  running its own offline simulation of NOT-heavy inventories (over-budget runs 1/4 vs the
  incumbent's 12/28); that harness spends leftover queries re-checking its deductions (18.0
  queries/episode). Labeled's judge switched twice on query cost (9.9 vs 10.5, then 11.5 vs 12.1
  with 0 re-asks), found two round-1 candidates byte-identical to a round-0 one, and one candidate
  with an undefined attribute in a fallback path. Both judges flagged the `tokens_out = 0` column as
  an accounting gap (cache hits from the proposers' own check runs; fixed afterwards) and ignored it.

## Label-isolation audit (why there is a rerun)

Every proposer and judge session transcript (42) was scanned for gold values or instance truths in
tool results. **Strict batch 0 was clean**: the three round-0 proposers and the judge saw nothing
beyond run status, traces and source. From round 1 of batch 0 on, proposers read
`branch_lineage.jsonl` (aggregate outcome of earlier candidates) and, in batches 1–2, four proposers
read trace JSON heads containing per-episode gold records of the incumbent; the batch-2 strict judge
saw aggregate outcomes in the lineage (it said so and discarded them). No session saw an instance
truth table. So the zero-hint claim holds for the first step only; the stream as a whole does not
qualify. The loop was changed (worker reveal levels, controller-private store outside the repository,
public-only traces and logs, OFF-LIMITS paragraph in the prompts, automatic per-session audit that
discards a violating candidate or verdict) and both arms were rerun as `strict2_b10g3r2` /
`labeled2_b10g3r2` on the same 30 instances — see `09_cd_tthe_isolated/` once finished.

## Two more channels found while rerunning (2026-10-12 18:20)

* **Cross-session memory.** Claude Code keys its auto-memory by the repository root with every
  non-alphanumeric character turned into `-`; the loop's memory isolation computed a different slug
  and so always emptied a non-existent directory. From batch 2 of the pilots on, proposer sessions
  wrote notes to the real memory directory (archived in `cross_session_memory_notes/`: "trace JSON
  files hold gold, judge from .md only", "falsified query hypotheses", "how to get the system
  prompt") and later sessions, including the first sessions of the rerun, read them. Batches 0–1
  predate every note (earliest 17:25 EDT).
* **The sandbox itself.** The rerun's proposers found the pilots' evolved candidates in `agents/`
  and the pilots' logs in `logs/`; one built its candidate by copying the body of a pilot harness.
  That rerun (`strict2` / `labeled2`) was stopped at batch 0 and discarded.

Fixes before the clean rerun (`strict3` / `labeled3`): every memory directory under the repo root's
slug is emptied before and archived after each session; `agents/` holds only `bare.py`, `logs/` only
the current run; the session audit additionally rejects reads outside `meta/cd_tthe/`, other runs'
logs, `.claude/` (memory, read or write), and git commands; the OFF-LIMITS paragraph says so.

## Takeaways

* A proposer with no library, no labels and no prior about the model's habits reconstructs the
  benchmark's hypothesis space from the rules in the system prompt in one round; a label-free judge
  selects it; the result transfers to fresh instances at 100% (20/20 across batches 1–2).
* Once accuracy saturates, selection turns on secondary criteria (queries, robustness to harder
  inventories, code hygiene); the judges' tie-breaks are judgements and should be read as such.
* Label isolation has to be enforced by what is on disk and by auditing the transcript, not by the
  prompt: Opus reads any JSON it finds.
