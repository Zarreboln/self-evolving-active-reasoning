# 10 — Test-time harness evolution on AR-Bench Situation Puzzles (2026-10-13 05:03–07:17 EDT)

The TTHE loop of [09](../09_cd_tthe_isolated/README.md) on a task with no enumerable hypothesis space.
Questioner: Qwen2.5-7B-Instruct (frozen, temperature 0.7). Referee and key-question judge: the official
Llama-3.1-405B (AWQ-INT4; TurtleBench-150 83.3 % vs the paper's bf16 82.7 %), through AR-Bench's own
referee and key-question prompts. Protocol: AR-Bench's player instruction, 25 yes/no questions, then the
final answer; outcome metric = official character-level F1 against the hidden story; process metric =
share of the puzzle's key questions hit (judged by the referee). Run `sp_strict1_b10g3r2`: 30 of the 100
test puzzles (stream seed 0), B = 10, G = 3, R = 2, **strict** signals (run status + referee replies only,
no F1 shown to proposer or judge), Claude Opus 5.5 proposer and judge inside Claude Code, full label
isolation and session audits as in 09. 12 sessions, $11.92 subscription-priced, 2 h 14 min wall;
18/21 sessions clean (3 proposer candidates discarded for mis-counted relative paths
that resolved outside the folder but did not exist — the audit now ignores such failed commands).

## Stream (prequential = incoming harness on the new batch before adaptation)

| batch | incoming → committed | F1 (char) | key-question recall | transductive F1 | pool | wall |
|---|---|---|---|---|---|---|
| 0 | bare → b0r1g2 | 63.7 | 55.0 | 79.4 | 7 | 2623 s |
| 1 | b0r1g2 → b1r1g0 | 74.5 | 65.0 | 67.4 | 5 | 2872 s |
| 2 | b1r1g0 → b2r1g0 | 79.9 | 71.7 | 68.8 | 6 | 2477 s |

Stream totals: prequential F1 63.7 → 74.5 → 79.9 (mean 72.7),
key-question recall mean 63.9.

**What grew.** Batch 0's winner (adversarial-audit proposer, round 1): each turn the model gets one fresh
message — the puzzle, the referee's replies grouped as CONFIRMED / RULED OUT / NOT IN THE STORY, and a
focus: first the assumptions a reader takes for granted (situation puzzles hinge on one being false), then
a *working story* the model maintains that must contain every confirmed fact, whose unconfirmed parts are
tested one at a time; after a run without a Yes, one broad question on an unasked basic aspect. Every
proposed question passes format checks (one fact, yes/no, no "or", not about a character's beliefs, no
invented names, no near-repeat); the final answer is rewritten from the confirmed facts. Batches 1 and 2
added vetoes on repeats and denied leads, and bans on entities the referee turned down twice and on
"the X" phrasing for unconfirmed guesses. The model writes every question and the final answer.

**The judge on SP.** With no oracle, the judge compared traces for repeated questions, leads chased after
a No, and final answers contradicting the referee, and twice called its choice "a judgement call, not a
clear win". At batch 1 it committed the pool's lowest-F1 candidate (0.674 vs the incumbent's 0.745) and at
batch 2 it passed over the best (0.802); on 10-puzzle batches those gold differences are within noise,
and the batch-1 choice then scored 0.799 prequentially on batch 2. On 100 puzzles (below) all five evolved
harnesses are within three F1 points of each other.

## Frozen evaluation: all 100 test puzzles (70 never used by the stream), no proposer, no judge

`evaluate.py --task sp`, 12 workers, 405B referee. The labeled-arm numbers of the paper are for context only:
AR-Bench Table 6 (zero-shot, char-level F1): Qwen-2.5-7B 51, Llama-3.1-405B 55, GPT-4o 63.

| harness | F1 (char), 100 | F1 unseen 70 | key-question recall, 100 | recall unseen | tokens/ep | crashes |
|---|---|---|---|---|---|---|
| no harness (AR-Bench zero-shot loop) | 60.4 [56.8,63.9] | 61.4 | 59.7 [53.3,65.7] | 57.6 | 670 | 0 |
| b0 commit: premise checks, one working story keeping every confirmed fact, question-format checks, answer rewritten from confirmed facts | 76.7 [74.5,79.1] | 75.7 | 59.0 [53.0,64.8] | 57.1 | 4486 | 0 |
| b1 candidate not chosen | 73.0 [70.0,76.0] | 73.8 | 64.3 [58.8,69.8] | 65.5 | 4685 | 0 |
| b1 commit: + repeat / denied-lead vetoes, contradiction check on the final answer | 76.2 [73.4,78.8] | 75.9 | 63.2 [57.8,68.5] | 60.0 | 4944 | 0 |
| b2 candidate not chosen (gold 0.802 on its batch), half the tokens | 76.3 [74.1,78.6] | 77.5 | 68.2 [63.3,73.3] | 67.6 | 2112 | 0 |
| b2 commit (final): + no entity denied twice, no 'the X' for unconfirmed guesses, invented-name check | 75.6 [73.4,77.8] | 75.4 | 68.7 [63.3,74.0] | 66.4 | 5263 | 0 |

**Reading.** Label-free evolution from ten failure traces lifts the frozen 7B from 60.4 to 74–77 F1 on
70 puzzles it never saw — above every zero-shot model in AR-Bench's own table, including GPT-4o (63),
with the 7B still writing every question. Key-question recall rises from 60 to 68–69 only in the later
generations (the first generation improves the *answer*, not the questioning). The cost is 3–8× more
output tokens per puzzle. The unchosen b2r0g1 matches the committed harness at half the tokens. b1r0g0 (re-run on 10-13 after the
405B referee was preempted mid-evaluation) is the weakest of the five at 73.0, still 13 points over bare.

Files: `result.json`, `opt_log.jsonl`, `branch_lineage.jsonl`, `session_audits.jsonl`, `trajectory.png`,
`judge_verdicts/`, `proposal_cards/`, `candidates/`, `eval_test100_scores.json`. Raw traces, transcripts,
controller records and evaluation records: release asset `sp_tthe_pilot_raw.tar.zst`.
