# Self-Evolving Harnesses for Active Reasoning

MAS.S62 *Self-Evolving AI* (Fall 2026) course project.
**Zining Liu** (ziningl@mit.edu) · **Yangzi Sun** (sunyz@mit.edu) — Massachusetts Institute of Technology

## The claim

An active-reasoning agent has to *ask* for what a problem withholds, one question at a time. The
dominant failure of a small model is not a wrong final answer but a stalled dialogue: questions
that stop narrowing what the answer could be. AR-Bench's own paper reports that SFT, DPO, ToT,
hand-written instructions, Proactive CoT and UoT all give it almost nothing. T3 (ICLR 2026) attacks
it with RL, but its trigger is a per-task hand-made proxy whose threshold the authors say is not
measurable.

We do not change the weights. We wrap the frozen model in a **harness** — what it is shown each
turn, what it is asked again, what is refused — and let that harness **evolve at test time, without
labels**, from the model's own failure traces. The questioner is a frozen Qwen2.5-7B-Instruct. The
proposer that writes candidate harnesses and the judge that picks among them are Claude Opus 5.5,
and the judge never sees a gold answer.

**The line that makes this a claim rather than a trick:** the harness may maintain state and veto,
but may not compute information gain or choose queries. **The frozen model writes every question.**
The judge enforces it — it rejected three harnesses that scored 10/10 because they handed the model
a menu to pick from.

## What we ran

Two task families, thirteen nodes. **Start here: [`notes/EXPERIMENTS_LOG.md`](notes/EXPERIMENTS_LOG.md)**
— one folded, traceable tree with every experiment's date, wall time, headline and caveats:
**A1–A8** the evaluation substrate (official definitions, task ceiling, judge calibration, baselines,
the metric audit, the independent replication) and **B1–B6** harness evolution (one-shot, test-time,
label isolation, cross-task, the judge's own ruler). Each leaf links into
[`experiments/`](experiments/), which holds the per-run commands, score files and traces.

### Circuit Decoding (Multi-Turn Puzzles, arXiv 2508.10142)

Three hidden boolean circuits, 18 one-bit queries, then the 24-bit joint truth table. The announced
gate inventory makes the hypothesis space enumerable, so the feasible set, the information of every
query and a max-entropy oracle are computable without the truth — which is why the harness can be
given *any* state and measured per turn.

| system | who wrote it | circuit-wise accuracy | joint truth table correct |
|---|---|---|---|
| bare (no harness) | — | 11.8–12.9 | 0.0–0.5 |
| EvoHarness-RL, ReAct / inline | official framework | 4.3–4.9 / 5.7–6.9 | 0.0 / 0.7–1.7 |
| enum15 (state summary every turn) | hand-written | 23.7 | 5.8 |
| EvoHarness-RL, always-on panels | official framework | 74.8–79.0 | 62.0–65.0 |
| oracle hint (max-entropy query every turn) | hand-written | 27.3 | 6.8 |
| veto_only (one-shot evolution, state_only) | proposer | 91.3 | 86.2 |
| **TTHE strict, final (label-free, 200 held-out)** | proposer + label-free judge | **99.5** | **98.5** |
| TTHE labeled, final | proposer + judge with outcomes | 100 | 100 |

`circuit-wise accuracy` is the benchmark's own metric; `joint truth table correct` is the paper's
noun for a solved episode. Everything else we report is marked (ours). Names and what the paper
leaves undefined: [`experiments/OFFICIAL_DEFINITIONS.md`](experiments/OFFICIAL_DEFINITIONS.md).

**strict ≈ labeled is the evidence for "label-free".** Both streams end at the ceiling; the
label-free one needs one more batch.

### Situation Puzzles (AR-Bench, arXiv 2506.08295)

No enumerable hypothesis space: a free-text hidden story, 25 yes/no questions answered by the
official Llama-3.1-405B referee, then a written answer. Frozen evaluation on all 100 test puzzles,
70 of which the evolution stream never saw.

| harness | F1 (char) | F1, unseen 70 | key-question recall | tokens/ep |
|---|---|---|---|---|
| bare (official zero-shot loop) | 60.4 | 61.4 | 59.7 | 670 |
| first committed | 76.7 | 75.7 | 59.0 | 4486 |
| final committed | 75.6 | 75.4 | **68.7** | 5263 |
| unchosen `b2r0g1` | 76.3 | **77.5** | 68.2 | **2112** |

For context, AR-Bench Table 6 zero-shot: GPT-4o 63, Llama-3.1-405B 55, Qwen-2.5-7B 51.

### …and why we do not take that table at face value

[Experiment 11](experiments/11_metric_audit/) audits the metric on these same 500 episodes.
**Answer length alone explains 65% of the variance in character-level F1 (r = +0.806); key-question
recall explains almost none (r = +0.050).** Hold answer length fixed and the headline +15.2 point
gain shrinks to between +0.4 and +7.2. The metric has no effective length penalty, nothing forces
the final answer to follow from the questions asked, and the referee itself is right about 83% of
the time.

So the honest reading is: **the later generations do learn to ask better** — recall 57.6 → 66.4 on
unseen puzzles, and the cheapest candidate reaches the best F1 with half the tokens, which a pure
length story cannot explain — **but a large part of the headline number is the benchmark's ruler,
not our method.** We keep reporting the official metric and put this beside it.

An independent preregistered replication by Yangzi Sun (65 cases, 260 generations, 520 blind
ratings) separates the questioning history from the final-answer instruction and adds semantic
grading: [report](https://github.com/sun-tze/self-evolving-active-reasoning/blob/sun/c2-confirmatory-65/experiments/sun_c2_confirmatory/REPORT.md).

### The judges

Two different judges, only one of which had a ruler. [Experiment 02](experiments/02_judge_calibration/)
calibrates the **referee** (the model that answers the player) on TurtleBench-150: the official
Llama-3.1-405B-INT4 scores 83.3, Opus 5 88.0, and eleven candidates were compared case by case —
none is a drop-in. By macro-averaged recall the order inverts, because every candidate collapses on
the "Unknown" class. The **TTHE judge** that commits harnesses needs no external benchmark: every
run stores the pool's gold, which that judge never saw, so pick accuracy and regret are directly
measurable ([`runner/judge_pick_eval.py`](runner/judge_pick_eval.py)). It picks the best harness
3/3 on labeled CD, 2/3 on strict CD, and **1/3 on Situation Puzzles**, giving up 0.114 F1 at worst.

## How the loop works

```
batch of instances ─▶ run the incumbent harness ─▶ traces (status only, no gold)
                                                        │
                        proposer (Opus 5.5, G branches × R rounds) ─▶ candidate harnesses
                                                        │
                        judge (Opus 5.5, label-free)  ─▶ commit or keep the incumbent
                                                        │
                                                   next batch
```

Label isolation is enforced, not assumed: gold records live in a controller-private store outside
the repository, workers receive a `reveal` level, and every proposer and judge session is audited
afterwards (path resolution with cwd tracking, heredoc bodies, value scans for gold strings). A
violation discards the candidate. Audits ship with each run as `session_audits.jsonl`.

## Layout

| path | content |
|---|---|
| [`experiments/`](experiments/) | one folder per experiment: setup, command, timings, results, takeaway |
| [`notes/EXPERIMENTS_LOG.md`](notes/EXPERIMENTS_LOG.md) | the traceable tree: A = evaluation substrate, B = harness evolution |
| `meta/cd_tthe/` | the test-time loop: `optimize.py`, `proposer.py` (prompts + audit), `evaluate.py`, task adapters for CD and SP, candidate harnesses in `agents/` and `agents_sp/` |
| `meta/cd_evolve/` | the earlier one-shot evolution loop |
| `runner/` | episode runners, scorers, judge calibration (`turtle_judge_eval.py`, `typed_judge_eval.py`, `judge_pick_eval.py`, `metric_audit.py`) |
| `baselines/evoharness_cd/` | our Circuit Decoding adapter for EvoHarness-RL |
| `engaging/`, `h100/` | cluster scripts and the machine notes for both servers |
| `RELATED_WORK.md`, `meta/journal.md`, `THIRD_PARTY.md` | literature with how far each prior work occupies our position, dated process journal, pinned external repos |

Raw per-episode traces are attached to the GitHub Release, one tarball per experiment family.

## Reproduce (sketch)

```bash
# one test-time evolution stream on Situation Puzzles (needs the claude CLI for the proposer,
# a served Qwen2.5-7B, and the 405B referee)
cd meta/cd_tthe && python optimize.py --task sp --run-name demo --signals strict \
    --proposer-backend claude --proposer-model opus --batch-size 10 --group 3 --rounds 2 \
    --solver-base-url http://127.0.0.1:18000/v1 --answerer-base-url http://127.0.0.1:18001/v1 \
    --answerer-model meta-llama/Llama-3.1-405B-Instruct-INT4

# frozen evaluation of finished harnesses, no proposer and no judge
python evaluate.py --task sp --out logs/eval --harnesses bare cand_... --split test --workers 12

# what the metric is really measuring
python ../../runner/metric_audit.py logs/eval --baseline bare --evolved cand_...
```
