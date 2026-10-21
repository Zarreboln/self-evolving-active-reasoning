# 11 — What the Situation Puzzles metric measures (2026-10-21)

AR-Bench scores Situation Puzzles with character-level F1 between the model's answer and the
hidden bottom. This experiment asks what that number actually tracks. It runs on records we
already had: the frozen evaluation of five harnesses on all 100 test puzzles
(`meta/cd_tthe/logs/eval_sp100/`), 500 episodes, no new inference.
Script: [`runner/metric_audit.py`](../../runner/metric_audit.py), raw output in `output.txt`.

The harness `cand_sp_strict1_b10g3r2_b1r0g0` is excluded: its first evaluation was interrupted when
the 405B referee was preempted, leaving zero-word answers at zero F1. Those points sit on the
length axis and inflate the very correlation under test (0.90 with them, 0.81 without). This is
recorded because the inflated figure was quoted once before the records were cleaned.

## F1 tracks answer length, not question quality

| correlated with character-level F1 | r |
|---|---|
| **answer length in words** | **+0.806** |
| output tokens for the whole episode | +0.403 |
| **key-question recall** (our process metric) | **+0.050** |

| model of F1 | R² |
|---|---|
| answer length alone | 0.650 |
| answer length + key-question recall | 0.652 (recall coefficient +0.021) |

With length partialled out, recall correlates with F1 at r = +0.068. **Answer length alone explains
65% of the variance in the benchmark's own metric; how much of the puzzle the questioning actually
uncovered explains almost none of what is left.**

Mean F1 rises monotonically with length, with no turning point — there is no effective length
penalty in the metric:

| answer length | n | mean F1 |
|---|---|---|
| 25–50 words | 44 | 0.458 |
| 50–75 | 94 | 0.634 |
| 75–100 | 190 | 0.766 |
| 100–125 | 134 | 0.826 |
| 125–150 | 26 | 0.868 |

## How much of our headline gain survives length control

| answer length band | bare | evolved (b2r1g0) | gap |
|---|---|---|---|
| 25–50 words | 0.459 (n=28) | 0.464 (n=6) | +0.005 |
| 50–75 | 0.606 (n=31) | 0.678 (n=15) | **+0.072** |
| 75–100 | 0.734 (n=24) | 0.777 (n=45) | +0.043 |
| 100–125 | 0.804 (n=9) | 0.808 (n=29) | +0.004 |
| **pooled (the headline)** | **0.604** | **0.756** | **+0.152** |

The evolved harness writes a 91-word answer where bare writes 61. Most of the headline +15.2 points
is that difference. Within a band the advantage is +0.4 to +7.2 points and shrinks as answers get
longer, which is consistent either with the advantage being small or with F1 saturating at length.
**F1 alone cannot separate those two readings.** Semantic grading can, which is what
[Sun's C2 probe](#independent-replication) and the planned blind semantic evaluation are for.

## What this implies about the benchmark

- **The metric rewards verbosity.** Nothing in it punishes a long answer that sprays plausible
  content. A method that lengthens answers gains score without understanding anything more. Our own
  harness was pushed along exactly this gradient: 670 → 5263 output tokens per episode.
- **Step-skipping is not prevented.** The task never requires the final answer to follow from the
  recorded question-answer history. An agent may waste all 25 questions and write a long answer from
  the surface alone. *Not yet quantified:* the F1 a degenerate answer (the surface text itself, or
  generic filler) scores against the bottoms. That needs the gold bottoms, which live on the servers.
- **The environment carries its own noise.** The referee is an LLM; our calibration puts the
  official Llama-3.1-405B-INT4 at 83.3% on TurtleBench-150, so roughly one answer in six that the
  questioner receives may be wrong. See [02](../02_judge_calibration/).
- **Reference quality varies.** In Sun's 65-case set, 8 puzzles were flagged for reference
  inconsistency before generation; their pooled semantic mean was 0.625/2 against 0.754/2 elsewhere.

**None of this changes what we report.** AR-Bench's metric is character-level F1 and our main tables
keep it, per [`OFFICIAL_DEFINITIONS.md`](../OFFICIAL_DEFINITIONS.md). This experiment is the
limitation section, not a replacement metric.

## Independent replication

Yangzi Sun ran a preregistered 65-case probe that separates the questioning history from the
final-answer instruction (bare / evolved dialogue × plain / evidence-constrained writer, 260
generations, 520 blind ratings), adding Semantic Correctness and Evidence Faithfulness. It reports
82 strict F1-versus-semantic rank reversals across 29 cases, and that the evidence-constrained
instruction did not improve semantic correctness. Report:
<https://github.com/sun-tze/self-evolving-active-reasoning/blob/sun/c2-confirmatory-65/experiments/sun_c2_confirmatory/REPORT.md>

Two readings of that work we do not adopt. It evaluates the **first** committed harness, whose
key-question recall we had already documented as flat; by the final harness recall rises from 57.6
to 66.4 on the unseen split. And the unchosen candidate `b2r0g1` reaches higher F1 than the first
generation with **half** the tokens and +10.5 points of recall, which a pure length story does not
explain. Its own data also shows the evolved dialogue doubling the count of answers both reviewers
scored fully correct (15/65 against 7/65) while the ordinal mean moves the other way.
