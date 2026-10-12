# 07 — Situation Puzzles, bare questioner with the official-grade judge (2026-10-10 21:29–21:36)

**Setup**: AR-Bench SP test set, 100 instances × 2 seeds, 25 turns, `runner/sp_runner_async.py`
with two endpoints: questioner Qwen2.5-7B-Instruct, respondent + keypoint judge
Llama-3.1-405B-Instruct AWQ-INT4 (calibrated in 02). Metrics: character / word F1 of the final
answer against the hidden story (AR-Bench's).

| condition | n | err% | f1_char | f1_word | respondent answers (no / yes / unknown) |
|---|---|---|---|---|---|
| bare, 405B-INT4 judge | 200 | 0.0 | **0.621** | 0.244 | 48% / 29% / 23% |
| bare, 7B as its own judge (2026-10-07, superseded) | 100 | — | 0.659 | 0.242 | almost never "no" |

`summary.txt` holds the summarizer output; raw episodes: release asset `sp405_traces.tar.zst`.

**Takeaway**: all earlier SP numbers were built on a respondent that could not say "no"; from
here on SP uses the 405B-INT4 judge. The earlier "belief_injected" SP condition read the hidden
story and keypoints and is retired; a transcript-only version is future work.
