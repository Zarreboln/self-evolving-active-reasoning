# Dialogues — Situation Puzzles

Readable per-case transcripts from the frozen evaluation (100 test puzzles). Each file shows the same puzzle solved by the bare loop, the first committed harness and the final one, with the referee's answer to every question, the final answer, and that episode's F1, key-question recall and output tokens.

The full records for all six harnesses × 100 puzzles are not committed (they carry the gold bottoms); they are in the release asset `sp_tthe_pilot_raw.tar.zst` under `logs/eval_sp100/results/*.jsonl`.

| case | why it is here |
|---|---|
| [5](case005.md) | 进化版提问质量提升最大 |
| [47](case047.md) | 提问质量提升,次大 |
| [36](case036.md) | F1 最高但关键问题一个没问到 — Sun 批评的那一类 |
| [84](case084.md) | bare 反超最多 |
