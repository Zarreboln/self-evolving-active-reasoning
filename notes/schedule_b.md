# Experiment B schedule (written 2026-11-25)

Rules: every GPU job lowprio on ml-16-node-001, at most 4 GPUs at once, resumable after preemption.
Baselines run their own released code; we only add the glue that points them at our model and splits.
A monitor watches every job; any anomaly stops the work and goes to the user.

Model setting: Qwen3.5-9B with thinking off + each task's official reasoning agent. Only this setting is run
(user, 2026-11-25: setting 2, thinking on + the non-reasoning agent, is dropped).

| step | what | my coding | GPU (setting 1) | planned dates |
|---|---|---|---|---|
| 0 | adapters follow the original protocols (done, 54ae3a6) | done | - | 11-25 |
| 1a | official starting points, AR-Bench (GN, DC): its own evaluators, test split | 11-25 | ~1-2 h | 11-25/26 |
| 1b | TextArena (Wordle, Mastermind, Hangman, 20Q): official agents + wrappers | 11-26 | ~1 h | 11-26 |
| 1c | BabyAI (BALROG agents), ALFWorld (ReAct), ScienceWorld (SwiftSage), Blicket (BALROG pair) | 11-26/27 | ~2 h | 11-27 |
| 2 | seeds for evolution built from the official code + equivalence check against step 1 | 11-27 | ~1 h | 11-28 |
| 3 | Meta-Harness official loop: per task (no sharing) and all 10 tasks as one setting (global) | 11-28 | ~15 h | 11-28 - 11-30 |
| 4 | Harness² (google-research/harness2) | 11-29 - 11-30 | ~15 h | 11-30 - 12-02 |
| 5 | Learning Meta-Skills (released part + paper for the rest) | 11-30 - 12-01 | ~15 h | 12-01 - 12-03 |
| 6 | ACE (ace-agent/ace) | 12-01 | ~10-15 h | 12-02 - 12-04 |
| 7 | our method, equal budget | 12-02 - 12-05 | matched | 12-05 - 12-08 |

Before steps 4-6: they call the Anthropic API directly (not Claude Code), so they need a pay-as-you-go API key;
budget to be confirmed with the user with measured numbers.
