# Official Default Baselines

Updated 2026-11-26. Model: Qwen3.5-9B with thinking turned off (vLLM, `enable_thinking: false`).
Every task is run with **its own benchmark's default agent**, unchanged. We only added adapter code so
that it calls our model and uses our data splits; every adaptation is described at the top of the
corresponding script in `multibench/mb/official/`.

## Test results (9 tasks)

| Task | Default agent | Test set | Episodes | Score | Episodes counted as failures |
|---|---|---|---|---|---|
| GN (guess the number) | AR-Bench `zero_shot` | official test set | 100 | **0.02** | — |
| DC (detective cases) | AR-Bench `zero_shot` | official test set | 100 | **0.45** | — |
| Wordle basic5 | TextArena standard agent | our split | 60 | **0.43** | — |
| Mastermind basic | TextArena standard agent | our split | 142 | **0.57** | — |
| Hangman basic | TextArena standard agent | our split | 200 | **0.22** | 50: the model's output ran on until it exceeded the context length |
| 20Q basic-things | TextArena standard agent (Jev as gamemaster) | our split | 49 | **0.06** | 12: the Jev service returned errors; **to be rerun** (see below) |
| BabyAI GoToLocal | BALROG `naive` | our split (fixed seeds) | 200 | **0.82** | — |
| ALFWorld pick_and_place_simple | ReAct | official valid_unseen | 24 | **0.58** | — |
| ScienceWorld test-conductivity | ReAct baseline from SwiftSage | official test set | 225 | **0.33** | — |

Scores are each benchmark's official metric, scaled to 0–1.

## Notes

- **Data splits**: GN, DC, ALFWorld and ScienceWorld use their official test sets. TextArena and BabyAI
  have no official split, so we split by a hash of the answer or by fixed random seeds.
- **How failures are counted**: a failure caused by the agent itself (for example, runaway output or a
  wrong format) counts as a failure. Problems with servers, the network or a third-party service are rerun.
- **The 12 episodes of 20Q**: we first thought Jev rejected requests that were too long. It turned out
  that Jev's service intermittently returns an error (`400 Unknown model`) when it receives many requests
  at once, even for short ones. This is a third-party service problem; the client now waits longer
  between retries, and these episodes will be rerun.
- **The 50 episodes of Hangman**: TextArena's standard agent sets no limit on output length, and the model
  sometimes repeats itself until the 64K context is full. This is a real failure of the agent under its
  default settings, so it counts as a failure.
- **What the adapter code changes**:
  - ALFWorld: the original used a text-completion model; our model is a chat model and repeats the `>` at
    the start of its reply, so it is removed. ReAct's examples use older ALFWorld's `put X in/on Y`, which
    is sent as `move X to Y`, the same command in ALFWorld 0.4.
  - ReAct and SwiftSage: the models they called (text-davinci-002, gpt-3.5) are no longer served, so they
    call our model instead; everything else follows the original.
  - SwiftSage's version without reasoning: the original matches actions with sentence embeddings; we use
    the same word-overlap matching as its reasoning version.
  - 20Q: TextArena's code fixes GPT-4o as the gamemaster; we use Jev.
- **For reference only (not a baseline)**: AR-Bench's reasoning agent `proactive_cot` was stopped partway.
  GN: all 79 finished episodes correct; DC: 10 of 36 finished episodes correct.

## In progress

The official baselines for 42 candidate tasks are running: each task first plays 20 episodes on the search
split, used to choose the tasks for the evolution experiments, and then its full test set, for reporting.
They will be added to this table.
