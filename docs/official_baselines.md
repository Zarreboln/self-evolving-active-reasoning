# Official Default Baselines

Updated 2026-11-26. Model: Qwen3.5-9B with thinking turned off (vLLM, `enable_thinking: false`).
Every task is run with **its own benchmark's default agent**, unchanged. We only added adapter code so
that it calls our model and uses our data splits; every adaptation is described at the top of the
corresponding script in `multibench/mb/official/`.

## Test results: the 14 tasks of the experiment

The development tasks are where harnesses are evolved. The test tasks are where they are deployed: either
from a benchmark never seen during evolution, or a new subtask of a seen benchmark (see
[notes/exp_b_design.md](../notes/exp_b_design.md)).

| Group | Task | Default agent | Test set | Episodes | Score | Episodes counted as failures |
|---|---|---|---|---|---|---|
| Development | Wordle basic5 | TextArena standard agent | our split | 60 | **0.43** | — |
| Development | Wordle basic7 | TextArena standard agent | our split | 20 | **0.20** | 2: prompt over the context length |
| Development | Hangman basic | TextArena standard agent | our split | 200 | **0.22** | 50: runaway output over the context length |
| Development | BabyAI GoToLocal | BALROG `naive` | our split (fixed seeds) | 200 | **0.82** | — |
| Development | BabyAI Open | BALROG `naive` | our split (fixed seeds) | 200 | **0.75** | — |
| Development | ScienceWorld test-conductivity | ReAct baseline from SwiftSage | official test set | 225 | **0.33** | — |
| Development | DC (detective cases) | AR-Bench `zero_shot` | official test set | 100 | **0.45** | — |
| Unseen benchmark | Mastermind basic | TextArena standard agent | our split | 142 | **0.57** | — |
| Unseen benchmark | ALFWorld pick_and_place_simple | ReAct | official valid_unseen | 24 | **0.58** | — |
| Unseen benchmark | 20Q basic-people | TextArena standard agent (Jev as gamemaster) | our split | 17 | **0.12** | 3: Jev service errors, **to be rerun** |
| Unseen benchmark | 20Q basic-places | TextArena standard agent (Jev as gamemaster) | our split | 32 | **0.06** | 3: Jev service errors, **to be rerun** |
| Unseen benchmark | 20Q basic-things | TextArena standard agent (Jev as gamemaster) | our split | 49 | **0.06** | 12: Jev service errors, **to be rerun** |
| Unseen benchmark | Minesweeper basic | TextArena standard agent | our split | 200 | **0.21** | 1: prompt over the context length |
| Seen benchmark, new subtask | Wordle hard5 | TextArena standard agent | our split | 200 | **0.30** | 1: prompt over the context length |
| Seen benchmark, new subtask | Hangman hard | TextArena standard agent | our split | 200 | **0.17** | 43: runaway output over the context length |
| Seen benchmark, new subtask | ScienceWorld grow-plant | ReAct baseline from SwiftSage | official test set | 33 | **0.21** | — |

In the experiment, 20Q is one test task made of the three themes pooled (50 problems in all). Each test task
uses at most 50 problems (see the design note); the table above shows the full test sets.

### Other tasks measured (candidates, not used in the experiment)

| Task | Default agent | Test set | Episodes | Score |
|---|---|---|---|---|
| GN (guess the number) | AR-Bench `zero_shot` | official test set | 100 | 0.02 |
| BabyAI PickupLoc | BALROG `naive` | our split (fixed seeds) | 200 | 0.47 |
| ALFWorld look_at_obj_in_light | ReAct | official valid_unseen | 18 | 0.94 |
| ALFWorld pick_heat_then_place_in_recep | ReAct | official valid_unseen | 23 | 0.78 |
| Mastermind hard / extreme | TextArena standard agent | our split | — | 0.42 / 0.27 |
| Minesweeper medium / hard | TextArena standard agent | our split | — | 0.07 / 0.07 |

Scores are each benchmark's official metric, scaled to 0–1.

## Notes

- **Data splits**: GN, DC, ALFWorld and ScienceWorld use their official test sets. TextArena and BabyAI
  have no official split, so we split by a hash of the answer or by fixed random seeds.
- **How failures are counted**: a failure caused by the agent itself (for example, runaway output or a
  wrong format) counts as a failure. Problems with servers, the network or a third-party service are rerun.
- **The Jev errors of 20Q** (18 episodes over the three themes): we first thought Jev rejected requests that were too long. It turned out
  that Jev's service intermittently returns an error (`400 Unknown model`) when it receives many requests
  at once, even for short ones. This is a third-party service problem; the client now waits longer
  between retries, and these episodes will be rerun.
- **Context-length failures** (Hangman 50 and 43 episodes, a few in Wordle and Minesweeper): TextArena's
  standard agent sets no limit on output length, and the model sometimes repeats itself until the 64K
  context is full; the server then rejects the next request. This is a real failure of the agent under its
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

- The 18 20Q episodes that failed on Jev errors will be rerun.
- Evolution runs (Meta-Harness and the other methods) have started on ORCD. Their results will go on a
  separate page.
