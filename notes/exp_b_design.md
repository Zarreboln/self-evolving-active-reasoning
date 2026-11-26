# Experiment B: design (settings fixed 2026-11-26)

**These settings (tasks, splits, rounds, selection, test protocol) apply to every experiment in this project:
baselines, our method and all ablations. Changing them needs the user's approval.**

## Question
Can a harness learned on a few tasks be shared with new tasks, including tasks from benchmarks never
seen during learning, so that they adapt faster, at lower cost and with less overfitting, when the
sharing is decided by the data (a soft hierarchy) rather than fixed by hand (one harness for all tasks,
or one per benchmark)?

## Tasks (14)
All start from each benchmark's default agent (results: `docs/official_baselines.md`).

**Development (7)**
| Group | Tasks |
|---|---|
| Word games | Wordle basic5, Wordle basic7, Hangman basic |
| Acting in an environment | BabyAI GoToLocal, BabyAI Open, ScienceWorld test-conductivity |
| Asking questions | DC (AR-Bench) |

Two benchmarks (Wordle, BabyAI) have two subtasks each. Without them, "one harness per benchmark" would be
the same as "one harness per task".

**Test time, benchmark never seen (4)**
- Mastermind basic
- ALFWorld pick_and_place_simple
- 20Q basic, with the three themes (people, places, things) pooled
- Minesweeper basic

**Test time, benchmark seen but subtask new (3)**
- Wordle hard5
- Hangman hard
- ScienceWorld grow-plant

Changed on 2026-11-26 from the earlier 20-task set:
- Wordle basic7 and BabyAI Open moved to development.
- Dropped:
  - Mastermind hard and extreme;
  - Minesweeper medium and hard (official scores 0.07, at the floor);
  - ALFWorld pick_heat;
  - 20Q places and people as separate tasks.

## Method
See `notes/method_decomposition.md`. In short, the method is a soft general-to-specific tree:
- **Structure.** One shared code base, and one configuration per node.
- **Weights.** Taken from the distance of a task vector to the node centres (HSML, σ = 1).
- **Depth.** At most 4 levels, decided by the data.
- **At deployment.** Thompson sampling over leaves for the first ~10 problems, then the configuration mixed
  by weight.

## Phase 1: before deployment (development tasks, scores available)
| Item | Setting |
|---|---|
| Rounds | 8, one candidate per round |
| Search split | 20 problems per task, the same problems every round; the code-editing model sees their traces |
| First version | its search scores are the official baseline's search runs (20 episodes per task) |
| Selection split | 20 problems per task (Wordle basic7: 15, its whole selection split); never shown to the code-editing model |
| Choosing the final version | the 3 versions with the highest search score (the first version included) are run on the selection split; the best one is deployed |
| Early stop | stop if 3 rounds in a row bring no gain beyond noise; the saved budget is not given to other methods |
| Equal budget | every method evaluates 7 × 20 = 140 episodes per round |

Only methods that produce several candidates use the selection step: Meta-Harness (both variants), ours and
its ablations. ACE and Harness² do not produce candidates to choose from.

Noise test (paired):
- Two versions are run on the same problems with the same seeds.
- Start with 20 problems and decide at once if the difference is clear. Otherwise add problems up to 50.
- The decision uses a one-sided sign test at p < 0.1.

## Phase 2: deployment / test time (new tasks, environment feedback only)
| Item | Setting |
|---|---|
| Episodes per problem | 1 |
| Problems per task | 50 (ALFWorld 24 and ScienceWorld grow-plant 33, their whole official test sets); 307 in all |
| Order | one fixed random order of the problems, the same for every method |
| What a method sees | only what the environment shows during play (e.g. "Congratulations", hitting a mine, ScienceWorld's displayed score) |

Scores are averaged per benchmark first, then over benchmarks. Unseen and seen benchmarks are reported
separately.

## Compared methods
| Method | Before deployment | At test time |
|---|---|---|
| Official default agent | none | none (existing test results reused) |
| Meta-Harness, one harness per benchmark | evolve | fixed |
| Meta-Harness, one harness for all tasks | evolve | fixed |
| Learning Meta-Skills | build the skill bank | write a harness per problem |
| ACE | offline playbook | update the playbook online |
| Harness² | run its stream over the development tasks (no scores) | its own loop, limited to 1 episode per problem |
| Ours | soft tree | as in "Method" above |

**Ablations (2)**
- **Hard tree (TreeMAML).** Each task follows exactly one path.
- **No levels.** Soft weights only, with no tree.

"One harness for all tasks" and "one per task / per benchmark" are covered by the two Meta-Harness variants.

## Budget (estimate)
| Item | Episodes | GPU hours |
|---|---|---|
| Per method: search 8 × 140 + test 307 | about 1,430 | about 5.4 |
| Per method with a selection step: + 7 × 20 × 3 | + 420 | + 1.6 |
| All 8 methods (official agent reused) | | about 51 |

- **Code-editing model.** At most 6 methods × 8 rounds = 48 Claude Code sessions.
- **Distinct problems.**
  - Development: 7 × 40 = 280 (Wordle basic7: 35).
  - Test: 307.

## Rules for all methods
- **Shared stores.** No game or benchmark names and no answers to specific problems in any shared store
  (component library, skill bank, playbook, experience library).
  - Answer copying is detected by comparing how many seen-split answers a store contains against unseen-split
    answers.
- **Failures.**
  - A failure caused by the agent (runaway output, wrong format) counts as a failure.
  - Problems with servers, the network or third-party services are rerun.
- **What the code-editing model sees.** What the playing model saw, including answers the environment
  reveals. It never sees the selection or test splits, or the environment's hidden state.

## What is reported
1. Average score on the new tasks' test problems, with unseen and seen benchmarks separately.
2. Score along the test stream (problem 1, 2, ...), which shows how fast each method adapts at test time.
3. Cost:
   - GPU hours and API spend, before and during deployment;
   - the number of new tasks after which our extra cost before deployment is paid back.
4. *(Open: needs a test-split run on the development tasks, not in the budget above.)* Change against the
   official default agent on each development task (negative transfer).
5. *(Open, same as 4.)* Search-versus-test gap on the development tasks.
