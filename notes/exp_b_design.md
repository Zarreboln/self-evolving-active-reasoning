# Experiment B: design (agreed 2026-11-26)

## Question
Can a harness learned on a few tasks be shared with new tasks, including tasks from benchmarks never
seen during learning, so that they adapt faster, at lower cost and with less overfitting, when the
sharing is decided by the data (a soft hierarchy) rather than fixed by hand (one harness for all tasks,
or one per benchmark)?

## Tasks (20)
All use each benchmark's default agent as the starting point (results: `docs/official_baselines.md`).

**Development (5)**: Wordle basic5, Hangman basic, DC (AR-Bench), BabyAI GoToLocal,
ScienceWorld test-conductivity. Two word games, one questioning task, two tasks that act in an environment.

**Test time, benchmark never seen (10)**: Mastermind basic, hard, extreme; Minesweeper basic, medium, hard;
ALFWorld pick_and_place_simple, pick_heat_then_place_in_recep; 20Q basic-places, basic-people.

**Test time, benchmark seen but subtask new (5)**: Wordle hard5, Wordle basic7, Hangman hard, BabyAI Open,
ScienceWorld grow-plant.

Results on unseen benchmarks are averaged per benchmark first, so a benchmark with more subtasks does not
count more.

## Method
The harness is split into a fixed set of component types (following the seven families of Learning
Meta-Skills, 2609.38143). Components are organised as a **soft tree** with three levels, following the
tree of TreeMAML (Garcia et al., 2021) and the per-level soft assignment of HSML (Yao et al., 2019):

- **Root**: components every task uses.
- **Group nodes**: components shared by a subset of tasks. A task attaches to group nodes with weights
  γ(t, g) = softmax over groups of gain(t, g) / τ, and can attach to several groups at once. This is the
  difference from TreeMAML, where each task follows exactly one path.
- **Leaves**: each task's own components.

A table records, for every component and task, how much the score on the task changes when the component
is added (the gain table). Each round alternates two steps, following the E/M structure of DPM
(Jerfel et al., 2019):

- **E step**: recompute γ from the gain table. τ → ∞ gives one shared harness; τ → 0 gives a hard tree.
- **M step**: pick a node; the model that edits code (Claude Code) reads failures from the tasks with high
  weight on that node and makes one change to it; the change is evaluated mainly on those tasks, and the
  score changes are written back. Before a shared component is changed, it is re-tested on the other tasks
  that use it; if a task drops beyond noise, the component is forked instead of overwritten (OSML).

**Where a change lives is decided by the data.** A change starts at the leaf of the task it was made for.
If it also helps other tasks beyond noise (in at least two benchmarks), it moves up to a group node; if it
helps all groups without hurting any, it moves to the root. A change that helps a set of tasks no existing
group matches opens a new group (DPM spawn rule).

A new task mounts the root, attaches to group nodes by its similarity to the development tasks (task
instructions, action format, first episode; HSML-style soft assignment), starts with an empty leaf, and
then updates its weights from the wins and losses the environment reports during play.

## Two phases for every method
1. **Before deployment** (development tasks): scores on the search split are available. Budget: the same
   total number of evaluated episodes for every method (5 episodes per task per round, one candidate per
   round, 8 rounds). Final harnesses are chosen on the selection split, not the search split.
2. **Deployment / test time** (new tasks): no scores; a method sees only what the environment says during
   play (e.g. "Congratulations", hitting a mine, ScienceWorld's displayed score). Each test problem gets at
   most 3 episodes. 50 test problems per new task.

## Compared methods
| Method | Before deployment | At test time |
|---|---|---|
| Official default agent | none | none |
| Meta-Harness, one harness per benchmark | evolve | fixed |
| Meta-Harness, one harness for all tasks | evolve | fixed |
| Learning Meta-Skills | build the skill bank | write a harness per problem |
| ACE | offline playbook | update the playbook online (online-only also reported) |
| Harness² | run its stream over the development tasks (no scores) | its own loop (empty-library version also reported) |
| Ours | soft tree of components + γ | root + γ-weighted groups + empty leaf; γ updated from environment feedback |

**Ablations of our method** (same code, only the membership rule changes): hard tree fixed by hand
(by benchmark or task type; fixed TreeMAML), hard tree learned by clustering rows of the gain table
(learned TreeMAML), 0/1 overlapping membership, a flat version with no levels (soft weights only), and one
shared harness (τ → ∞).

## Rules for all methods
- No game or benchmark names and no answers to specific problems in any shared store (component library,
  skill bank, playbook, experience library). Answer copying is detected by comparing how many seen-split
  answers a store contains against unseen-split answers.
- A failure caused by the agent (runaway output, wrong format) counts as a failure; problems with servers,
  the network or third-party services are rerun.
- The model that edits code sees what the playing model saw (including answers the environment reveals),
  never the selection or test splits or the environment's hidden state.

## What is reported
1. Average score on the new tasks' test problems (unseen benchmarks and seen benchmarks separately).
2. Score along the test stream (problem 1, 2, ...): how fast each method adapts at test time.
3. Cost (GPU hours, API spend) before and during deployment, and the number of new tasks after which our
   extra cost before deployment is paid back.
4. Change against the official default agent on each development task (negative transfer).
5. Search-versus-test gap on the development tasks.
