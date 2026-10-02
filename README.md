# Self-Evolving Questioning Policies
### for active reasoning of LLM agents

MAS.S62 *Self-Evolving AI* (Fall 2026) course project.
**Zining Liu** (ziningl@mit.edu) — Massachusetts Institute of Technology

## The question

A question that buys nothing still costs a turn. Active reasoning asks an agent to acquire the
information a problem withholds, one question at a time, and its dominant failure is not a wrong
final answer but a stalled one: the agent keeps interacting while its questions stop narrowing
what the answer could be.

The strongest current remedy, **T3** ([arXiv:2510.12264](https://arxiv.org/abs/2510.12264),
ICLR 2026), detects that stall and truncates the training trajectory. Three things about it
open this project:

1. The fix lives **inside RL**. At inference an agent cannot discard the episode it is in —
   detecting a stall is only useful paired with something to do next.
2. The criterion is stated task-agnostically but **instantiated by hand** per task. The
   patience `k` takes the values 1, 3, 5 and 2 across the four reported tasks. Hand-fitting a
   criterion per task is exactly the labour a self-improving system should absorb.
3. On preference estimation the refinement proxy reads the **ground-truth** preference vector,
   which no deployed agent has.

So: can the signal be computed at inference, and can the rule that reads it be improved by the
agent itself, with no gradient update?

## Why the main arena makes this exact

Movie Recommendation from **Multi-Turn Puzzles**
([arXiv:2508.10142](https://arxiv.org/abs/2508.10142), Google DeepMind) is, as far as we found,
the only released task where the refinement signal is exactly computable *without* the ground
truth. The user's utility is linear over `k=8` observable attributes; the agent sees the
attribute scores of 20 candidate films and may ask 10 questions of one admissible form,
*would you prefer A over B*; answers are yes / no / no preference.

Each answer is therefore a halfspace constraint `w·(a_A − a_B) > 0` whose coefficients the
agent already knows. The consistent set `H_t` is an exactly characterised region of weight
space and its contraction needs no hidden information.

Measured on the 1000 released episodes: weights lie strictly on the grid {0.0 … 0.9}, so
`|H_0| = 10^8`; the action space is `C(20,2) = 190` questions; ties ("no preference") occur in
**0.57%** of all pairs. Ten binary questions can remove at most a factor of `2^10`, so the task
is underdetermined by design — which is why the benchmark scores the *rank* of the recommended
film rather than recovery of the weights.

## One idea, two layers

Both layers act on one artifact: a small set of written rules, `When [condition], [action]`, deciding
whether the next turn is spent on another question, which question, and what to do once
questioning has stopped paying. Rules are proposed, scored by accuracy bought per question
asked, and kept or retired. No model weights change.

| | |
|---|---|
| **Task-specific layer** | Derive a task's instantiation from outcomes — what counts as the hypothesis set, how refinement is measured, how many flat steps to tolerate, what to do on a stall |
| **General layer** | A mapping from a task's signature to a starting instantiation, plus an exact error decomposition: *did not ask well* vs *had the answer and did not use it* |

The decomposition is the point. Given the constraints collected in an episode, the
best-supported recommendation is computable. The gap to the agent's choice is error from not
using the information; the gap from there to the optimum is error from not having asked well.
That split is what belief deviation means, and here it is measured rather than estimated.

## Layout

```
envs/         Movie Recommendation and Circuit Decoding environments + deterministic scorers
signal/       hypothesis-set tracking, contraction, the stall indicator
policy/       rule schema, propose/score/retire loop, task signature -> instantiation
baselines/    T3's published settings, one global setting, AR-Bench's six strategies,
              and the exact greedy-contraction policy as an upper reference
experiments/  run configs and logs, one directory per dated run
```

## Data

- **Multi-Turn Puzzles** — [huggingface.co/datasets/arianhosseini/mt_puzzles](https://huggingface.co/datasets/arianhosseini/mt_puzzles).
  Data and prompts only; no environment or scorer is released, so both are ours to build. We
  validate them by reproducing the frontier-model numbers the paper reports before any of our
  own runs.
- **AR-Bench** — [tmlr-group/AR-Bench](https://github.com/tmlr-group/AR-Bench) (ICML 2025), the
  transfer domain. Ships the full harness and six questioning strategies. Note its
  `arbench/utils/inference.py` returns a placeholder string on API failure, so a transient
  error is indistinguishable from a wrong answer; since rules are scored by outcome, that path
  needs explicit retry and failure accounting before any evolution run.

## Status

Proposal stage (October 2026). Code lands from the week of 6 Oct: build and validate the Movie
Recommendation environment and scorer, implement exact `H_t` tracking and the greedy reference,
and freeze the rule schema before either loop is written.
