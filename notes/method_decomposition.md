# Method: abilities decomposed into shared sub-abilities (chosen 2026-11-25, draft)

Replaces the specialization tree of `method_formulas.md` §8 as the main method. The specialization tree
(TreeMAML-style, general at the root, specific at the leaves) stays as a figure and as a comparison.
Figure: `~/Desktop/soft_hierarchy_fig/decomposition_tree.png`.

## Structure (main figure: `~/Desktop/soft_hierarchy_fig/soft_hierarchy.png`, decided 2026-11-26)
- The harness h splits into abilities (h1, h2), which split into **atomic abilities** (h1.1, h1.2, h2.1,
  h2.2, h2.3). Top = composite, bottom = atomic. A node can have any number of children.
- **Tasks attach only to atomic abilities**, never directly to h1/h2: a task draws on several atomic
  abilities, with weights, across branches (e.g. one task uses h1.2 strongly and h2.1 weakly).
- An atomic ability serves every task that weighs it.
- Hard version (TreeMAML): the same tree, but each task follows exactly one path to one atomic ability.
- Earlier figure with tasks on top as callers (`decomposition_tree.png`) is kept but is no longer the main
  picture.

## What is soft: splits into smaller versions
When two tasks want a shared ability changed in conflicting ways (the change helps one and hurts the other
beyond noise), the ability is not overwritten and not copied. It splits: the common part stays in the
ability, and the differing part becomes smaller versions below it. Example: "prune by feedback" keeps
"read which candidates the feedback rules out"; below it, "strict" deletes them (task A, exact feedback) and
"tolerant" down-weights them (task B, answers that are sometimes wrong).
Each task has weights over the smaller versions of a split (e.g. A: strict 0.9, tolerant 0.1) and runs the
one it weighs most (code cannot run half of two functions). Weights come from measured gain through a
softmax with temperature τ, as in DPM. They also decide where a change is credited and on which callers it
is tested, and a new task borrows them from similar tasks. Without conflicts there is only one version and
no weight.

## How sharing grows (library learning)
1. A change is first made inside one task's ability (the code-editing model, Claude Code, sees only that
   task's traces).
2. If the same change also helps other tasks beyond noise (in at least two benchmarks), it is pulled out into
   a sub-ability below them and both call it.
3. If it keeps helping more tasks, it sinks further toward the primitives.
4. Before a shared sub-ability is changed, it is re-tested on its callers; if a caller drops beyond noise, the
   sub-ability splits as above instead of being overwritten (the replay test follows OSML).
Closest published precedent: library learning in program synthesis (DreamCoder, Ellis et al., 2021; LILO,
Grand et al., 2024) and compositional skill libraries (Voyager, Wang et al., 2023). To be checked against the
papers before any formula is written down.

## Gain (credit)
gain(caller, callee) = score change of the caller's task when the callee is swapped out (for the next-best
alternative) or removed. Measured on the search split before deployment.

## A new task: how the hierarchy is updated
At deployment (no scores, only the environment's wins and losses):
1. A top-level node is created for the task. Its first ability is composed of existing sub-abilities chosen
   by similarity; at each split it borrows the weights of similar tasks.
2. While playing, it adjusts its weights at each split from wins and losses.
3. If that is not enough, the code-editing model edits only this task's own top-level code. Shared abilities
   are never edited at deployment.
Between deployments (scores on the development tasks' search split are available):
4. New code in the task's ability that also helps other tasks is pulled out and sinks down; if it does the
   same job as an existing ability, it becomes a smaller version under that ability.
5. If the task needs a shared ability changed in a way that hurts its other callers, the ability splits.
6. Smaller versions that no task weighs much are removed, so the library does not keep growing.
Old tasks change only when a change has been re-tested on them with no drop beyond noise.

Previous summary of this step:
Composes existing sub-abilities, weighted by similarity to the tasks that use them; adjusts the weights from
the wins and losses the environment reports. If nothing fits (flat weights, or every candidate keeps losing),
the code-editing model writes a new sub-ability from the task's environment-visible traces, built on the
existing primitives; it is kept only if the following problems improve. Shared nodes are not edited at test
time.

## Open (to decide)
- Exact softmax and gain formulas, once the precedent papers are read.
- How the seven slot families of Learning Meta-Skills map onto abilities (slots may become the fixed
  interfaces of primitives).
- Ablations: hard composition (each role filled by exactly one callee), flat library (no levels), one shared
  harness, and the specialization tree as a comparison.
