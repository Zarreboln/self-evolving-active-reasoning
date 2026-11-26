# Method: a soft general-to-specific tree of harnesses (decided 2026-11-26, draft)

Main figure: `~/Desktop/soft_hierarchy_fig/soft_hierarchy.png`.
Earlier readings are kept for reference only: the ability decomposition (`decomposition_tree.png`, and this
file before commit 13695a9) and the atomic-ability tree (`soft_hierarchy_atomic.png`).

## Structure: one shared code base, one configuration per node (decided 2026-11-26)
The tree follows TreeMAML: each level is one more step of improvement on a smaller group of tasks. As in HSML,
there is one set of shared parameters, and each node only gates it.
- **Shared code base (HSML's θ0).** One harness made of modules, for example pruning candidates, compressing
  the history, checking the action format and reflecting on failures. Every code edit happens here.
- **Module interfaces.** The seven slot families of Learning Meta-Skills serve as the fixed interfaces of the
  modules.
- **Node = configuration (HSML's gate).** A node stores:
  - which modules are on;
  - parameter values, such as the number of candidates, the number of retries and the temperature;
  - which prompt snippets are added.
- **Inheritance.** A child inherits its parent's configuration and overrides only a few entries.
  - Root: the configuration fitted to the failures of all tasks.
  - Inner nodes and leaves: overrides fitted to their own group, so lower nodes are more specific.
- **Shape.** A node can have any number of children, and branches can have different depths.
- **Precedent for gating.** HSML ablation A4c: gating one shared initialisation beats generating a separate
  initialisation per cluster by about 4 points.

What this gives over storing a code diff per node:
- A code fix reaches every task that has the module on, and there are no diffs to merge.
- A configuration can be mixed by weight (see below), so a task no longer has to run a single leaf.
- A new leaf is usually only a new configuration, so growing one at deployment cannot break shared code.
- Promotion moves a configuration value up to the parent. Moving it down to a child is the reverse.

The cost is that the code-editing model must write general modules with switches, not patches for one task.

## What is soft
- **Weights.** Each task t has a weight w(t, n) on every node n, as in HSML:
  - **Task vector e(t).**
    - At first, e(t) is computed from the task description only.
    - After 3 episodes run with the root configuration, e(t) is recomputed from the description together
      with a summary of these traces. Only environment-visible content is used, so this stays label-free.
    - The same fixed local text-embedding model is used for development and new tasks.
  - **Centres.** Each node n has a centre c(n). The initial tree and centres come from hierarchical
    clustering of the development tasks' vectors.
  - **Weights per level (HSML, Eq. 6).** w(t, child) ∝ exp(−‖(e(t) − c(child)) / σ‖² / 2), normalised over
    the children of one node.
    - Distances are first divided by the median pairwise distance between development tasks.
    - σ = 1 is a fixed constant, not tuned. HSML also fixes σ, to 10 or 2 depending on the experiment.
  - **Weight on a leaf.** The product of the weights along its path.
- **Running.** A task runs a configuration mixed from the leaves by these weights:
  - continuous values are averaged by weight;
  - on/off switches and prompt snippets are on if their weighted vote is above one half.
- **Editing.** A node's configuration is fitted to the failures of the tasks below it, each counted in
  proportion to its weight.
- **Hard version (TreeMAML).** The same tree, but every weight is 0 or 1, so each task follows exactly one
  path and runs one leaf.

## One round before deployment (scores available)
1. **Scores.** Run every task with its mixed configuration and record the scores on the search split.
2. **Code.** The code-editing model edits the shared code base on the failures of all tasks. A change is kept
   only if no task drops beyond noise and the mean improves beyond noise.
3. **Configurations, top-down.** The root's configuration is fitted first, then each child's overrides on its
   tasks' weighted failures, then the leaves.
4. **Centres.** Each node's centre is set to the weighted mean of its tasks' vectors. A task's weight here is
   its weight on the node times its positive gain from the node (soft k-means; HSML learns centres by
   gradient, which is not available here).
5. **Reshape.**
   - **Split.** If a configuration change helps some tasks of a node and hurts others beyond noise, the node
     splits: the change goes to a new child for the tasks it helps.
   - **Promote.** If a child's override also helps the tasks of its siblings beyond noise, it moves up to the
     parent.
   - **Prune.** Children with little total weight are removed.

**Noise test (used by keep, split and promote).**
- Old and new versions run on the same problems with the same seeds.
- Count the problems that only the new version solves and those that only the old version solves.
- Start with 20 problems and decide at once if the difference is clear. Otherwise add problems up to 50 and
  use a one-sided sign test at p < 0.1.

## Relation to TreeMAML and HSML
- **TreeMAML.** The k-th inner step averages gradients within the level-k cluster. Here the level-k
  configuration is fitted to the pooled failures of its cluster.
- **HSML.** Shared parameters plus a gate per cluster. Soft assignment by distance to cluster centres, one
  level at a time.
- **Differences from both.**
  - What is shared is code, not network weights.
  - Nodes are kept between rounds.
  - The shape of the tree grows and shrinks.

## A new task at deployment (environment feedback only, no scores)
1. **Start.** Its vector is computed as above (description first, then description and 3 traces). Its leaf
   weights come from the distances to the node centres.
2. **Choose by Thompson sampling.**
   - For about the first 10 episodes, the task runs single leaves, not the mix, among the 3 leaves it weighs
     most.
   - Each leaf has a Beta distribution over its win rate. The prior comes from the distance weights.
   - Each episode samples one value per leaf and runs the leaf with the largest value.
   - A win adds 1 to that leaf's α, and a loss adds 1 to its β.
   - Afterwards, the posterior means become the weights, and the task runs the mixed configuration.
3. **Grow.** If nothing fits (the posteriors of all 3 leaves stay low), a new leaf is grown under the
   highest-weight node, or under the root if the weights are flat.
   - The new leaf is a new configuration, written from the task's environment-visible traces only.
   - Shared code and shared nodes are never edited at deployment.
4. **Between deployments.** Scores on the development tasks' search split are available. The new leaf's
   overrides can be promoted, and code changes it needs go through step 2 above.

## Ablations
- Hard tree (TreeMAML: one path per task).
- One configuration for all tasks (root only).
- One harness per task (leaves only, no sharing).
- A tree fixed by hand (one branch per benchmark).
- No reshaping: the shape of the tree is fixed after the first round.
- A code diff per node instead of a configuration per node.

## Open (to decide)
- Whether to enlarge the development set (5 tasks may be too few for a tree). This changes the experiment
  design, so it is not decided yet.
- How the Beta prior is built from the distance weights, and the exact threshold for "posteriors stay low".
- Which embedding model.
