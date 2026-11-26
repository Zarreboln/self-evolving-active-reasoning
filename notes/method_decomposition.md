# Method: a soft general-to-specific tree of harnesses (decided 2026-11-26, draft)

Main figure: `~/Desktop/soft_hierarchy_fig/soft_hierarchy.png`.
Earlier readings are kept for reference only: the ability decomposition (`decomposition_tree.png`, and this
file before commit 13695a9) and the atomic-ability tree (`soft_hierarchy_atomic.png`).

## Structure
The tree follows TreeMAML: each level is one more step of improvement on a smaller group of tasks.
- **Root h.** Improved on the pooled failures of all tasks, so every task can gain from it.
- **Inner nodes (h1, h2, ...).** Each starts from its parent and is improved further on the failures of its own
  group only.
- **Leaves (h1.1, ..., h2.3).** The most specific versions. Lower nodes fit fewer tasks, and fit them better.
- **Storage.** A node is stored as its parent plus its own changes (a diff). A change to a parent therefore
  reaches every node below it. If a child's diff no longer applies, the code-editing model merges it.
- **Shape.** A node can have any number of children, and branches can have different depths.

## What is soft
- Each task t has a weight w(t, n) on every node n, and the weights over the children of one node sum to 1.
  This is the same soft assignment as HSML, one level at a time.
- **Editing.** A node is improved on the failures of the tasks below it. Each task's failures count in
  proportion to its weight on that node, so the code-editing model sees more traces from tasks that weigh it
  more.
- **Running.** A task runs the leaf it weighs most. Code cannot run half of two harnesses.
- **Hard version (TreeMAML).** The same tree, but every weight is 0 or 1, so each task follows exactly one
  path.

## One round before deployment (scores available)
1. **Scores.** Run every task on the leaves it weighs most and record the scores on the search split.
2. **Weights.** w(t, child) = softmax over children of gain(t, child) / τ, as in DPM. Here gain is the task's
   score on that child's subtree minus its score on the parent.
3. **Improve, top-down.**
   - The root is edited first, on the failures of all tasks. The edit is kept only if the mean score over all
     tasks improves beyond noise.
   - Each child is then edited on its tasks' weighted failures. Its edit is kept if the weighted score of its
     tasks improves beyond noise.
   - Leaves are edited last.
4. **Reshape.**
   - **Split.** If an edit to a node helps some of its tasks and hurts others beyond noise, the node splits:
     the edit goes to a new child for the tasks it helps, and the others keep the node as it was.
   - **Promote.** If an edit made in one child also helps the tasks of its siblings beyond noise, it moves up
     to the parent.
   - **Prune.** Children with little total weight are removed.

**Cost per round.** About one call to the code-editing model per node edited (root + inner nodes + leaves),
plus the evaluations. This matches the faithful harness version of TreeMAML.

## Relation to TreeMAML and HSML
- **TreeMAML.** The k-th inner step averages gradients within the level-k cluster. Here the k-th level is
  edited on the pooled failures of its cluster.
- **Differences from TreeMAML.**
  - Nodes are kept between rounds. TreeMAML has one shared initialisation and rebuilds its clusters for each
    batch.
  - Membership is soft.
  - The shape of the tree grows and shrinks.
- **HSML.** Like HSML, assignment is soft, level by level. Unlike HSML, each node holds its own code instead of
  a gate over one shared set of parameters.

## A new task at deployment (environment feedback only, no scores)
1. Its first weights are borrowed from the most similar development tasks. It runs the leaf it weighs most.
2. While it plays, its weights move toward the branches where it wins more. It may switch leaves.
3. If no leaf fits (the weights stay flat, or every candidate keeps losing), the code-editing model grows a new
   leaf for it. The leaf starts from the highest-weight node, or from the root if the weights are flat, and is
   written from the task's environment-visible traces only. Shared nodes are never edited at deployment.
4. Between deployments, with scores on the development tasks' search split, the new leaf's changes can be
   promoted upward by step 4 above. Old tasks change only if the change is re-tested on them with no drop
   beyond noise.

## Ablations
- Hard tree (TreeMAML: one path per task).
- One harness for all tasks (root only).
- One harness per task (leaves only, no sharing).
- A tree fixed by hand (one branch per benchmark).
- No reshaping: the shape of the tree is fixed after the first round.

## Open (to decide)
- Whether weights come from gain (DPM style) or from a task embedding and distance to node centres (HSML
  style), and the value of τ.
- The noise threshold used in the keep, split and promote tests (number of problems per test).
- How the seven slot families of Learning Meta-Skills map onto what a node may change.
