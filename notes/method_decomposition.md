# Method: abilities decomposed into shared sub-abilities (chosen 2026-11-25, draft)

Replaces the specialization tree of `method_formulas.md` §8 as the main method. The specialization tree
(TreeMAML-style, general at the root, specific at the leaves) stays as a figure and as a comparison.
Figure: `~/Desktop/soft_hierarchy_fig/decomposition_tree.png`.

## Structure
- **Top level**: one ability per task, the task's full harness.
- **Lower levels**: sub-abilities that the abilities above call (e.g. keep a candidate list, prune it by
  feedback, plan then act). The lower an ability sits, the more tasks call it.
- **Bottom**: primitives that most tasks call (e.g. read feedback, check that an action is valid, repair the
  output format).
- Every node is a callable function with a fixed input/output. An edge means "calls".
- The graph need not be balanced: an ability can call a primitive directly, skipping a level.
- Tasks sit at the top (callers), primitives at the bottom (callees), as in a call graph. Because an ability
  can have several callers, this is a layered hierarchy (a DAG), not a strict tree; the paper says
  "hierarchy". Lower abilities tend to be called by more tasks; this is a tendency, not a rule.

## What is soft
A role (e.g. "prune by feedback") can have several versions. Each caller has weights over the versions of a
role it calls and runs the one it weighs most (code cannot run half of two functions). The weights also
decide where a change is credited and on which callers it is tested. Weights come from
measured gain (below) through a softmax with temperature τ, as in DPM.

## How sharing grows (library learning)
1. A change is first made inside one task's ability (the code-editing model, Claude Code, sees only that
   task's traces).
2. If the same change also helps other tasks beyond noise (in at least two benchmarks), it is pulled out into
   a sub-ability below them and both call it.
3. If it keeps helping more tasks, it sinks further toward the primitives.
4. Before a shared sub-ability is changed, it is re-tested on its callers; if a caller drops beyond noise, the
   sub-ability is forked instead of overwritten (OSML).
Closest published precedent: library learning in program synthesis (DreamCoder, Ellis et al., 2021; LILO,
Grand et al., 2024) and compositional skill libraries (Voyager, Wang et al., 2023). To be checked against the
papers before any formula is written down.

## Gain (credit)
gain(caller, callee) = score change of the caller's task when the callee is swapped out (for the next-best
alternative) or removed. Measured on the search split before deployment.

## A new task
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
