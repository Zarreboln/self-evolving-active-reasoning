# Soft Hierarchies for Sharing Harnesses Across Tasks

**Zining Liu** — Massachusetts Institute of Technology

The code around a frozen language model (its harness: prompts, how the history is kept, control flow,
tools) can be improved automatically. When existing methods share a harness across several tasks, the
scope of sharing is fixed by hand: either one harness for all tasks, or one per benchmark, a hard
hierarchy. This project learns a **soft hierarchy** instead: the data decides **which parts of a harness
are shared between which tasks**, and a task can belong to several groups to different degrees, so that a
new task, including one from a benchmark not seen before, adapts faster, at lower cost, and with less
overfitting.

## Status

- **Official default baselines**: every task is run with its own benchmark's default agent, unchanged.
  Results: [docs/official_baselines.md](docs/official_baselines.md).
- **Baseline methods** (in progress): Meta-Harness (one harness per benchmark, and one harness for all
  tasks), Harness², ACE, and Learning Meta-Skills, each run from its released code.
- **Our method** (in progress): the harness is split into a fixed set of component types; each task has
  a continuous weight on each component, which decides which components it uses and how credit for a
  change is shared; evolution and sharing alternate. Formulas and their sources:
  [notes/method_formulas.md](notes/method_formulas.md).

## Layout

| Path | Contents |
|---|---|
| `multibench/mb/benches/` | One interface for 17 benchmarks and 101 subtasks (rules and scoring as in each benchmark's official code) |
| `multibench/mb/official/` | Scripts that run each benchmark's default agent (with only the adapter code added) |
| `multibench/mb/evolve/` | Code for harness evolution: candidates run in an isolated subprocess, static checks, a check against copying answers, evaluation and export |
| `sandbox/` | Docker sandbox for the model that edits the code (Claude Code): it sees only its own workspace and can reach only the Anthropic API |
| `notes/` | Pre-registration, experiment results, method formulas, the check of official agents, the schedule, paper notes |
| `meta/arbench_mh/` | The AR-Bench DC and SP environments, imported by the adapters |

## Earlier project

The MAS.S62 active-reasoning project (Meta-Harness on AR-Bench, with Yangzi Sun) is kept in full under
the tag [`mas-s62-ar-report`](../../tree/mas-s62-ar-report).

## Third-party code

See [THIRD_PARTY.md](THIRD_PARTY.md).
