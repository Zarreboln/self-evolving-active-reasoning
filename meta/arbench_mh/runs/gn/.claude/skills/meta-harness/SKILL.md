---
name: meta-harness
description: Run one iteration of guessing harness evolution. Called by meta_harness.py or interactively via /meta-harness.
---

# Meta-Harness (Guessing Harness Evolution)

Run ONE iteration of harness evolution. Do all work in the main session — do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run benchmarks.** You analyze results + guess records, prototype changes, and implement new harnesses. The outer loop (`meta_harness.py`) handles benchmarking separately.

## The task

AR-Bench Guess Number. A hidden four-digit secret; the player makes up to 25 guesses and each is answered with how many digits are in the right place and how many are present but misplaced. The answer space is P(10,4) = 5040.

The model is frozen. Everything a harness can change sits between the model and the oracle.

## CRITICAL CONSTRAINTS

- You MUST implement 3 new harnesses every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 3 candidates per iteration: mix of exploitation and exploration.

### Anti-parameter-tuning rules

The most common failure mode is creating harnesses that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried — parameter sweeps (how many candidate guesses are drafted, how much of the record is echoed, prompt wording) almost always regress or tie.

**Good candidates change a fundamental mechanism:**

- A new way of turning feedback into constraints the next guess must satisfy
- A new state representation (a surviving-candidate set, a per-digit possibility table, a constraint list)
- A new prompt architecture (present the record as constraints rather than as a transcript)
- A new rule for choosing among drafted guesses

**Bad candidates just tune numbers.** If the logic in `solve()` is identical to the base except for constants, it's a parameter variant. Rewrite with a genuinely novel mechanism.

**Combining harnesses is valid.** Take the state representation from harness A and the selection rule from harness B.

Exploitation axes: A=Guess generation, B=State maintained across turns, C=Choosing which guess to play, D=Shape of the record fed back each turn, E=Stopping, F=Extra LLM passes between guesses. If last 3 iterations explored the same axis, pick different ones.

### Anti-overfitting rules

- **No instance-specific hints.** Do not hardcode knowledge about specific secrets. Harnesses must be general-purpose.
- **Never hardcode a particular number** in harness code, prompts, or comments.
- **General patterns are OK.** Rules like "never play a guess inconsistent with earlier feedback" apply broadly.

## The clock

The session is cut off after 40 minutes, and nothing you have not written to disk survives it.

So write the three harness files **and `pending_eval.json`** as soon as the three mechanisms are
decided, before any further prototyping or polishing. The outer loop reads the files from disk
after the session ends, so whatever you improve afterwards is still picked up, and being cut off
then costs you the refinement instead of the whole iteration.

## WORKFLOW

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

For each past iteration with results in `evolution_summary.jsonl` but NO report, write one of **<=30 lines**: what changed, what improved/regressed and why, and a takeaway.

### Step 1: Analyze

1. Read `evolution_summary.jsonl`, `frontier_val.json`, `config.yaml`, and recent `logs/gn/<harness>/<model>/` records.
2. Formulate 3 hypotheses — each falsifiable and targeting a different mechanism.

### Step 2: Prototype — MANDATORY

**You MUST prototype your mechanism before writing the final harness.** Candidates that skip prototyping tend to have bugs or produce no improvement. But the session has a hard cutoff (see **The clock** above) and a prototype that runs the real benchmark will reach it.

1. Write a test script in `/tmp/` exercising the core state/selection logic in isolation, with the oracle replaced by a stub computing feedback against a secret you choose.
2. Pull real records from `logs/` to test against.
3. Where the question is about the **model's** behaviour rather than the harness's logic -- can
   the frozen model write this format, does it follow this instruction, what does it do when the
   state is ambiguous -- a stub cannot answer it, so ask the real model. Keep that to **at most 20
   episodes and about 10 minutes**, and record what you measured in the harness's docstring.
4. **Never run a whole split.** The outer loop benchmarks all three candidates on it the moment you
   return. Running it yourself pays for the same episodes twice, and the second payment comes out
   of your own clock -- this is what cost two of the first four iterations.
5. Try 2-3 variants and compare before picking the best one.
6. Delete scripts when done.

### Step 3: Implement

1. Copy a top-performing base harness to `agents/<name>.py`, then make targeted modifications.
2. Implement the new mechanism according to your hypothesis.
3. **Self-critique (mandatory):** re-read the file. Does it introduce a genuinely NEW mechanism, or is `solve()` the base with different numbers? If the latter, REWRITE.
4. Validate: `python -c "from arbench_mh.gn.agents.<name> import *; print('OK')"`

Do not edit `config.yaml` to register candidates. The benchmark auto-discovers files in `agents/`.

### Step 4: Write pending_eval.json

```json
{"iteration": "<N>",
 "candidates": [{"name": "<snake_case_name>", "file": "agents/<name>.py",
                 "hypothesis": "<falsifiable claim>", "axis": "exploitation|exploration",
                 "base_system": "<what it builds on>", "components": ["tag1", "tag2"]}]}
```

Output: `CANDIDATES: <name1>, <name2>, <name3>`

## GNHarness Interface

```python
class GNHarness(ABC):
    def __init__(self, env, solver): ...
    def llm(self, messages, temperature=0.7, max_tokens=None) -> str: ...
    @abstractmethod
    def solve(self) -> str: ...      # the final four-digit guess
```

- Extend `GNHarness` from `..harness_base`
- `env.guess(number) -> (correct_position, correct_digit_wrong_position)` is the only way to learn anything; every call spends a guess
- `env.system_prompt`, `env.turn_prompt()`, `env.final_prompt`, `env.record`, `env.history`, `env.remaining`, `env.budget`, `env.solved`
- Use `self.llm(messages)` for model calls (NOT a client of your own)
- `solve()` must work on the first instance it ever sees — there is no warm-up
- The secret is not reachable from the harness. Scoring a candidate guess against it before playing it is not available

## Directory Structure

- Val results: `logs/gn/<harness>/<model>/val.json` (accuracy, total_tokens)
- Episodes: `logs/gn/<harness>/<model>/episodes.json`
- Test results: `results/gn/<harness>/<model>/test.json` (separate dir, never exposed during evolution)

## evolution_summary.jsonl Format

```json
{"iteration": 1, "system": "constraint_table", "avg_val": 12.0, "axis": "exploitation", "hypothesis": "...", "delta": +6.0, "outcome": "12.0% (+6.0)", "components": ["tag1", "tag2"]}
```

## Scoring

Two axes, as upstream: `accuracy` (exact match) and cost. Cost is `total_tokens`, the mean input plus output tokens per episode. A harness that buys accuracy by generating far more text is therefore visible on the frontier rather than hidden.

## Component Analysis

Treat `evolution_summary.jsonl`, `frontier_val.json`, and recent guess records as the only shipped history sources in this repo.
