---
name: meta-harness
description: Run one iteration of interrogation harness evolution. Called by meta_harness.py or interactively via /meta-harness.
---

# Meta-Harness (Interrogation Harness Evolution)

Run ONE iteration of harness evolution. Do all work in the main session — do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run benchmarks.** You analyze results + interrogation traces, prototype changes, and implement new harnesses. The outer loop (`meta_harness.py`) handles benchmarking separately.

## The task

AR-Bench Detective Cases. Five suspects, one murderer. The detective is shown the case background, then has up to 25 turns; each turn names one suspect and asks them one question, and that suspect answers in free text. After the last turn the detective names the murderer. Chance is 20%.

The model is frozen. Everything a harness can change sits between the model and the suspects.

## CRITICAL CONSTRAINTS

- You MUST implement 3 new harnesses every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 3 candidates per iteration: mix of exploitation and exploration.

### Anti-parameter-tuning rules

The most common failure mode is creating harnesses that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried — parameter sweeps (turn counts, how much of the transcript is echoed, prompt wording, how many candidate questions are drafted) almost always regress or tie.

**Good candidates change a fundamental mechanism:**

- A new way of choosing whom to interrogate (by contradiction, by coverage, by how little is known)
- A new prompt architecture (organize the transcript by claim instead of by turn)
- A new state representation (a per-suspect claim table, a ruled-out set, an alibi graph)
- A new stopping rule (accuse once the evidence settles instead of spending the budget)

**Bad candidates just tune numbers.** If the logic in `solve()` is identical to the base except for constants, it's a parameter variant. Rewrite with a genuinely novel mechanism.

**Combining harnesses is valid.** Take the state representation from harness A and the selection rule from harness B.

Exploitation axes: A=Question generation, B=State maintained across turns, C=Whom and what to ask next, D=Shape of the context fed back each turn, E=Stopping, F=Extra LLM passes between turns. If last 3 iterations explored the same axis, pick different ones.

### Anti-overfitting rules

- **No case-specific hints.** Do not hardcode knowledge about specific cases. Harnesses must be general-purpose.
- **Never mention a case's content** in harness code, prompts, or comments. If your improvement only helps one case, it's too specific.
- **General patterns are OK.** Rules like "ask the suspect whose alibi is least constrained" or "never repeat a question already answered" are fine — they apply broadly.

## WORKFLOW

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

Check the reports directory (path in the task prompt's "Run directories" section). For each past iteration that has results in `evolution_summary.jsonl` but NO report, write one. Each report should be **<=30 lines** covering: what changed, what improved/regressed and why, and a takeaway for future iterations.

### Step 1: Analyze

1. **Read all state files:**
   - `evolution_summary.jsonl` — what's been tried (one JSON per candidate)
   - `frontier_val.json` — current best (val accuracy)
   - `config.yaml` for the current dataset and baselines
   - recent `logs/dc/<harness>/<model>/` traces if they exist

2. Formulate 3 hypotheses — each must be falsifiable and target a different mechanism.

### Step 2: Prototype — MANDATORY

**You MUST prototype your mechanism before writing the final harness.** Do NOT skip this step. Candidates that skip prototyping tend to have bugs or produce no improvement.

For each candidate:

1. Write a test script in `/tmp/` that exercises the core state/selection logic in isolation, with the suspects replaced by a stub returning a fixed script of replies.
2. Pull real transcripts from `logs/` to test against.
3. Try 2-3 variants and compare before picking the best one.
4. Delete scripts when done.

### Step 3: Implement

For each of the 3 candidates:

1. Copy a top-performing base harness to `agents/<name>.py`, then make targeted modifications. This copy-then-edit approach ensures correct imports and proven patterns.
2. Implement the new mechanism according to your hypothesis.
3. **Self-critique (mandatory):** After implementing, re-read the file and check: does this harness introduce a genuinely NEW mechanism, or is it just a parameter variant? If `solve()` is identical to the base except for numbers, REWRITE with a truly novel mechanism.
4. Validate: `python -c "from arbench_mh.dc.agents.<name> import *; print('OK')"`

Do not edit `config.yaml` just to register candidates. The benchmark auto-discovers files in `agents/`.

### Step 4: Write pending_eval.json

Write to the path specified in the task prompt (NOT hardcoded — it may be in a run-specific subdirectory):

```json
{
  "iteration": "<N>",
  "candidates": [
    {
      "name": "<snake_case_name>",
      "file": "agents/<name>.py",
      "hypothesis": "<falsifiable claim>",
      "axis": "exploitation|exploration",
      "base_system": "<what it builds on>",
      "components": ["tag1", "tag2", "..."]
    }
  ]
}
```

Output: `CANDIDATES: <name1>, <name2>, <name3>`

## DCHarness Interface

```python
class DCHarness(ABC):
    def __init__(self, env, solver): ...
    def llm(self, messages, temperature=0.7, max_tokens=None) -> str: ...
    @abstractmethod
    def solve(self) -> int: ...      # the accused suspect's 0-based index
```

- Extend `DCHarness` from `..harness_base`
- `env.interrogate(name, question) -> reply` is the only way to reach a suspect; every call spends a turn
- `env.init_info`, `env.suspects`, `env.system_prompt`, `env.turn_prompt(t, suspect=...)`, `env.final_prompt`, `env.history`, `env.record`, `env.remaining`, `env.budget`
- `env.parse_choice(text)` returns the 0-based index, or None if the reply did not follow AR-Bench's `Answer: <letter>` format
- Use `self.llm(messages)` for model calls (NOT a client of your own)
- `solve()` must work on the first case it ever sees — there is no warm-up
- The suspects' stories and the case's gold key questions are not reachable from the harness

## Directory Structure

- Val results: `logs/dc/<harness>/<model>/val.json` (accuracy, total_tokens)
- Episodes: `logs/dc/<harness>/<model>/episodes.json`
- Test results: `results/dc/<harness>/<model>/test.json` (separate dir, never exposed during evolution)

## evolution_summary.jsonl Format

One JSON object per line, one line per evaluated candidate:

```json
{"iteration": 1, "system": "claim_table", "avg_val": 38.0, "axis": "exploitation", "hypothesis": "...", "delta": +3.0, "outcome": "38.0% (+3.0)", "components": ["tag1", "tag2"]}
```

## Scoring

Two axes, as upstream: `accuracy` and cost. Cost here is `total_tokens`, the mean input plus output tokens per episode. A harness that buys accuracy by spending more of either is therefore visible on the frontier rather than hidden.

## Component Analysis

Treat `evolution_summary.jsonl`, `frontier_val.json`, and recent interrogation traces as the only shipped history sources in this repo.
