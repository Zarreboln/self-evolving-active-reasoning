---
name: meta-harness
description: Run one iteration of guessing harness evolution on AMIGO. Called by meta_harness.py or interactively via /meta-harness.
---

# Meta-Harness (AMIGO harness evolution)

Run ONE iteration of harness evolution. Do all work in the main session — do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run the benchmark.** You analyse results and episode transcripts, prototype changes, and implement new harnesses. The outer loop (`meta_harness.py`) handles benchmarking separately.

## The clock

The session is cut off after 40 minutes, and nothing you have not written to disk survives it.

So write the three harness files **and `pending_eval.json`** as soon as the three mechanisms are
decided, before any further prototyping or polishing. The outer loop reads the files from disk
after the session ends, so whatever you improve afterwards is still picked up, and being cut off
then costs you the refinement instead of the whole iteration.

## The task

AMIGO, "Guess My Preferred Dress" (arXiv 2603.28662), difficulty tier tau=0.6. A gallery of dress photographs is shown; one is the hidden target. The player has **20 turns**. Each turn is one yes/no question about one observable construction attribute of the target, answered Yes / No / Unsure. A question that breaks a protocol rule is answered **Skip** and the turn is gone. At the end the player names an image by its position.

The model is frozen. Everything a harness can change sits between the model and the environment.

### What the measurements say about where the headroom is

From 100 episodes on each of the five tiers with the paper's own player loop (`baseline_paper`):

- **Skip eats most of the budget.** Around half the turns come back Skip under our first detector; replayed through the paper's own prompts, 65%. The rate barely moves across the five difficulty tiers, so it is a property of how questions get drafted, not of how hard the task is.
- **Of those Skips, only about a quarter are the deterministic P1 check** (empty, not a question, two questions, not Yes/No-answerable). The other three quarters are the two prompt-based checks: asking about a prohibited attribute or an image index, or asking again about a fine-grained attribute type already settled.
- **It never stops early.** 19.1 to 19.3 turns of 20, every tier. It also does not commit when the answers have already singled a dress out.
- At tau=0.6 the paper's loop scores **26.0% against a 9.9% chance rate**.

### The protocol, and what a harness may do about it

The eleven rules are **published**, so a harness may implement them and screen its own drafts before submitting one. That is legitimate and is where the P1 quarter of the waste sits. What it may not do is ask the environment whether a draft is legal: `env.ask` is the only way to find out, and the turn is spent either way. The two prompt-based checks are a judgement no harness can reproduce exactly, so the rest of the waste has to be earned by drafting better questions.

## CRITICAL CONSTRAINTS

- You MUST implement 3 new harnesses every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 3 candidates per iteration: a mix of exploitation and exploration.

### Four invariants, all audited

- **Frozen solver.** `self.llm(...)` only. No clients of your own, no base_url or api_key, no network, no subprocesses.
- **No looking at the pictures.** You may hand images to the model and reason over what it says; you may not analyse pixels yourself. No PIL, no numpy on image data, no hashing or comparing image bytes. The player in this task is a vision model, and a harness that did its own vision would be a different system.
- **No cross-episode state.** Every episode starts from nothing: no module-level cache, no file written in one episode and read in another. Episodes are evaluated concurrently, and on another task a harness found it could merge sixteen concurrent episodes into one model call and divide the per-episode cost by sixteen — that is a property of the runner, not of the task. The benchmark snapshots your module's globals before the run and compares them after; a candidate that mutated them is rejected.
- **No instance-specific hints.** Nothing hardcoded about a particular gallery, dress or id.

### Anti-parameter-tuning rules

Check `evolution_summary.jsonl` for what has been tried. Parameter sweeps — how many drafts, how much of the transcript is echoed, prompt wording — almost always regress or tie.

**Good candidates change a fundamental mechanism:**

- A new way of turning answers into constraints over the gallery
- A new state representation (a per-image score, an attribute table, a ruled-out set)
- A new way of drafting a question: what the model is shown of the record, whether it drafts several and one is chosen, how the attribute already covered is kept out
- A new screening rule applied to a draft before the turn is spent
- A new stopping rule, and a new rule for which image to name

Exploitation axes: A=question drafting, B=state across turns, C=choosing among drafts, D=shape of the record fed back, E=screening before submitting, F=stopping and naming. If the last 3 iterations explored the same axis, pick different ones.

## WORKFLOW

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

For each past iteration with results in `evolution_summary.jsonl` but NO report, write one of **<=30 lines**: what changed, what improved or regressed and why, and a takeaway.

### Step 1: Analyse

1. Read `evolution_summary.jsonl`, `frontier_val.json`, `config.yaml`, and recent transcripts under `logs/amigo/<harness>/<model>/episodes.json`. Each episode records every question and whether it came back Yes, No, Unsure or Skip.
2. Formulate 3 hypotheses — each falsifiable and targeting a different mechanism.

### Step 2: Prototype — MANDATORY, and on a budget

**You MUST prototype your mechanism before writing the final harness.** Candidates that skip prototyping tend to have bugs or produce no improvement. But the session has a hard cutoff (see **The clock** above) and a prototype that runs the real benchmark will reach it.

1. Write a test script in `/tmp/` exercising the core state and selection logic in isolation, with the environment replaced by a stub: a fixed script of Yes/No/Unsure/Skip replies, or replies replayed from a recorded episode.
2. Pull real transcripts from `logs/` to test against — they are the cheapest source of what the model actually writes.
3. Where the question is about the **model's** behaviour rather than the harness's logic — can it draft five questions at once, does it keep to a format, what does it write when told an attribute is already covered — a stub cannot answer it, so ask the real model. Keep that to **at most 20 episodes and about 10 minutes**, and record what you measured in the harness's docstring.
4. **Never run a whole split.** The outer loop benchmarks all three candidates on it the moment you return. Running it yourself pays for the same episodes twice, and the second payment comes out of your own clock.
5. Try 2-3 variants and compare before picking the best one.
6. Delete scripts when done.

### Step 3: Implement

1. Copy a top-performing base harness to `agents/<name>.py`, then make targeted modifications.
2. Implement the new mechanism according to your hypothesis.
3. **Self-critique (mandatory):** re-read the file. Does it introduce a genuinely NEW mechanism, or is `solve()` the base with different numbers? If the latter, REWRITE.
4. Validate: `python -c "from arbench_mh.amigo.agents.<name> import *; print('OK')"`

Do not edit `config.yaml` to register candidates. The benchmark auto-discovers files in `agents/`.

### Step 4: Write pending_eval.json

```json
{"iteration": "<N>",
 "candidates": [{"name": "<snake_case_name>", "file": "agents/<name>.py",
                 "hypothesis": "<falsifiable claim>", "axis": "exploitation|exploration",
                 "base_system": "<what it builds on>", "components": ["tag1", "tag2"]}]}
```

Output: `CANDIDATES: <name1>, <name2>, <name3>`

## AMIGOHarness interface

```python
class AMIGOHarness(ABC):
    def __init__(self, env, solver): ...
    def llm(self, messages, temperature=0.7, max_tokens=None) -> str: ...
    @abstractmethod
    def solve(self) -> int: ...      # the 0-based gallery index named
```

- Extend `AMIGOHarness` from `..harness_base`
- `env.ask(question) -> "Yes" | "No" | "Unsure" | "Skip"` is the only way to learn anything, and **always** spends a turn
- `env.gallery`, `env.n`, `env.image_block(i)`, `env.gallery_blocks()`, `env.rules`, `env.history` (Skips included), `env.answered` (Skips left out), `env.remaining`, `env.budget`
- Use `self.llm(messages)` for model calls, never a client of your own. Messages take OpenAI-style content blocks, so an image block goes straight into a user message.
- `solve()` must work on the first episode it ever sees — there is no warm-up
- The target is not reachable from the harness

## Directory structure

- Val results: `logs/amigo/<harness>/<model>/val.json` (accuracy, chance, skip_rate, turns, total_tokens)
- Episodes: `logs/amigo/<harness>/<model>/episodes.json`
- Test results: `results/amigo/<harness>/<model>/test.json` (separate dir, never exposed during evolution)

## Scoring

Two axes, as upstream: `accuracy` (the named image is the target) and cost. Cost is `total_tokens`, the mean input plus output tokens of the frozen model per episode. **The chance rate is not a fixed number**: galleries differ in size, so it is the mean of 1/gallery over the episodes, and `val.json` records it. A harness that beats the previous best but not chance has not done anything.

## evolution_summary.jsonl format

```json
{"iteration": 1, "system": "screened_drafts", "avg_val": 31.0, "axis": "exploitation",
 "hypothesis": "...", "delta": +5.0, "outcome": "31.0% (+5.0)", "components": ["tag1", "tag2"]}
```
