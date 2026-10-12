---
name: meta-harness-cd
description: Run one iteration of harness evolution for Circuit Decoding (fixed Qwen2.5-7B-Instruct task model). Called by evolve.py.
---

# Meta-Harness (Circuit Decoding harness evolution)

Run ONE iteration. Do the work in the main session; do not delegate the analysis or the implementation to subagents (constraints get lost and you end up with parameter variants).

**You do NOT run the real benchmark.** The outer loop (`evolve.py`) evaluates every candidate on the fixed task model after you finish. You analyze the last results and the raw traces, prototype, implement 3 new harnesses, and write `pending_eval.json`.

## The task being solved by the (fixed) task model

Circuit Decoding (Multi-Turn Puzzles): three hidden 3-input boolean circuits A, B, C built from an announced gate inventory (e.g. "1 AND, 5 OR, 2 NOT"). The task model gets 18 queries, one per turn, of the form `<query>A(0, 1, 1)</query>`; the environment answers with one output bit. After the 18th answer it must output all three 8-row truth tables as a 24-bit string in `<guess>...</guess>`.

Everything about the task is fixed and NOT yours to change: the system prompt (`mt_puzzles/cd_instruction.txt`), the budget (18), the environment lines (official wording from the paper transcript: `A(0, 0, 1) = 1` / `Please send your next query. 17 queries remaining.`; the 18th answer arrives together with the final prompt), the final prompt, the query/guess formats, the model (Qwen2.5-7B-Instruct, temperature 0.7, max 1200 output tokens, 12k context).

The hypothesis space is exact and public: the inventory implies a finite set H of joint truth tables (252 / 2886 / 3924 depending on the inventory). Each answer filters H. Max-entropy querying pins the truth in 11.6 queries on average; 18 is enough for 100% in principle. Measured on this 7B model: the bare harness collapses the feasible set to one hypothesis (uniq@end) in only ~34% of episodes and the final guess is right in 0.3%; the hand-written belief summary `enum15` raised joint truth table correct to 6.7% but LOWERED uniq@end to 26%, raised repeated queries from 4% to 12% and illegal turns from 0.3% to 2.7%, and left ex-ante efficiency at 0.61 (random fresh queries would score 0.63). The model mostly scans entries in order, ignores "do not ask these again", and rarely uses the summary for anything but copying a listed configuration at the end.

## What a harness is here

A single file `agents/<name>.py` with three functions. It builds the text appended to the environment's feedback each turn, from PUBLIC information only:

```python
NAME = "<name>"          # must equal the file stem
PARENT = "<name of the harness this was copied from>"

def init_state(view) -> dict:              # once per episode, before the first query
def build_feedback(view, state) -> str:    # after each answered query (turns 1..17); "" = nothing appended
def final_hint(view, state) -> str:        # appended before the final-guess prompt; "" = nothing
# optional:
def filter_action(view, state, text, q) -> None | str
    # called on every model reply before it reaches the environment. `q` is the parsed query
    # index (None if the reply had no parsable query). Return a message to send the model
    # instead of accepting the reply; it replies again. At most 2 re-asks per turn, none of
    # them spends one of the 18 queries, all of them cost tokens (counted). Return None to accept.
def compose_messages(view, state, msgs) -> list[dict]
    # called before every model call with the full message list; return what the model should
    # see instead (compress the history into a ledger, reorder, drop chatter...). Invariants
    # enforced by the runner: msgs[0] (system prompt) unchanged; every observation line
    # "A(0, 0, 1) = 1" must still appear in some user message; the last message is the user turn.
    # A violation errors the episode and the candidate.
```
The one thing a harness can never do: send a query the model did not write. It can re-ask, rewrite what the model sees, and tell it what to do -- but the `<query>` that reaches the environment is always the model's own reply.

`view` (see `../../runner/cd_harness_run.py: View`) has: `c=3, k=3, budget=18, turn`, `H` (numpy (|H|,24) uint8, the hypothesis space), `alive` (indices of H still consistent with every answer so far), `game` (`game.expected_bits(q, alive)`, `game.available_bits(alive) -> (bits, q)`), `history` ([(q, bit)] in order), `transcript` (the model's raw replies so far), `components`, `last_q`, `last_bit`, `n_before`, `n_after`, `last_bits_expected`, `last_bits_available`, and `view.fmt(q)` -> `"A(0,1,1)"`. Query index q = circuit*8 + row, row = 4*x0 + 2*x1 + x2.

The truth table is never in `view`. A harness therefore cannot leak it; it also cannot look up anything instance-specific, and must not try (no hardcoded tables, no per-instance rules, no mention of instance ids). Everything must be a general mechanism computed from H, alive, history and transcript.

Helpers you may import: `from harness_lib import fmt, determined, count_line, determined_text, enumeration_text, observations_text, split_table_text, best_query`. Read `harness_lib.py` first. Do not edit `harness_lib.py`, existing agents, `evolve.py`, `claude_wrapper.py`, `tools/`, or anything under `../../runner`. Never import from another candidate; copy the code you want to reuse.

## Regime (stated in the task prompt — obey it)

- `open`: the harness may compute anything from the view, including ranking or recommending queries, and may tell the model what to do.
- `state_only`: the harness may present, reorganize, filter or summarize the feasible set and the history in any form, but must NOT rank, score or recommend queries, and must not compute information gain, entropy or 0/1 split counts for candidate queries. `available_bits`, `expected_bits`, `best_query`, `split_table_text`, `log2`, `entropy` are rejected by a static check. Deciding what to ask stays with the model; you only change what it sees.

## Metrics (per candidate, 50 search instances x 2 seeds = 100 episodes)

- `circuit-wise accuracy` — PRIMARY: the benchmark's own metric (Multi-Turn Puzzles, Table 1). Fraction of the 3 circuits whose full 8-row truth table is right in the final guess.
- `joint truth table correct` — the whole 24-bit joint truth table is right (the paper's noun; our reading of a successful episode). Tie-breaker.
- `uniq@end` — ours, diagnostic. Feasible set collapsed to exactly one hypothesis after the 18 queries. Luck-free: it measures information gathering, not the final guess.
- `eff` — mean ex-ante efficiency over legal turns: expected bits of the chosen query / best available. 1.0 = oracle, 0.63 = random fresh entry, 0.54 = scanning entries in order.
- `zero%` — queries whose answer was already determined (0 bits). `orcl%` — queries equal to the max-entropy query. `rep%` — queries identical to an earlier one. `ill%` — turns without a parsable query (they burn one of the 18). `prs%` — final guess parsable. `alive` — mean |feasible set| at the end. `tok` — output tokens per episode.

Selection (RRSI-style, see `evolution_summary.jsonl` fields `admissible` / `selection`): a noise band `delta` is estimated from the bare baseline by bootstrap (about 0.09 at 200 episodes, 0.13 at 100). A candidate becomes the new frontier only if (1) its circuit-wise accuracy is not below the best-so-far minus delta, and (2) if its gain over the incumbent exceeds delta, its token cost per episode grew by at most beta0 + beta1 * gain (0.1 + 2.0 * gain, relative); a gain inside the band is admitted only when it saves at least 5% tokens. So: do not pay for small gains with long messages, and do not expect +5 points to register.

Before any candidate is benchmarked, a critic screens the file: file/network/interpreter access and any reference to the truth or the instance file are rejected outright (regex), then an LLM review rejects instance leakage, degenerate near-copies of the parent, unbounded work, and regime violations. Rejections appear in `evolution_summary.jsonl` with the reason. Write mechanisms, not lookup tables.

## CRITICAL CONSTRAINTS

- You MUST implement 3 new harnesses this iteration, each testing ONE mechanism. Mix exploitation (build on the frontier) and exploration (a different mechanism).
- Do NOT write "the frontier is optimal", do not stop early, do not skip prototyping.
- Anti-parameter-tuning: a candidate whose logic equals an existing agent except for constants (list caps, thresholds, wording tweaks) is not a candidate. Check `evolution_summary.jsonl` for what has been tried. Good candidates change WHAT the model sees (structure, position, granularity, filtering, per-circuit vs joint views, consistency checks of the model's own stated reasoning in `transcript`, framing the choice as a short menu, ...) or WHEN it sees it.
- Anti-overfitting: no instance-specific content, no inventory-specific tricks that would not apply to a fresh inventory, no mention of instance ids.
- Context is finite: the 7B model has 16k tokens of context, a 700-token system prompt, and it is verbose (2-6k output tokens per episode under bare). enum15 overflowed in 2% of episodes at 12k. 17 turns x your text + its replies must fit; `compose_messages` is the tool for keeping the context short. Measure the length of what you send (the mock traces show it verbatim; `max_context_chars` is in every episode result).

## WORKFLOW

### Step 0: Post-eval reports (write if missing)
For each past iteration that has rows in `evolution_summary.jsonl` but no `reports/iter<N>.md`, write one (<= 30 lines): what changed, what moved (circuit-wise accuracy, joint truth table correct, uniq@end, eff, rep%, ill%), what the traces show the model actually did with the text, one takeaway.

### Step 1: Analyze
Read `evolution_summary.jsonl`, `frontier_val.json`, `reports/`, then the raw traces of the frontier and of the latest candidates: `python tools/show_trace.py traces/<name>/search.jsonl --list`, `--worst 5`, `--key <k>`. Look at what the model wrote right after your text. Where does the feasible set stop shrinking? Does the model repeat, scan, or query determined entries? Does it use the enumeration at the end? Form 3 falsifiable hypotheses, each about a different mechanism.

### Step 2: Prototype — MANDATORY
`tools/run_mock.sh agents/<name>.py sequential 10` and `... greedy 10` run the harness against scripted task models in seconds (no GPU). The scripted models do not read your text (only `follow` reads a "Most informative next query: X(a,b,c)" hint; a re-ask from `filter_action` just makes them reply again), so this checks that the harness runs on real instances, respects the compose_messages invariants, what it emits, and how long the text is — not whether the 7B model will respond to it. Real instances for prototyping: `data/cd_search_v2.jsonl` (the 100 search instances; `controller.components` is the inventory, the rest is what the environment knows). Inspect the output with `tools/show_trace.py logs/mock/<name>__sequential.jsonl --key <k>`. Prototype in `/tmp/`, delete scratch files when done.

### Step 3: Implement
Copy the most relevant existing agent to `agents/<name>.py`, modify it, set `NAME`/`PARENT`. Self-critique: is this a new mechanism or a parameter variant? Then validate: `tools/run_mock.sh agents/<name>.py sequential 5` and `tools/run_mock.sh agents/<name>.py greedy 5` must both run without error.

### Step 4: Write pending_eval.json (path given in the task prompt)
```json
{"iteration": <N>, "regime": "<open|state_only>",
 "candidates": [
   {"name": "<snake_case>", "file": "agents/<name>.py", "parent": "<existing agent>",
    "hypothesis": "<falsifiable claim about uniq@end / eff / rep%>",
    "axis": "exploitation|exploration", "mechanism": "<one sentence>",
    "prediction": "<expected uniq@end and why>"}
 ]}
```
Finish with the line: `CANDIDATES: <name1>, <name2>, <name3>`
