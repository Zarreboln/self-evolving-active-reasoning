# Proposer cost test (2026-11-26)

## Setup
- **Purpose.** Measure what one proposer session costs, and whether the proposer produces a sensible change,
  before choosing the proposer model for experiment B.
- **Workspace.** The same for every model: Wordle basic5 pilot (`~/evo_b_runs/pilot_wordle`), parent c008
  (search score 0.983, 29 wins and 1 loss). Built by `orchestrate.fill_workspace`, and run in the Docker
  sandbox (`sandbox/run_proposer.py`).
- **Cost recording, as in MAS.S62 HW2.**
  - DeepSeek is priced from the session's own token counts at DeepSeek's prices.
  - Claude Code's `total_cost_usd` uses Anthropic list prices. For DeepSeek it is far too high (HW2: about
    9×; here 14×).

## Results
| Proposer | Billed to | Start (UTC) | Time | Turns | Tokens (input / cache read / output) | Cost |
|---|---|---|---|---|---|---|
| Opus 5.5 | Claude subscription | 09:58:12 | 1.9 min | 16 | 30 + 35.6k cache write / 345k / 7.9k | $0.51 at list price (subscription, not metered) |
| DeepSeek V4-Pro | DeepSeek account | 09:58:12 | 21.4 min | 48 | 81k / 3.17M / 84k | **$0.29** at the off-peak price (Claude Code reported $4.09) |
| Qwen3.5-9B (Prime) | Prime credits | 10:09:17 | 1 s | 0 | 0 | $0: Prime's Anthropic-format API does not serve this model |
| Qwen3-235B (Prime) | Prime credits | 10:09:17 | 1 s | 0 | 0 | $0: the account allows only google-vertex and nebius, and this model is served by deepinfra, parasail and venice |

**A note on the DeepSeek price.**
- The session started at 09:58 UTC on a Thursday, inside DeepSeek's peak window (06:00–10:00 UTC, prices
  doubled), and ran until 10:19.
- The off-peak price gives $0.29. An upper bound, if the whole session were billed at the peak price, is $0.58.

## What each proposer changed
Both found the same cause of the one remaining loss: after the validator's retries ran out, the harness
submitted a guess it already knew was invalid (NICE, 4 letters), twice, and lost.
- **Opus** submits a word checked in code instead of the invalid one. The diff is 25 lines.
- **DeepSeek** never sends a guess known to be invalid. The diff is 44 lines.
- Both pass the static checks. Neither has been scored yet.

## Other notes
- DeepSeek runs with thinking on: about 11 times slower than Opus here, and about 9 times more tokens read.
- Prime-hosted Qwen models are reachable only through an Anthropic-to-OpenAI bridge
  (`sandbox/bridge/`, LiteLLM). The bridge is built but not run yet; this waits for the user's go-ahead,
  because it spends Prime credits (estimate under $1).
