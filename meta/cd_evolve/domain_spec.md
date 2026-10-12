# domain_spec.md — Meta-Harness applied to CircuitDecoding (active reasoning)

Written 2026-10-10 following meta_harness_official/ONBOARDING.md. "Confirmed" = decided in conversation; "default" = my proposal, change freely; "unknown" = open.

## Task and success criterion
- Task: Multi-Turn Puzzles CircuitDecoding, 300 released instances (3 inventories x 100), 18 queries, final 24-bit guess. Example: inventory {AND 1, OR 5, NOT 2}; query `<query>A(0,1,1)</query>` -> "The output of that query is 1."; final `<guess>0001...1</guess>`.
- Task model (fixed): Qwen2.5-7B-Instruct, vLLM, temperature 0.7, max_tokens 1200, context 12288. Confirmed (the paper's own weak-model regime; the 7B is where the hand-written harness failed).
- Primary metric: uniq@end (feasible set collapsed to one hypothesis after 18 queries). Secondary: exact. Diagnostics: eff, zero%, oracle%, repeat%, illegal%, parsed%, tokens. Confirmed (uniq@end is luck-free and isolates information gathering from the final copy step).

## What the harness may change / what stays fixed
- Editable: one file per candidate `agents/<name>.py` implementing `init_state / build_feedback / final_hint` over a `View` with public information only (H, alive, history, transcript, game). Interface in `../../runner/cd_harness_run.py`.
- Fixed: system prompt, budget, environment feedback lines (official wording from the MTP paper appendix transcript: `A(0, 0, 0) = 1` + `Please send your next query. N queries remaining.`; 18th answer delivered with the final prompt), final prompt, formats, model and sampling settings, scoring code. Unknown: official handling of a reply without a query (we re-ask once, the turn counts).
- Truth isolation: structural — the truth table is never passed into a harness. Instance-specific content is forbidden by the proposer prior and checked by reading the candidate.
- Two regimes (confirmed as the experiment's design): `open` (harness may compute and recommend decisions) and `state_only` (harness may only present state; static check rejects information-gain / ranking calls). Prediction (Probe 4): `open` climbs toward the oracle ceiling by turning the model into a copier; `state_only` stays near bare.

## Evaluation and selection
- Split: `splits/cd_split.json`, seed 20261010, search 50 (17/17/16 by inventory), held-out 250. Confirmed.
- Candidate evaluation: search split x 2 seeds = 100 episodes, concurrency 48, on the Engaging vLLM server (job `serve7b`), traces synced back to `logs/<run>/traces/<name>/search.jsonl`, score in `logs/<run>/scores/<name>.json` (bootstrap CIs).
- Proposer access: this workspace only — `agents/`, `logs/<run>/{evolution_summary.jsonl, frontier_val.json, reports/, traces/, scores/}`, `data/cd_search.jsonl` (the 50 search instances, with truths — like validation labels), tools. Held-out instances live in `../../mt_puzzles/cd_test.jsonl` and held-out results in `../../results/cd_heldout/<run>/`, both outside the workspace (operational isolation, same as the reference implementation).
- Selection: frontier = best uniq@end on search (ties by exact). Final candidate chosen before any held-out number is seen. `evolve.py --test` freezes the run, evaluates baselines + frontier on held-out x 3 seeds, and refuses further evolution under that run name.
- Failure handling: candidates that fail import / mock smoke test are recorded as failed and skipped; episodes with API errors are retried once on resume; a candidate with >5% errored episodes is marked invalid.

## Baselines, budget, stopping
- Seeds (all evaluated under identical conditions first): bare, enum15 (= hand-written belief summary), symbolic, facts (ladder a), splits (ladder c), oracle_hint (ladder d, ceiling), t3_gate.
- Budget per iteration: 3 candidates x ~4.5 min GPU + one proposer session (`claude -p --model opus`, subscription; timeout 2400 s). Default: 8 iterations per regime; measure the first iteration before committing (unknown: proposer token cost per session -> measured from `logs/<run>/claude_sessions/*/meta.json`).
- Stopping: fixed iteration count; no plateau stop (reference rule). Kill criterion for the paper: the evolved harness must beat `enum15` on held-out uniq@end with a CI excluding 0; otherwise the paper reports the ladder only.

## Execution evidence saved
- Per episode: every message sent, every raw model reply, per-turn feasible-set sizes, bits chosen/available, best query, flags; final prompt and reply; harness state dump; harness file sha; model/temperature/budget.
- Per candidate: score JSON with CIs; per iteration: summary row, proposer session log (prompt, tool calls, tokens, cost), report.

## Smallest first implementation and acceptance check
1. CPU: all seeds x scripted policies (done 2026-10-10: oracle policy -> 100% uniq@end under every harness; sequential -> 20%, eff 0.535 matching the synthetic reference).
2. GPU: `bare` and `enum15` on the search split must reproduce the direction seen in results/gate_cd.jsonl (bare uniq@end ~34%, enum15 lower, exact higher). Unknown until job 25535085 (ladder) finishes.
3. One proposer iteration in regime `open`: 3 valid candidates, benchmarked, summary row written. Then decide the iteration budget.

## Constraints
- H100 server is off limits these days (user). Engaging only: mit_normal_gpu 6 h jobs; the serve job must be resubmitted; `evolve.py` waits for `serve_7b.endpoint`.
- No secrets in the workspace; the proposer runs with the subscription (ANTHROPIC_API_KEY unset, CLAUDECODE unset).
- Unknown: proposer cost/rate limits under the Max subscription when 2 regimes x 8 iterations run back to back.
