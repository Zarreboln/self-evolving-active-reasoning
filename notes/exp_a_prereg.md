# Experiment A: pre-registration (written 2026-11-25, before any A result)

## Question
Does the gain of a harness component depend on the task in a *structured, partially shared* way?
Three outcomes, decided below before the data:
- **Global**: every component's gain is about the same on every task → one shared harness is enough; the fuzzy hierarchy has nothing to explain.
- **Saturated / per-task**: gains differ by task but with no shared pattern → only per-task evolution helps; nothing to share.
- **Structured** (A passes): gains have task-dependent signs and a low-dimensional pattern across tasks that the three dimension groups (pure epistemic / intermediate / world-changing) do not fully capture.

## Setup
- Policy Qwen3.5-9B (vLLM, thinking off, T=0.7, max_tokens 8192, 64K context). DC suspects: same model. SP referee / 20Q and Guess Who gamemaster: Jev jev-1.14.0 via JevHub.
- Harness: the react seed (`mb/harness/react.py`) plus one component at a time (`mb/harness/strategies.py`): `belief` (carried hypothesis list), `plan` (carried plan), `check` (verify before terminal action), `brief` (≤50 words reasoning). `seed` is the control.
- Cells: gn/all, dc/all, mastermind/basic, hangman/basic, twentyq/basic-things, wordle/basic5 (pure epistemic); blicket/n5, sciworld/test-conductivity (intermediate); alfworld/pick_and_place_simple, babyai/GoToLocal (world-changing).
- Instances: first 30 of the search split (blicket/n5 has 20), identical across strategies (paired). One episode per (strategy, instance).
- Score: each bench's official score (all in [0,1] here). Errored episodes are rerun on resume; any left at the end are dropped pairwise and reported.

## Analysis (fixed now)
1. Δ[s,t] = mean over instances of score(s) − score(seed), 95% paired bootstrap CI (10k resamples). "Significant" = CI excludes 0.
2. Sign test: count components with at least one significantly positive and one significantly negative cell.
3. Model comparison on the 4×10 Δ matrix, by split-half cross-validation over instances (200 random halves; fit on one half, squared error on the other):
   - **G** global: Δ[s,t] = a_s
   - **H** hard groups: Δ[s,t] = a_s + b_{s,group(t)}, groups = the three dimension groups
   - **L1** low-rank: Δ[s,t] = a_s + u_s·v_t (rank 1), and **L2** rank 2
   - **S** saturated: Δ[s,t] fitted per cell
4. Reported per cell: Δ, CI, seed score, tokens per episode.

## Pass rule
A passes if **both** hold:
- (sign) ≥ 2 of the 4 components have cells of both significant signs;
- (structure) the best of L1/L2 has lower held-out error than G, H and S, and the task loadings v_t are not separated by group (some task's |v_t| ordering crosses a group boundary — i.e. H is beaten by L).

If only the sign part holds, the result is "per-task" (S best) or "grouped" (H best) and the fuzzy-hierarchy premise is weakened; we write that down before proceeding to B.

## Predictions (for the record, not used by the rule)
- `belief` helps the enumerable-hypothesis games (mastermind, wordle, gn, hangman) and is neutral or negative on alfworld/babyai.
- `brief` hurts mastermind/wordle (they need reasoning) and is neutral on alfworld.
- `check` helps dc and twentyq (premature accusation/guess is their common failure).
- `plan` helps sciworld and alfworld, little effect on the guessing games.

## Budget
~1,450 A episodes + ~130 seed episodes rerun with Jev for 20Q/SP/Guess Who. Two H100s, lowprio on ml-16-node-001; estimate 3–5 GPU-h. Jev < $1.
