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

## Amendment 1 (2026-11-25, after the first analysis, BEFORE running this test)
The original structure rule (low-rank must beat the saturated model S in held-out fit) tests the
wrong thing. S is accurate whenever each cell has enough data; its cost is that every new task
has to be measured on every component from scratch. What sharing buys is sample efficiency: from a
few episodes on a new task, predict how every component will do there. This is the "fast
adaptation" claim, so it is the test. The original verdict stays on record as computed.

Leave-one-task-out few-shot prediction (existing data, no new runs):
- Hold out task t. From the other 9 tasks fit a_s (mean gain per component) and the rank-r
  interaction factors U_r (SVD of the residual), r = 1, 2.
- On t, draw k paired instances (k = 1, 2, 3, 5, 8, 12) as the "adaptation" data; the remaining
  instances of t give the target Δ. 200 random draws per (t, k).
- Estimators of t's 4-vector of gains:
  - S_k: the per-cell means on the k instances (all 4 components measured: 5k episodes)
  - G: a_s (0 episodes)        - H: mean of t's dimension group on the 9 tasks (0 episodes)
  - L_r,k: a_s + U_r w, w by least squares on the k instances (5k episodes)
  - L1-probe,k: only seed + belief measured on k instances (2k episodes); w from that one
    component, the other three predicted
- Error: mean squared error against the target, averaged over tasks and draws; also reported
  without GN.
- Sharing pays off if L_r,k (or L1-probe,k) reaches a given error with fewer episodes than S_k.
