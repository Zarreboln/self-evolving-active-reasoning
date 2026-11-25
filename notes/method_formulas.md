# Method formulas: follow the published ones (decided 2026-11-25)

Rule: every variant uses a published formula unchanged in form; only the objects change (gradients → evaluations, parameters → harness components). Deviations are listed at the end and each one is an ablation, not a silent change. Paper notes: `notes/papers_hier_meta/`.

Notation: component c ∈ library C; task t = (subtask, policy model); ℓ(t,c) = score on t after starting from seed ⊕ c and adapting k inner steps; Δ(t,c) = ℓ(t,c) − ℓ(t,seed).

## 1. Soft assignment — DPM (Jerfel et al. 2019, arXiv 1812.06080) E-step, Eq. 2
γ(t,c) = softmax_c( ℓ(t,c) / τ )
DPM's logit is the support log-likelihood after adapting from component ℓ; ours is the adapted score.
- τ → 0: hard argmax (= OSML's final per-layer argmax, arXiv 2010.11545 §3)
- τ → ∞: uniform (= one global harness)
- two-level: τ = ∞ for the global component, τ = 0 for the task-specific one

Comparison variant — HSML (Yao et al. 2019, arXiv 1905.05301) Eq. 6:
γ(t,c) = softmax_c( −‖(g_t − μ_c)/σ‖² / 2 ), g_t = task descriptor (dimension labels + first-probe results) instead of HSML's learned autoencoder.

## 2. Initialisation of a new task — HSML Eq. 8 (mask, not generate)
HSML: θ0_i = θ0 ∘ o_i.  Ours: h0(t) = seed ⊕ { c : γ(t,c) > η }.
HSML ablation A4c (Table 10): masking beats generating the initialisation (~4 pt) and warm-starts new clusters.

## 3. Hard / fixed structures — TreeMAML (arXiv 2103.04691)
- Fixed TreeMAML: tree from prior knowledge → our three dimension groups.
- Learned TreeMAML: online top-down clustering (Alg. 2, threshold ξ·σ_T), task feature = mean gradient (Eq. 8, Task2Vec) → ours: the gain vector Δ(t,·), cosine similarity.
- Degenerate cases = baselines: C_k = 1 for all k → global; C_k = #tasks → none (MAML); C_1 = 1, C_K = #tasks → two-level.

## 4. Outer update — DPM M-step
DPM: θ^(ℓ) ← θ^(ℓ) − β Σ_t γ(t,ℓ) ∇L_t.  Ours: an inner-loop edit e on task t is credited to component c with weight γ(t,c).
Promotion: Σ_t γ(t,c)·gain_t(e) beyond noise AND e recurs independently in ≥ 2 distinct benches (our addition).

## 5. Spawning a component — DPM Subroutine 3 + App. D
ρ_c = ℓ(t,c) + λ·log n_c ;  ρ_new = ℓ(t, local version) + log ζ ;  spawn if γ_new > ε = 0.95·T/(L+1); cool-down after a spawn with old components frozen (App. D.1); n_c over a sliding window of 5 (App. D.2).
Ours default λ = 0 (CRP prior is rich-get-richer under bench imbalance); λ = 1 is an ablation.

## 6. Replay against regression — OSML Alg. 1 line 8
A change to a shared component is evaluated on tasks currently using it (γ-weighted, ≥ 1 per bench); a drop beyond noise → fork instead of overwrite. Report forgetting as DPM does (mean drop on earlier benches after a new one is added).

## Deviations (each is an ablation)
- Inner loop is LLM code editing, not gradient steps → ℓ(t,c) is expensive. Two-stage: k = 0 insertion gain Δ (what experiment A measures) as cheap logit, full ℓ only for top-k components.
- λ = 0 instead of the CRP prior.
- Cross-bench recurrence for promotion (not in any of the papers).
- HSML Table 10: hierarchy vs flat gains mostly within CI — "fuzzy beats hard" must be shown here at equal budget, not cited.

## First use
After experiment A: plug its 4×10 Δ matrix in as ℓ(t,c) at k = 0, plot γ for several τ, and run Learned-TreeMAML clustering on Δ(t,·).
