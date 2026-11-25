# Method formulas: follow the published ones (decided 2026-11-25)

Rule: every variant uses a published formula unchanged in form; only the objects change (gradients → evaluations, parameters → harness components). Deviations are listed at the end and each one is an ablation, not a silent change. Paper notes: `notes/papers_hier_meta/`; PDFs and pdftotext in `~/Desktop/self_evolve_ai/papers/`.

Every citation below was checked against the PDF text on 2026-11-25. Where a mapping is ours and not the paper's, it says **(ours)**.

Notation: component c ∈ library C; task t = (subtask, policy model); ℓ(t,c) = score on t after starting from seed ⊕ c and adapting k inner steps; Δ(t,c) = ℓ(t,c) − ℓ(t,seed).

## 1. Soft assignment — DPM (Jerfel et al., arXiv 1812.06080 v3)
Paper: Eq. 2, γ_j^(ℓ) ∝ p(x_j | φ̂_j^(ℓ)), the support likelihood after adapting from cluster ℓ; normalised by a τ-softmax in the E-step of Subroutines 2–3 (τ is an input of Alg. 1; τ = 1 in their miniImageNet runs, App. C.2.1).
Ours: γ(t,c) = softmax_c( ℓ(t,c) / τ ).
- τ → ∞: uniform assignment = one global harness. The paper states this limit itself: the uniform mixture "can also be seen as the non-uniform mixture in the infinite temperature (τ) limit" (§6).
- τ → 0: argmax = hard assignment **(ours, by the limit)**; matches OSML's final per-layer selection b*_l = argmax o_b (arXiv 2010.11545 §3.1).
- two-level: τ = ∞ for the global component, τ = 0 for the task-specific one **(ours)**.

Comparison variant — HSML (Yao et al., arXiv 1905.05301) Eq. 6, soft assignment by distance to learnable centres:
p = exp(−‖(h − c_k)/σ^l‖²/2) / Σ_k' exp(−‖(h − c_k')/σ^l‖²/2), σ^l a per-level scale.
Ours: one level, h = task descriptor (dimension labels + first-probe results) in place of HSML's learned autoencoder representation.

## 2. Initialisation of a new task — HSML, mask a shared initialisation
Paper: Eq. 8 defines the gate o_i = FC_σ(g_i ⊕ h_i^L) (sigmoid); the text right after it applies the gate as θ0_i = θ0 ∘ o_i.
Ours: h0(t) = seed ⊕ { c : γ(t,c) > η }.
Evidence (Table 10, App. H, 5-way 1-shot): A4c "learn the initialization directly" averages 45.19 vs HSML 49.35 (−4.2; Bird 53.95/60.98, Texture 32.35/35.01, Aircraft 52.15/57.38, Fungi 42.31/44.02). The paper's explanation is hedged: masking "may" constrain the optimisation space and "serve as the warm-start for a new cluster of tasks in continual learning".

## 3. Hard / fixed structures — TreeMAML (Garcia et al., arXiv 2103.04691, Adapt-NLP 2021)
- Fixed TreeMAML: tree given by prior knowledge, C_k determined by it (§3.1) → ours: the three dimension groups.
- Learned TreeMAML: tasks clustered on the mean loss gradient g_ik (Eq. 8, following Achille et al. 2019 / Task2Vec) with an online top-down algorithm (§3.2, Alg. 2) whose threshold ξ is a multiple of the std of intra-cluster similarities; cosine similarity in the synthetic experiment (§4.2) → ours: the gain vector Δ(t,·), cosine.
- Degenerate cases stated in §3.1: C_k = m (number of tasks) at all steps recovers MAML; C_k = 1 averages the gradient across all tasks. Ours: C_k = m → no sharing; C_k = 1 → global; C_1 = 1 then C_K = m → two-level **(ours)**.

## 4. Outer update — DPM M-step
Paper (Subroutine 3): return β ∇_θ [ Σ_{j,i} γ_j^(ℓ) log p(x_ji | φ̂_j^(ℓ)) + log n ], i.e. each cluster moves along the γ-weighted adapted log-likelihood.
Ours: an inner-loop edit e on task t is credited to component c with weight γ(t,c). Promotion: Σ_t γ(t,c)·gain_t(e) beyond noise AND e recurs independently in ≥ 2 distinct benches **(ours)**.

## 5. Spawning a component — DPM
Paper:
- Subroutine 3 E-step: ρ^(ℓ) = Σ_i log p(x_i | φ̂^(ℓ)) + log n (existing); ρ^(L+1) = Σ_i log p(x_i | φ̂^(L+1)) + log ζ (new, θ^(L+1) drawn from G0); γ = τ-softmax(ρ); spawn if γ^(L+1) > ε, else drop it and renormalise. CRP prior Eq. 3.
- ε = 0.95·T/(L+1), T = meta-batch size, L = non-empty clusters (App. C.2.2; same for miniImageNet, C.2.3).
- Cool-down after a spawn: no new spawns and existing clusters frozen for a fixed number of iterations (App. D.1).
- log n counted over a moving window of size 5 and multiplied by a tunable coefficient, because loss scale and log n differ by orders of magnitude (App. D.2).
- The paper notes the threshold "still preserves the preferential attachment ('rich-get-richer') dynamics" (App. D.3).
Ours: default λ = 0 on the log n term — rich-get-richer is exactly what bench imbalance must not get; λ = 1 is an ablation.

Paper inconsistencies to be aware of: the body calls the infinite-mixture steps Subroutine 3 and the algorithm Alg. 1, App. D.3 calls them "Subroutine 4" / "Algorithm 3"; App. D.1 says "task-aware" for both cool-down settings (whole phase, and 1k iterations) — one of them must mean task-agnostic.

## 6. Replay against regression — OSML
Paper: Alg. 1 line 8, "Sample task T_k that also use w0^{b*_l} from buffer B", to update each selected block (Eq. 6). OSML reports no forgetting metric.
Ours: a change to a shared component is evaluated on the tasks currently using it (γ-weighted, ≥ 1 per bench); a drop beyond noise → fork instead of overwrite.
Forgetting reported as DPM does: "the average drop in accuracy on the first two" datasets at the end of training (Fig. 8 caption) → ours: mean drop on earlier benches after new ones are added.

## Deviations (each is an ablation)
- Inner loop is LLM code editing, not gradient steps → ℓ(t,c) is expensive. Two-stage: k = 0 insertion gain Δ (what experiment A measures) as a cheap logit, full ℓ only for the top-k components.
- λ = 0 instead of the CRP prior.
- Cross-bench recurrence for promotion (in none of the papers).
- HSML's own Table 10: flat clustering (A3b) vs HSML differs by 2.90 / 0.75 / 1.27 / 0.64 points, while each entry's ± is 1.35–1.50; three of the four gaps are smaller than one ±. "Fuzzy beats hard" must therefore be shown here at equal budget, not cited.

## First use
After experiment A: plug its 4×10 Δ matrix in as ℓ(t,c) at k = 0, plot γ for several τ, and run Learned-TreeMAML clustering on Δ(t,·).
