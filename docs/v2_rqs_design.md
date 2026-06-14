# v2 surrogate design: RQS coupling + intra-bunch standardization

## Deficiencies to fix (measured on v1)
- **D1 — witness longitudinal-phase-space (z–pz) over-thickening.** True intact-witness LPS is a
  razor-thin filament: thinness (minor/major PCA std ratio of standardized z,pz) ≈ 0.10–0.20,
  slice/projected energy spread ≈ 0.15. The v1 flow floors at ≈ 0.26–0.30 (0.10 → 0.30) — over-thick
  by 2–3×. Drive thinness (≈0.5) is reproduced fine (0.50 → 0.53).
- **D2 — witness per-coordinate dispersion / lower R².** Dispersion-ratio (σ_flow/σ_true): drive ≈1.0
  all coords; witness under-disperses y (~0.87), py (~0.82), wide pz. Witness ε_n,y R²=0.53, σ_E
  R²=0.55 (vs drive ε_n,x R²=0.98, inter-bunch R²>0.97, viability AUC 0.999 — all fine).

## Mechanism hypothesis (the critic should stress-test this)
The sampled cloud is `x_std = μ_k + L_k · w`, `w = flow(z)`, `z~N(0,I)`; `Σ_k = L_k L_kᵀ` (full Cholesky).
- **Thinness is governed mainly by the WHITENING Σ_k (linear covariance), not the flow.** A thin
  tilted filament is a near-rank-deficient Σ_k; if the whitening MLP predicted it accurately, the
  Gaussian `μ_k + L_k·w` would already be thin (the flow only adds non-Gaussian *shape*). The
  measured over-thickening ⇒ the predicted Σ_k is too round.
- **Why Σ_k is mispredicted (D1/D2):** v1's per-bunch scaler std is the std of *pooled raw* particles,
  which for the witness is dominated by **inter-sample centroid spread (~25× the intra-bunch size)**.
  So in standardized units a single witness bunch is tiny (intra-size ~0.04, thin direction ~0.004),
  and the whitening must predict a near-singular, strongly-anisotropic Σ_k close to the diag floor
  (1e-3) — a badly-conditioned regression that biases toward rounder Σ_k. This is the root of both
  D1 (thinness) and D2 (per-coord dispersion).
- **What the flow (RQS vs affine) is for:** after whitening removes the *linear* correlation, `w`
  carries the *non-Gaussian* residual (LPS curvature/banana, sharp caustic edges, multimodality).
  Affine coupling (bounded `tanh·0.5` scale, Gaussian-ish maps) smooths these; RQS (monotonic
  rational-quadratic splines) represents sharp/curved/near-degenerate structure far better.
- Note: full-covariance whitening already subsumes a linear "de-chirp", so no separate de-chirp stage.

## Change 1 — intra-bunch per-bunch standardization (preprocess)
- New scaler **std = std of pooled PER-SAMPLE-CENTERED particles** (subtract each density bunch's own
  6D centroid before pooling). Mean = pooled raw mean (unchanged placement frame). Raw clouds in the
  h5 are unchanged; only `*_norm.json` changes.
- Effect: standardized intra-bunch shape ~O(1) ⇒ `Σ_k`~O(1), well above the diag floor, and `μ_k` now
  regresses the (large, smooth) per-sample centroid instead of the flow squeezing to 0.04. Better
  conditioned for *both* whitening and flow. Expected to be the **primary** lever for D1/D2.
- Config: `preprocess --scaler {pooled,intrabunch}` (default `intrabunch`); re-run preprocess
  (raw clouds identical given the seed). v1 used `pooled`.

## Change 2 — RQS coupling (model)
- `TwoBunchFlow(coupling="affine"|"rqs")`; keep affine (v1, A/B). Whitening/heads/NLL/sampling
  unchanged (coupling is swapped behind `_forward_enc`/`_inverse_enc`).
- Hand-rolled `RQSCouplingLayer` with identical interface `forward/inverse(z,cond)->(z',logdet)`:
  split (z1,z2); conditioner `MLP(cat(z1,cond)) -> (3K-1) params per transformed dim` (K widths,
  K heights, K-1 internal derivatives); **unconstrained rational-quadratic spline** on `[-B,B]`,
  linear identity tails outside; analytic forward, inverse, and log|det|. K=8 bins, B=5 (whitened
  `w` is ~unit, so B=5 covers it). Reference hyperparams from the legacy normflows RQS
  (n_bins 8–24, tail_bound 3, linear tails).
- **Library decision:** hand-roll (no new dep) — repo philosophy is dependency-light hand-rolled
  coupling, torch 2.11 is new (old spline libs risk compat breakage), and the kernel is pinned by
  rigorous tests. Fallback: `pip install nflows` for just `unconstrained_rational_quadratic_spline`
  if the critic prefers a tested kernel.
- Floors kept: whitening diag floor 1e-3 (now non-binding: intra-bunch scaler puts the thin
  direction ~0.1 ≫ 1e-3), off-diag `5·tanh` bound.

## Expected impact
- Change 1 → fixes the bulk thinness/dispersion (better-conditioned Σ_k) — the main D1/D2 lever.
- Change 2 → fixes residual non-Gaussian shape (curvature, sharp edges) + general density fidelity.
- Combined target: witness LPS thinness flow→close to true (~0.12), witness ε_n,y/σ_E R² up, no
  regression in drive / inter-bunch / viability.

## Risks / open questions for the before-critic
1. Is the mechanism right — is thinness mostly a **whitening/scaler** problem (so Change 1 is the
   real fix and RQS is secondary for D1)? Could the bottleneck instead be whitening-MLP capacity, or
   NLL not rewarding thinness (it should: a fat model wastes probability mass on thin data)?
2. Does the intra-bunch scaler create a new problem — `μ_k` now spans a large range (witness centroid
   /intra-size up to ~50σ); can the MLP regress that, and does it hurt the drive (already fine)?
3. RQS hand-roll correctness (bin search, tails, inverse, log-det) — the main bug surface; what edge
   cases must the tests cover (bin boundaries, tail transitions, monotonicity, extreme params)?
4. Stability: RQS (unbounded expressivity) vs affine (bounded scale) — training instability / NaN risk?
5. Will these two changes actually move D1/D2, or is there a deeper limit (caustic edges fundamentally
   hard for continuous flows; finite P=1024; data)?

## Validation
RQS unit tests (forward∘inverse=id to 1e-5; autograd-logdet == analytic to 1e-5; tails identity;
monotonic). Affine path unchanged (existing tests pass). Train v2, compare to v1 on: witness LPS
thinness (true vs flow), dispersion_ratio, witness ε_n,x/y & σ_E R², viability AUC, all diagnostics.

---

## CRITIC REVISION (incorporated — before-critic ran numerical checks on the v1 checkpoint)

The critic's numerical decomposition corrected the mechanism and re-prioritized the levers:

- **Thinness is a pure second-moment (Σ_k) property.** Oracle test: a Gaussian with each bunch's
  *true* 2×2 z–pz covariance reproduces self-std thinness 0.155 vs true 0.154 — non-Gaussian shape
  adds ~nothing. ⇒ **RQS does NOT fix D1** (thinness); it is for heavy-tailed transverse density /
  general fidelity (D2-ish) only.
- **The diag-floor claim was FALSE** (witness L-diag is 0.3–2.0, 300× the floor; off-diag 1.49 < 5).
  The real fault: the whitening predicts a **near-constant, too-round Σ_k** — correlation between
  predicted and true per-bunch anisotropy ≈ **0.03** (ignores the knobs). Root = (a) pooled-std
  squeezes the witness Σ_k target to condition ~490 (a hard regression target) AND (b) the whitening
  head is a **single Linear layer** off a **shared** encoder, too low-capacity to track per-knob,
  per-bunch anisotropy. In v1 the affine flow claws thinness back from the round Σ_k (0.935→0.31) but
  stalls; the residual `w` is not isotropic (cond 22, pz mean −1.38, heavy tails).

**Revised change set (all config-gated for ablation):**
1. **Intra-bunch scaler** (as above) — improves the Σ_k target: witness z–pz condition 490→61,
   thin-direction scale 0.033→0.24. Primary conditioning fix. (μ_k burden ≈21–27σ on witness pz, not
   50; inter-bunch observables unaffected — de-standardized per bunch; minor drive re-tune, pz std ×0.39.)
2. **Whitening-head capacity** — replace the single `Linear` with a **per-bunch MLP** (separate head
   per k, ≥2 hidden layers) so drive-round/witness-thin no longer share one linear map and the head
   can track per-knob anisotropy. (Highest-leverage D1 fix per the critic.)
3. **Covariance-matching loss** `w_cov` — match the sampled-cloud per-bunch 6×6 covariance to truth
   (relative, standardized frame; reuse a beam-matrix term). NLL tolerates a 4×-too-round Σ_k; this
   directly rewards correct second moments (D1 thinness + D2 dispersion/σ_E).
4. **RQS coupling** `coupling="rqs"` — hand-rolled, **B=8, K=16**, linear tails. Reframed as a
   D2/density-fidelity improvement, NOT the D1 fix. Keep affine as the v1 baseline option.
   (After-critic: B must cover the whitened-residual tail. Measured oracle per-bunch-whitened
   residual on v2 data: p99.9 ≈ 4–7, max ≈ 7–17 per dim ⇒ B=8 covers p99.9 with adaptive bins
   concentrating resolution in the |w|<3 bulk; the ~0.05% beyond 8 is linear-tailed. The
   after-critic's "tail to 27" was an artifact of an under-fit affine-proxy whitening, not the
   real per-bunch-MLP whitening.)

Dropped: lowering the diag floor (not binding); separate de-chirp (full-cov whitening subsumes it);
more particles (oracle reproduced thinness at P=1024 — not the limiter).

**RQS test additions:** C¹ continuity at bin boundaries and at ±B; identity+logdet=0 in tails;
monotonicity on a dense grid; degenerate/extreme conditioner params (no NaN); reverse_mask parity.

**Revised likelihood (critic): MEDIUM** that v2 fixes D1 — scaler+capacity+cov-loss attack the right
lever (Σ_k), oracle ceiling (~true) is reachable; risk is whitening under-fit persisting if capacity
is still short. RQS credited for D2/density only.
