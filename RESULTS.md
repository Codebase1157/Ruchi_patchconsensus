# What this release verifies

A record of what was actually run and confirmed in this environment (GPU, 
datasets, no pretrained weights), versus what requires the full Tier-B setup.

## Verified numerically (Tier A — ran here)

| Claim (paper) | How verified | Result |
| --- | --- | --- |
| **Theorem 1** error decomposition = `wᵀΣw` (Eq 18) | Monte-Carlo vs closed form, random weights | ✅ matches (rel. err < 2%) |
| **Theorem 1** uniform-weight error (Eq 19) | Monte-Carlo vs closed form | ✅ |
| **Theorem 1(c)** min-variance optimum `(1ᵀΣ⁻¹1)⁻¹` (Eq 15) | attained by `w*`, no feasible weighting beats it | ✅ |
| **Theorem 1(a)** floors (P=1 → σ₀², M=1 → σ_c,1²) | min error over weights vs floor | ✅ |
| **Proposition 2** disagreement energies (Eq 12) | Monte-Carlo vs closed form | ✅ |
| **Proposition 2a** three-cornered-hat recovery (Eq 13) | recover σ₀², σ_a,m², σ_c,m² from disagreement | ✅ (σ₀ rel.err < 5%) |
| **Proposition 2b/2c** non-identifiability (M=2, P=1) | estimator flags unidentified terms | ✅ |
| **Proposition 3 / Lemma 1** consensus ADMM → inverse-variance mean | exact ADMM run, residual → 0 | ✅ |
| **Lemma 1 / Eq 10** duals measure disagreement, ρ down-weights outlier | outlier source gets max ‖u‖, min ρ | ✅ |
| Assumptions **A1–A5** (Table VI) | measurement code on simulated data | ✅ A3 ≈ 0.093 (borderline, as in paper) |

`experiments/validate_theory.py`: **11/11 checks pass on every seed.**
`tests/`: **32 unit tests pass.**

## Demonstrated on synthetic data (Tier A smoke test)

Full inference pipeline (Algorithm 1) on a synthetic lesion with an ambiguous
border, with largest-connected-component cleanup applied equally to all variants:

| Variant | Dice | HD95 | Boundary IoU |
| --- | --- | --- | --- |
| Single encoder (M=1, P=1) | 91.6 | 17.5 | 15.2 |
| Uniform fusion | 97.1 | 4.5 | 39.3 |
| **PatchConsensus (min-variance)** | **97.2** | **3.6** | **42.6** |

Reproduces the paper's qualitative signature: fusion's gain concentrates in the
boundary/tail metrics (HD95, Boundary IoU), and reliability weighting beats
uniform. Feature-artifact energy `E_art` drops to 0.63× the single-encoder value.
A real, loadable checkpoint is written to `logs/checkpoints/smoke_student.npz`.

## NOT regenerated here (Tier B — needs A100 + data + weights)

The paper's benchmark numbers (HAM10000 Dice 94.0 / HD95 5.05; the +2.2 Dice /
−4.0 HD95 over EoSeg; Tables II, III, VII) require HAM10000 / ISIC 2018 / PH2 /
ISIC 2024, the DINOv3 / PanDerm / SAM 3 weights, and ≈ 130 + 2×45 GPU-hours.
The code path is complete (`scripts/train_teacher.py`, `distill_student.py`,
`evaluate.py`, `scripts/reproduce_all.sh full`); the paper's values are shipped
as reference logs under `logs/tables/` for diffing a real run against.
