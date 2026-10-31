# ECS-RUL method design: credibility-tempered survival fusion (CTSF)

Status: design frozen 2026-11-01, before any ECS experiment. Adjustments after
experiments are recorded as amendments, never as silent edits.

## 0. Problem, stated as the repository forces it

On FD002 the same model scores 11.96 RMSE on the 235 engines with ≥50 observed cycles
and 16.66 on all 259. The 24 extra engines have observation coverage c ∈ [0.084, 0.196]
while every training window had c ≥ 0.2; the age term in arm A is an additive bias whose
magnitude is proportional to c, so those 24 engines are pure extrapolation. Back-solving
the published numbers gives their RMSE as 39.9 (A), 26.8 (BE), 12.0 (PE) — and simply
predicting the cap scores 12.41. **When the sensor evidence is weak, the only thing that
still works is the lifetime prior.** The method below is built around that fact instead
of around a bigger network.

Formal question: given a partially observed trajectory w_{1:L} of one engine under
operating condition(s) g, and its age a = L cycles survived, produce p(R | w_{1:L}, a)
such that

* the point estimate is accurate when evidence is strong,
* the distribution **falls back to the age-conditional lifetime prior when evidence is
  weak**, and
* the stated intervals are calibrated *conditionally* on coverage and on life stage,
  not merely on average.

## 1. Model

### 1.1 Age-conditional survival prior

Let T be the lifetime (cycles to failure). Estimate S(t) = P(T > t) from the training
fleet, per subset (Weibull MLE; Kaplan–Meier as a non-parametric check). The RUL prior
given survival to age a is the truncated conditional distribution

  p₀(R | a) = p_T(a + R) / S(a),   R ≥ 0,

with survival function S₀(R | a) = S(a + R)/S(a). This is the "legal, observable age"
component: it uses only the fact that the engine has survived a cycles, never the true
lifetime.

### 1.2 Sensor evidence

Two ingredients, both computed on condition-normalised inputs (the existing
`OperatingConditionScaler` for FD002/FD004; min-max for FD001/FD003, held fixed as the
project's registered preprocessing):

* a **health summary** h(w) ∈ R^d from the 50-cycle window (the existing channel-
  independent encoder, unchanged, so the comparison isolates the fusion);
* a **conditional density model** q_φ(w | T) over the window given a candidate lifetime,
  implemented as a discretised mixture over RUL bins (K = 32 bins) trained by
  cross-entropy. This yields the likelihood term of the fusion; its normalisation is
  immaterial because only ratios matter.

The likelihood used in the update is L(R) = q_φ(w | R bin).

### 1.3 Evidence credibility α — the component that must earn its place

α ∈ [0, 1] is *not* a learned gate on features. It is an estimate of **how much
information the sensor evidence carries beyond the age prior**, computed by
cross-fitted predictive information gain:

1. Split the training engines into K folds (engine-level, the project's split seeds).
2. For fold k, fit the likelihood model on the other folds; on fold k compute, per
   engine, the log predictive density of the realised RUL under
   (a) the prior alone, ℓ₀ = log p₀(R | a), and (b) the fused model at α = 1,
   ℓ₁ = log p₁(R | w, a).
3. The per-engine gain is Δ = ℓ₁ − ℓ₀ (a log-Bayes-factor-like quantity). Positive Δ
   means the sensors beat the prior for that engine.
4. Regress Δ on two **interpretable, separately measurable** credibility factors:
   * **coverage** κ = min(1, L / L*) with L* = 50 (the window length; the fraction of
     the window that is real rather than padded),
   * **consistency** ρ = agreement of degradation trends across the 14 channels
     (1 − normalised dispersion of per-channel trend slopes, clipped to [0,1]).
   The regression is a monotone (isotonic) map Δ̂ = f(κ, ρ), fitted on held-out folds,
   and α = σ(f(κ, ρ)) with a single temperature. Report the fitted surface; if Δ does
   not depend on κ or ρ, **the credibility hypothesis is false and the method reduces
   to α = constant** — that is a pre-registered falsification, not a failure to hide.

Why this and not a gate: a gate minimises training loss; α here estimates a quantity
with a definition (information gain), is fitted on held-out folds, and can be checked
against the realised gain. The two-factor decomposition is what the literature has not
done (review §4): existing work uses a single mixed similarity, a sample entropy, or an
inverse variance.

### 1.4 Fusion: tempered likelihood

  p(R | w, a) ∝ p₀(R | a) · L(R)^α,    α = α(κ, ρ; θ).

This is a **power prior with a data-estimated, time-varying exponent** (Ibrahim & Chen
2000; Hobbs 2012 for conflict-driven borrowing). The paper states this lineage in the
first paragraph of Related Work. Endpoints are mandatory ablations: α ≡ 0 (prior only),
α ≡ 1 (sensor only), α constant (fitted scalar), α from a plain feature gate.

Outputs: posterior median (point), 5–95% interval, and failure risk
P(R ≤ r | w, a) = F(r) for any horizon r.

### 1.5 Training and inference

* Stage A: fit S(t) (closed form) and q_φ (cross-entropy, same encoder as arm A, same
  budget: 80 epochs, patience 20, 3 seeds).
* Stage B: cross-fitted Δ on the training folds; fit the isotonic surface f(κ, ρ);
  freeze.
* Inference: one forward pass for L(R), one closed-form prior, one scalar α; the fusion
  is a 32-bin reweighting — negligible cost. Report parameters and latency.

## 2. Pre-registered hypotheses (each falsifiable)

H1 **Fallback**: on engines with κ < 0.4, the fused model's RMSE is lower than α ≡ 1
   by ≥ 2.0 (the short-trajectory gap is 39.9 vs an oracle 12.4, so a well-behaved
   fallback has room). Falsified if the gap is < 1.0.
H2 **Conditional calibration**: bin-wise coverage (RUL quartiles × coverage terciles)
   of the 90% interval is within [0.85, 0.95] in every bin, and the worst-bin
   miscoverage is smaller than for variance-weighted fusion (Srinivasan-style) and for
   α ≡ 1. Falsified if any comparator matches it.
H3 **Credibility is real**: Δ (realised information gain) increases with κ and ρ on
   held-out folds (one-sided test, α = 0.05). Falsified if either slope is ≤ 0 —
   in which case the two-factor construction is abandoned and reported as such.
H4 **No free lunch from parameters**: against a parameter-matched constant gate, the
   fused model's advantage in H1/H2 survives (the credibility mechanism, not the extra
   width, does the work).

## 3. Experimental plan

### 3.1 Data

* FD001–FD004, all 707 official test engines (frozen padding rule from E22a; the
  coverage factor makes padding explicit rather than hidden).
* N-CMAPSS (DS02) for a second regime structure — train and evaluate within the
  simulated fleet, no cross-device claims.
* One independently collected degradation dataset (XJTU-SY or PRONOSTIA), trained
  separately; no cross-device transfer claimed.

Split discipline: every confirmatory number comes from engines never used for fitting
(fold-level CV on training engines; official test read once after freeze). The existing
E21/E22a test numbers are already observed and are labelled as such — they are used for
the *comparison table*, and the confirmatory evidence for the mechanism is the
cross-fitted gain (H3) plus the short-trajectory and conditional-coverage results on
held-out folds.

### 3.2 Comparators (all on identical inputs, splits, budget)

1. arm A (existing, cycle channel), 2. arm PE (existing), 3. plain regression with the
same backbone (no prior), 4. **age prior alone** (S₀ only), 5. **sensor evidence alone**
(α ≡ 1), 6. **plain concatenation** (prior summary features appended to the encoder
input), 7. **plain feature gate** (sigmoid of a small MLP on κ, ρ and evidence),
8. **variance-weighted fusion** (Srinivasan-style: weights ∝ 1/σ² of the two
components), 9. a recent published model with public code if one is found to run under
the same budget (Dual-Mixer, already reproduced in E22b, as the reference point).

### 3.3 Metrics

Capped RMSE and NASA score (both, with the convention stated); uncapped error against
true life; NLL, CRPS, and interval coverage (marginal and bin-wise conditional);
grouped error by coverage tercile, RUL quartile, and degradation stage; robustness under
condition shift, sensor noise, sensor dropout and sparse sampling; stability of the α
surface as observations grow; parameter count and latency.

### 3.4 Mandatory ablations

α ≡ 0 / α ≡ 1 / constant α / feature gate / variance weighting (all in §3.2); remove
the consistency factor; remove the coverage factor; replace the isotonic surface with a
linear map; remove the prior (sensor-only). Each result reported with the same seeds.

### 3.5 Mechanism controls inherited from the audit

C1: fix the coverage coordinate (c = min(L_pad/250, 1)) to eliminate extrapolation.
C2: change the padding shape (last-frame or constant instead of first-frame).
C3: training-time padding augmentation. Each isolates padding artefacts from the
mechanism; if the method's gain vanishes under C1–C3, it was a padding artefact.

## 4. GO / NO-GO

**GO** requires all of: H1 (fallback gain ≥ 2.0 on low-coverage engines), H2
(bin-wise coverage competitive or better than variance weighting), H3 (both credibility
slopes positive at α = 0.05), H4 (gain survives parameter matching), and no criterion
of §3.3 deteriorating materially on the full test set. Any one failure → NO-GO for the
mechanism, reported with the failing number.

**NO-GO** does not kill the project: the empirical results (short-trajectory
decomposition, coverage audit, conditional-coverage evidence) stand on their own as an
instrumentation paper, and the failure is documented in the ledger.

## 5. Deliverables

`src/ecs/` (prior, likelihood, credibility, fusion, calibration), `scripts/ecs_train.py`,
`scripts/ecs_eval.py`, `paper/ECS_LEDGER.md` (pre-registration hashes, every result,
including failures), and the ablation/robustness/calibration tables. Independent mock
review before any submission decision.
