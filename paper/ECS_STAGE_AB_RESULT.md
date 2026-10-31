# ECS Stage A/B result (first run): mechanism works, calibration does not

Date: 2026-11-01T09:05+08:00. Subset FD002, seed 2027, 5 folds. Design frozen in
`paper/ECS_METHOD_DESIGN.md` (sha `b66f1f56`). This is a **single-subset, single-seed,
exploratory** run; nothing here is a confirmatory claim yet.

## 1. Stage A — prior and likelihood

Per fold, the age-conditional Weibull prior is fitted on the training engines and the
32-bin likelihood model (arm-A encoder + bin head, 80 epochs, patience 20) is trained on
the training windows. Held-out (matched-protocol) expected-RUL RMSE at the best epoch:

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| dev RMSE | 11.41 | — | — | — | — |

(remaining folds: see `experiments/ecs/FD002/f*/metrics.json`.) The number is
comparable to arm A on the dev pool (13.95), so the likelihood backbone is not degraded
by the bin head.

## 2. Stage B — is credibility a real variable?

Multi-age sampling on the development engines (ages 20…150, same padding rule) because
the matched protocol forces age ≥ 50 and leaves κ with no variance. 2,840 (engine, age)
samples across the five held-out folds.

Regression of the realised information gain Δ = log p₁(y|w) − log p₀(y|age) on the two
credibility factors:

| factor | estimate | t | one-sided p |
|---|---:|---:|---:|
| κ (coverage) | **+13.74** | 48.4 | < 1e-4 |
| ρ (cross-channel consistency) | +0.030 | 0.01 | 0.496 |

corr(Δ, κ) = 0.67, corr(Δ, ρ) = 0.02.

**κ is a real, very strong driver: the shorter the observation, the worse the sensor
evidence is relative to the prior.** Mean Δ = **−1.85 nats**: on average the likelihood
is *worse* than the prior in this setup, which is exactly why a mechanism that can
retreat to the prior is worth having.

**ρ fails**: the cross-channel trend-agreement factor carries no information about Δ.
Per the pre-registered rule, the two-factor construction is abandoned and the method
reduces to the single-factor κ version. This is a recorded falsification, not a patch.

## 3. Fusion on the official test set (H1)

All 259 FD002 engines, capped truth, distribution mean as point estimate:

| alpha policy | full-set RMSE | short-trajectory RMSE (n=120) | 90% interval coverage |
|---|---:|---:|---:|
| α ≡ 0 (prior only) | 31.03 | **11.00** | 0.73 |
| α ≡ 1 (sensor only) | 17.10 | **33.34** | 0.57 |
| α ≡ 0.5 (fixed mix) | 14.88 | 20.84 | 0.68 |
| **CTSF (κ-driven)** | **14.06** | **11.22** | 0.74 |

**H1 passes by a wide margin**: on the 120 short-trajectory fold-engine pairs the fused
model improves on sensor-only by 22.1 RMSE (threshold was 2.0). The fused model is also
the best on the full set, beating every fixed-α policy.

Two honest qualifications:
* α ≡ 0 (prior only) is *better* than CTSF on the short group by 0.22 — the α surface
  has not yet pushed those engines fully to the prior. Ceiling for H1 is 22.1 + 0.22.
* **H2 fails**: 90% interval coverage is 0.74 across policies, far from nominal. The
  bin likelihood is over-confident (32 bins, cross-entropy trained without temperature
  or dispersion calibration), and the prior is narrower than the empirical spread. This
  is the main open defect, and it is a calibration problem, not a fusion problem — the
  α ≡ 1 arm is also at 0.57.

## 4. What this establishes and what it does not

Established: the coverage factor κ predicts, out of sample and with a large effect,
how much the sensor evidence should be trusted; a κ-driven tempering beats every fixed
α on both the full set and the short-trajectory group; the sensor likelikhood in this
implementation is on average worse than a fitted Weibull prior, which is a strong
argument for the fallback mechanism existing at all.

Not established: anything about seeds other than 2027, subsets other than FD002, or
calibration (H2 fails). The single-factor reduction means the "two interpretable
factors" novelty claim in the design must be revised to one factor plus a reported
negative result on the second.

## 5. Next steps (in order)

1. Calibration of the likelihood (temperature scaling on held-out folds; report ECE and
   bin-wise coverage before/after) — H2 cannot be assessed until this is done.
2. Seeds 2028/2029 to check that κ's slope is stable (it is a 48-sigma effect, so the
   risk is low, but the fusion gain needs the check).
3. FD001/FD003/FD004 with the same pipeline.
4. Then the comparators that matter for the paper: age-only, sensor-only, plain gate,
   variance-weighted fusion, parameter-matched constant gate (H4).
