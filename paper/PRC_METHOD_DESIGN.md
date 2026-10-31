# Protocol-Robust RUL Comparison (PRC): a method design, not a positioning note

Status: design document, 2026-11-01. Supersedes `POSITIONING_E22C.md` as the
methodological core. This document defines the algorithm, its statistical claims, the
controlled experiments that can falsify it, and the comparison set. The gate at the end
is binding: if PRC does not beat the baselines on coverage/FWER/power, the paper does
not claim a TIM methodological contribution.

## 1. What problem is being solved, exactly

Given two trained RUL models A and B, a finite, pre-specified set of evaluation
protocols 𝒫 = {p₁,…,p_K} that measure the **same** target quantity, and a compute
budget, decide one of three states:

* **A robustly better**: A beats B under *every* protocol in 𝒫, at a stated confidence.
* **B robustly better**.
* **Protocol-sensitive**: the sign of the difference depends on the protocol; no robust
  claim is licensed.

and, separately, recommend where the next unit of budget should go (engines, folds,
seeds, or protocols) to make the decision resolvable.

Two constraints define the scope and are not negotiable:

1. **Measurand grouping.** Capped-truth RMSE and uncapped-truth RMSE are different
   measurands, not two protocols for one measurand. 𝒫 may contain many *evaluation
   rules* (coverage, cutoff placement, window rule) applied to one measurand; it may not
   mix measurands. A comparison requested across measurands is reported as two
   decisions, never averaged.
2. **Correlation is the object of study, not a nuisance.** Engines are clustered within
   subsets/regimes; folds overlap (each engine trains 4 of 5 models); seeds share data;
   protocols are derived from the *same* per-engine predictions, so their estimates are
   strongly dependent. Any interval that treats these as independent is wrong, and the
   method's job is to be correct about them.

## 2. Estimand and decision rule

For protocol p, let δ_p = RMSE_B(p) − RMSE_A(p) (positive = A better). Define the
**protocol-robust advantage**

  θ = min_{p ∈ 𝒫} δ_p.

Decision, at family-wise level α:

* if LCB_α(θ) > 0 → **A robustly better**
* if UCB_α(θ) < 0 → **B robustly better**
* otherwise → **protocol-sensitive**, and the method reports the partition
  {p : δ_p > 0} vs {p : δ_p < 0} with per-protocol intervals.

The hard part is a *simultaneous* confidence band for (δ_p)_{p∈𝒫} that (i) respects
engine clustering, (ii) respects fold overlap, (iii) exploits — rather than ignores —
the dependence across protocols, and (iv) has verified coverage.

## 3. The estimator: engine-clustered, protocol-simultaneous multiplier bootstrap

### 3.1 Resampling scheme (two-level, cluster first)

For b = 1…B:

1. Draw engine weights w_i ~ Exp(1) (multiplier bootstrap) **within each subset**, so
   that engines are the independent units and regimes are stratified.
2. For every protocol p and every model m, recompute the per-protocol RMSE using the
   same weights: RMSE_m^*(p) = sqrt( Σ_i w_i e_{m,i}(p)² / Σ_i w_i ).
3. δ_p^* = RMSE_B^*(p) − RMSE_A^*(p).

Fold and seed variation enter through the per-engine prediction e_{m,i}(p), which is the
mean over the models in the ensemble (or a fixed single model); when both folds and
seeds are available, the per-engine prediction is itself an average, and its variance
component is estimated separately (§4) rather than resampled non-parametrically — the
multiplier bootstrap is on engines only, because engines are the exchangeable unit.

Why not resample folds: the five folds share 4/5 of their training engines, so
fold-level estimates are not independent draws; resampling them as if they were
understates variance. Fold heterogeneity is instead estimated and *added* to the
interval as a variance component (§4.2), with a sensitivity analysis for the
independence assumption.

### 3.2 Simultaneous band via max-t

Compute t_p^* = |δ_p^* − δ̂_p| / ŝe_p^*, where ŝe_p^* is the bootstrap SE of δ_p at
replicate level. Let q_{1−α} be the (1−α) quantile of max_{p∈𝒫} t_p^*. Then

  [δ̂_p − q_{1−α}·ŝe_p, δ̂_p + q_{1−α}·ŝe_p] for all p simultaneously,

which is narrower than Bonferroni whenever the protocols are positively dependent (they
are: same predictions, different derived labels), and wider than the naive per-protocol
interval exactly when it must be. θ's bounds follow conservatively as
[min_p L_p, max_p U_p]; a sharper θ interval by directly bootstrapping min_p δ_p^* is
computed as well and reported alongside (both are valid; the direct one is tighter and
is the primary, the band is the diagnostic that shows *which* protocol binds).

### 3.3 Variance components and the budget allocator

A three-level decomposition per protocol (engine / fold / seed) is estimated by nested
ANOVA on the per-cell estimates, giving V_e, V_f, V_s and the protocol covariance Σ_P.
The interval's variance for θ is then

  Var(θ̂) ≈ (1/N_e)·V_e + (1/(N_e·N_f))·V_f + (1/(N_e·N_f·N_s))·V_s + cross-terms,

and the allocator solves, under a cost model c_e·N_e + c_f·N_e·N_f + c_s·N_e·N_f·N_s ≤ C,
the discrete problem of maximising simulated power for the target margin. The closed-form
guidance (Neyman-type, n_k ∝ σ_k/√c_k, clipped to the nested constraints) is verified
against the simulator in §5.3 — the recommendation must reproduce the achieved power,
not merely look plausible.

## 4. What is new here, stated precisely (and what is borrowed)

| ingredient | provenance | our use |
|---|---|---|
| multiplier bootstrap, max-t simultaneous bands | standard (Chernozhukov et al.; Hothorn–Westfall) | applied to *protocol-indexed* estimands derived from one prediction set |
| cluster-robust inference | standard (Cameron–Gelbach–Miller; CR2) | engines as clusters; regimes as strata |
| nested variance decomposition, Neyman allocation | standard (experimental design) | extended to the *protocol* dimension with a cost model |
| measurand grouping + three-state decision | **new framing for RUL evaluation** | the decision rule and its error control |
| protocol transfer as a first-class estimand (θ = min_p δ_p with simultaneous guarantees) | **new for this literature** | replaces "report one protocol, hope it transfers" |
| budget allocator with protocol dimension under a nested cost model | **new combination** | validated against simulated power |

The honest claim: PRC is a **composition with a new estimand and a new decision rule**,
whose statistical guarantees are inherited but whose validity on this data structure
(clustered engines, overlapping folds, same-prediction protocols) is demonstrated, not
assumed. That is a legitimate TIM contribution only if §5 shows it beats the baselines.

## 5. Controlled experiments (the gate)

### 5.1 Known-truth simulator

Generate RUL predictions with a known structure: e_{m,i}(p) = b_m(p) + u_i + v_{f(i)} +
s_{m,seed} + ε, where b_m(p) is the protocol-dependent model bias (the estimand), u_i the
engine effect (sd σ_e), v_f the fold effect (σ_f), s the seed effect (σ_s), ε noise.
Grid: |δ| ∈ {0, 0.5, 1, 2, 4}; N_e ∈ {20, 50, 100, 260}; σ_e ∈ {0.5, 1, 2}·|δ|;
protocol heterogeneity (sign-stable vs sign-flipping); K ∈ {3, 6}. 2,000 replicates per
cell, seeded.

### 5.2 Baselines (all implemented on the same data)

1. **Per-protocol bootstrap + Bonferroni** (current practice).
2. **Cluster-robust sandwich (CR2) per protocol + Holm**.
3. **Bayesian hierarchical model** (engine random intercept, protocol random effect,
   posterior P(θ > 0) with a decision threshold).
4. *(stretch)* Distribution-free conformal-style bound on the per-engine loss
   difference.

### 5.3 Metrics and thresholds

* **Coverage** of the simultaneous band and of θ's interval at nominal 95%: must be
  within [0.93, 0.97] across the grid; below 0.93 is a fail.
* **FWER** for the three-state decision under H₀ (protocol-sensitive truth): ≤ 0.05
  nominal, measured, and reported per cell.
* **Power** under H₊ with |δ| = 1 and |δ| = 2: PRC must dominate baseline 1 by ≥10
  percentage points at equal N_e, and match or beat baseline 2.
* **Allocator validation**: for each recommended (N_e, N_f, N_s), the simulated power
  must land within ±5 points of the target.
* **Cost**: wall-clock per decision at N_e = 260, K = 6, B = 999.

### 5.4 Real-data validation

* FD001–FD004 with the frozen checkpoints (A/BE/PE × 5 folds × 3 seeds, plus the E22b
  z-score arms and Dual-Mixer). The paper reports the three-state decision for every
  model pair and every measurand, with the binding protocol named.
* **One independently collected degradation dataset** (not C-MAPSS, not simulated):
  candidate PRONOSTIA/FEMTO bearing set or XJTU-SY, trained with the same pipeline and
  analysed with PRC. The claim to be tested: the decision rule transfers, and the
  binding protocol differs from C-MAPSS's — which is the point of a protocol-robust
  method.

## 6. Falsification gates (written before the experiments)

* If PRC's coverage falls outside [0.93, 0.97] on more than one grid cell, the method is
  withdrawn, not patched.
* If PRC does not beat baseline 1 by ≥10 points of power at equal cost, and does not beat
  baseline 2 on at least one axis (coverage, FWER, or cost), the paper drops the "new
  method" claim and reverts to an empirical-audit paper — submitted elsewhere than TIM.
* If the allocator's recommendation misses the achieved power by >5 points in more than
  one cell, the allocator is reported as heuristic, not as a validated design rule.

## 7. Deliverables and order of work

1. `scripts/prc/simulator.py` — known-truth generator (§5.1).
2. `scripts/prc/prc.py` — estimator, band, three-state decision, allocator.
3. `scripts/prc/baselines.py` — the three baselines.
4. `results/prc/coverage.json`, `power.json`, `allocator.json` — the gate evidence.
5. Real-data run: C-MAPSS decisions (§5.4) + the independent dataset.
6. Only then: `main.tex`, with PRC as the method and the E-series as its calibration.

## 8. First validation round (2026-11-01T01:10+08:00) — status: NOT PASSED, three defects found

The coverage gate was run on the known-truth simulator (`scripts/prc/exp_coverage_v1.py`,
80 replicates per cell, 999 bootstrap reps). Current state, stated plainly:

| scenario | n | band coverage | theta coverage | decision behaviour |
|---|---:|---:|---:|---|
| H0 (no effect) | 100 | 0.94 | 0.96 | 100% protocol-sensitive (correct) |
| H0 (no effect) | 260 | 0.94 | 0.99 | 100% protocol-sensitive (correct) |
| H+ clear (bias 2.0/3.0/1.5/2.5) | 100 | 0.78 | 0.90 | 85% A robust |
| H+ clear | 260 | 0.57 | 0.82 | 95% A robust |
| H+ small (bias 0.5/1.0/0.3/0.7) | 100 | 0.85 | 0.95 | 100% sensitive (underpowered, expected) |

**Defect 1 (fixed).** The decision rule used min(delta)'s upper bound for the
"B robust" branch. min < 0 says only that A is not uniformly better; "B robust"
requires max(delta)'s upper bound < 0. The first version produced 55–63% false "B
robust" verdicts under H0. Fixed; H0 false-positive rate is now 0.

**Defect 2 (fixed, and a methodological point).** The simulator mixed unconditional and
conditional inference: seed-level variation was injected as a constant offset while the
bootstrap resampled engines only, so the SE was systematically too small. The design
now separates the two explicitly — conditional on the trained models (engines resampled
only) vs unconditional over training randomness (nested resampling required). This
distinction is itself part of the method and is the first thing a reviewer will ask
about.

**Defect 3 (OPEN — the gate is not met).** The symmetric max-t band under-covers when
the effect is large (0.57 at n=260), because the sampling distribution of an RMSE
difference is skewed once the bias is a sizeable fraction of the noise scale. Symmetric
bands assume away exactly the skew that is present. Candidate fixes, in order of
expected effectiveness: (i) percentile-t band with a nested (double) bootstrap to
studentise each replicate; (ii) BCa-style bias correction on the simultaneous quantile;
(iii) a log-variance stabilising transform on the ratio RMSE_B/RMSE_A. Until one of
these clears 0.93–0.97 across the grid, **PRC is not claimed as a validated method** and
the falsification gate in §6 applies: no TIM "new method" claim.

What this round does establish: the three-state decision rule is *correct* (H0 is never
declared a robust win), the conditional/unconditional distinction is necessary and
non-obvious, and the failure mode of the naive symmetric band is now measured rather
than assumed.
