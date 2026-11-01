# CIDO-RUL — review package

**Branch**: `cido-rul` (forked from `ecs-rul`; all earlier work preserved)
**Repository**: https://github.com/enterzxk/deepseekrul/tree/cido-rul
**Date**: 2026-11-01
**Status**: two theory gates run before any model training; **architecture-level novelty
claim is NO-GO**. The open question is which of two paths to take (§5).

---

## 1. What was proposed

A RUL architecture (`CIDO-RUL`) with three components:

1. **Condition-aware observation model** `y = g_φ(h, u) + b + ε`.
2. **Coherent innovation operator**: split the multi-sensor innovation into a shared
   degradation component and per-channel perturbations, yielding a correction direction,
   per-channel credibility and an effective-information scalar.
3. **Evidence-driven state transition**: the degradation rate is a filtered state, updated
   from the innovation, with first-passage time giving the RUL distribution.

The brief explicitly prohibited claiming novelty from stacking attention/Transformer/Mamba
blocks, and required an equivalence audit **before** implementation.

## 2. The audit (`paper/CIDO_NOVELTY_AUDIT.md`, sha `754b951a`)

~60 papers screened, 36 read at the level of state / observation / transition equations.
Verification tags are carried per entry; unverified items are excluded from load-bearing
claims.

**Result: two of the three components already exist.**

* The shared/specific split is classical factor analysis applied to fault diagnosis —
  Apley & Shi (2001, *Technometrics*, fault geometry vector), Dunia et al. (1996, *AIChE J.*,
  per-sensor validity index), Lock et al. (2013, *Ann. Appl. Stat.*, JIVE with explicit
  identifiability constraints), Kullaa (2011, *MSSP*, local-vs-global discrimination),
  Hajiyev (2025, *IJPHM*, recursive `R` estimation ⇒ automatic downweighting).
* The evidence-conditioned transition + rate-as-state + first-passage RUL is
  **Si, Wang, Hu, Chen & Zhou (2013, *MSSP* 35:219)**, which has the identical state space,
  a Kalman update of the rate from the innovation, and the exact first-passage density
  integrated over the rate's posterior. Also Dual EKF (Wan et al. 2000), Niu (2026, *RESS*),
  Wu et al. (2026, *AMM*).
* One argument in the original plan was **simply wrong** and has been deleted: "attention is
  a weighted sum while our decomposition is a projection". The GLS split estimate is also a
  weighted sum, `ŝ = (I + LᵀΨ⁻¹L)⁻¹LᵀΨ⁻¹ν`; the real differences are where the weights come
  from and the presence of a residual model.

**One genuine gap survived**: every first-passage RUL paper found assumes the drift is
exogenous (a deterministic function of time, or of an exogenous state). None makes the drift
a function of the filtering innovation. That is not cosmetic — it invalidates the
inverse-Gaussian closed form.

## 3. Gate F2 — first passage under a filtered drift: **PASS**
(`paper/CIDO_F2_RESULT.md`, sha `3c74bc75`; `scripts/cido/fpt_theory_check.py`)

8,000 Monte Carlo paths per cell, ten `τ_c/τ_f` cells, `h_τ = h_0 + ∫r_s ds + σW_τ` with
`r = exp(OU)`.

| `τ_c/τ_f` | KS mixture | KS effective-diffusion | KS plug-in | fitted coefficient |
|---:|---:|---:|---:|---:|
| 0.003 | 0.456 | **0.020** | 0.119 | 2.20 |
| 0.01 | 0.417 | **0.011** | 0.199 | 1.90 |
| 0.03 | 0.351 | **0.028** | 0.283 | 1.90 |
| 0.1 | 0.254 | 0.058 | 0.376 | 1.80 |
| 0.3 | 0.155 | 0.101 | 0.417 | 1.30 |
| 1.0 | 0.064 | 0.214 | 0.471 | 0.60 |
| 3.0 | **0.026** | 0.348 | 0.479 | 0.30 |
| 10 | **0.016** | 0.518 | 0.495 | 0.10 |
| 30 | **0.007** | 0.655 | 0.483 | — |
| 100 | **0.007** | 0.776 | 0.482 | — |

Claims:
1. The coefficient 2 in `σ_eff² = σ² + 2s²τ_c` is confirmed (fitted 1.90/1.90/2.20), i.e. no
   free parameter.
2. The two published families are the two asymptotes, crossing at `τ_c ≈ τ_f`.
3. **The plug-in (rate treated as known) is never accurate**: KS 0.12–0.50 everywhere.
4. A **gap band `τ_c/τ_f ∈ [0.1, 3]`** exists where neither endpoint reaches KS 0.05 (worst
   0.101). The RUL-relevant range falls inside it, so the first-passage readout must be
   numerical, not closed-form.

## 4. Gate F1 — identifiability of `h` against per-channel bias states: **FAILED**
(`paper/CIDO_F1_RESULT.md`, sha `a5ad8e64`; `scripts/cido/f1_identifiability.py`)

Proposition 1 claimed that an affine observation model makes `(h, b)` non-identifiable and a
nonlinear one restores identification. **The rank claim is true; the consequence is false.**

* F1a (rank of the accumulated observation information): affine smallest eigenvalue
  **−7.1e−11**, condition 5.4e36; nonlinear **1.84e+04**, condition 70.5. The null direction
  `(1, −L)` is real.
* F1b (consequence) was run four times. The first three were invalid, and the reasons matter:
  v1 injected a *common-mode* offset (not the confounded direction); v2 varied the initial
  prior (forgotten within 30 cycles); v3 varied the process-noise ratio (the filter had been
  handed the true rate, so `h` was pinned by the dynamics).
* v4 (rate 15 % too high) is valid and **reverses the prediction**: the affine model is
  twice as accurate, RMSE **0.0206 vs 0.0617**.

Why the proposition is wrong: (a) F1a accumulated only *observation* information, while the
trajectory system is identified by the **dynamics asymmetry** (`h` drifts, `b` does not) plus
the prior precision, so per-step rank deficiency does not imply trajectory
non-identifiability; (b) the null direction is one-dimensional inside a 14-dimensional bias
space, so "the split is arbitrary" does not mean "`h` is wrong"; (c) a nonlinear sensitivity
`aλe^{−λh}` decays as the engine degrades, making the nonlinear model worse exactly where
accuracy matters.

Per the pre-registered rule, the bias-state operator is demoted to an engineering component.

## 5. The decision this package is asking for

Three candidate operators have now been removed — two by the audit, one by F1. **No
architecture-level novelty claim currently stands.** What remains:

| asset | state | supports |
|---|---|---|
| Prop. 2 + gap band | verified numerically | a methodological contribution: how to compute the RUL distribution when the rate is driven by the observations, with both published closed forms located as asymptotes |
| numerical FPT sampler | designed, not built | vehicle for that contribution; also delivers calibrated intervals, which most deep RUL papers do not report |
| `encode()` / `build_windows(meta)` | built, bit-exact verified | infrastructure, no claim |

Two paths:

**A. Reframe around the first-passage result** — "how should the RUL distribution be computed
when the degradation rate is driven by the observations?" Prop. 2 answers what the FPT
literature assumes away; the gap band shows both published closed forms are wrong in the
RUL-relevant range; the fix (sample the joint process) is simple and nobody does it. This is
a measurement-methodology contribution and does not require a novel network.

**B. Stop the architecture line** and consolidate the existing measurement work
(E18–E22 + ECS) into a submission.

## 6. Specific questions for the reviewer

1. Is the F2 result (the gap band, and "the plug-in is never accurate") a sufficient
   contribution for IEEE TIM on its own, or is it a lemma in search of a paper?
2. Is the F1 negative result being interpreted correctly? In particular, is the claim
   "the trajectory system is identified by the dynamics asymmetry rather than by observation
   nonlinearity" right, or is there a regime (unknown rate, no drift assumption, different
   noise structure) where the confounding does bite?
3. Given that three operators have been removed, is continuing to search for a network-level
   operator the right use of effort, or is the base rate now evidence that this project's
   comparative advantage is measurement methodology rather than architecture?
4. If path A is chosen, what is the minimum set of experiments for TIM: calibration (PIT,
   interval coverage) on all four subsets plus one independent dataset, with the analytic
   laws as diagnostics — or does it need a full model comparison as well?

## 7. File index

| document | GitHub |
|---|---|
| literature / novelty audit | `paper/CIDO_NOVELTY_AUDIT.md` |
| method design (v2, post-audit) | `paper/CIDO_METHOD_DESIGN.md` |
| F2 gate result | `paper/CIDO_F2_RESULT.md` |
| F1 gate result | `paper/CIDO_F1_RESULT.md` |
| gate ledger + code state | `paper/CIDO_LEDGER.md` |
| F2 script | `scripts/cido/fpt_theory_check.py` |
| F1 script | `scripts/cido/f1_identifiability.py` |
| F2 raw result | `results/cido/fpt_theory_check.json` |
| F1 raw result | `results/cido/f1_identifiability.json` |

Reproduction: both scripts run standalone on a machine with numpy + scipy
(`python scripts/cido/fpt_theory_check.py 8000 0.05` and
`python scripts/cido/f1_identifiability.py 120`). Seeds are fixed, so the JSONs committed
under `results/cido/` are reproducible; runtime is a few minutes on CPU.
