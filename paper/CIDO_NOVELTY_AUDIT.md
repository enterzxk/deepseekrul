# CIDO-RUL: Architecture Novelty Audit

**Date**: 2026-11-01
**Purpose**: decide, before writing any model code, whether the two proposed core
operators are new. The rule applied throughout: **a structure counts as "already done"
if the computation is the same, regardless of the name used for it.**

**Method.** Four parallel literature/repository searches, ~60 candidate papers screened,
36 read at the level of state/observation/transition equations. Every entry below carries
a verification tag: `[V]` = title+author+DOI checked on the publisher page; `[S]` =
bibliographic entry checked, mechanism inferred from abstract/highlights only; `[U]` =
not verified. Where ScienceDirect paywalled the equations, the mechanism is marked `[S]`
even when the paper is otherwise solid.

---

## 0. Verdict in one paragraph

Both originally proposed operators are **already implemented in the literature, under
different names**: the shared/specific innovation split is classical factor analysis
applied to fault diagnosis (`Σ = ΛΛᵀ + Ψ`, Apley & Shi 2001, Dunia et al. 1996), and the
"evidence-conditioned transition with rate-as-state plus first-passage RUL" is
Si, Wang, Hu, Chen & Zhou (2013, MSSP), which has the *identical* state-space skeleton —
`h_k = h_{k−1} + θ_{k−1}Δ + σΔB`, `θ_k = θ_{k−1} + w_k`, `y_k = x_k + v_k`, Kalman update
of θ from the innovation, exact first-passage RUL density integrated over θ's posterior.
Proposing CIDO as described would be rejected on two citations.

Two things survive the audit and are carried into the redesign (§4): (i) the **h-vs-bias
identifiability argument**, which is available only if the observation model is nonlinear
in the health state, and (ii) the **first-passage law under a filtered (evidence-driven)
drift**, which no existing FPT paper covers because that community assumes the drift is
exogenous. Both are stated as propositions with falsification tests in
`CIDO_METHOD_DESIGN.md`.

---

## 1. Candidate 1 — "Coherent Degradation Innovation" (shared + specific split)

### 1.1 What was checked

Decomposition of an innovation vector `ν ∈ R^M` into a shared low-rank component and a
per-channel specific component, with per-channel credibility and an effective-information
scalar derived from the split.

### 1.2 Hits

| Work | Structure | What it already does | Tag |
|---|---|---|---|
| Apley & Shi 2001, *Technometrics* 43(1):84–95, `10.1198/00401700152404354` | `x = C v + w`, `v ~ N(0,I_p)`, `w ~ N(0,σ²I)`; `c_i` = fault geometry vector | shared fault sources + independent noise + sparse loading for identifiability + fault-direction interpretation | `[V]` |
| Dunia, Qin, Edgar & McAvoy 1996, *AIChE J.* 42(10):2797–2812, `10.1002/aic.690421011` | PCA model `x = Pt + e`, reconstruction along fault direction `Ξ_i` | **per-sensor validity index SVI ∈ [0,1]**, exactly the "per-channel credibility" output | `[V]` |
| Dunia & Qin 1998, *AIChE J.* 44(8) / *Comput. Chem. Eng.* | reconstruction-based identification | **reconstructability criterion** `ΞᵀC̃Ξ` nonsingular; two faults separable iff residual projections differ | `[S]` |
| Kullaa 2011, *MSSP* 25(8):2976–2989, `10.1016/j.ymssp.2011.05.017` | `E[u|v] = Kv`, GLRT per channel | distinguishes **sensor fault (local) from structural damage (global)** — precisely the failure mode CIDO must handle | `[V]` |
| Lock et al. 2013, *Ann. Appl. Stat.* 7(1):523–542, `10.1214/12-AOAS597` | `X_i = U_iV_iᵀ + W_iS_iᵀ + E_i` with `U_iᵀU_j = 0` | shared + individual decomposition **with identifiability constraints** | `[V]` |
| Hajiyev 2025, *IJPHM* 16(1), `10.36001/IJPHM.2025.v16i1.4204` | recursive estimate of `R̂_k` from `ν_k` | **fault ⇒ R inflates ⇒ K → 0 ⇒ channel auto-downweighted**; covers additive bias and multiplicative noise | `[V]` |
| Han, Zhang, Fu & Zhou 2021, ICLR (TMC) | per-view evidence `e_v` → Dirichlet → `u_v = K/S_v` | "per-channel credibility + scalar uncertainty" as an output layer | `[S]` |
| Candès et al. 2011, *J. ACM* 58(3), `10.1145/1970392.1970395` | `min ‖L‖_* + λ‖S‖₁` | low-rank + sparse split; fails on stuck-at and slow drift (sparsity violated) | `[V]` |

### 1.3 Verdict

**Covered.** Three independent traditions each already contain the mechanism: factor
analysis (the covariance structure), process monitoring (the per-channel validity index),
and adaptive Kalman filtering (the online downweighting). The specific computation
`ν → (shared projection, per-channel residual, precision scalar)` is, in exact arithmetic,
the GLS/因子-analysis posterior: `ŝ = (I + LᵀΨ⁻¹L)⁻¹LᵀΨ⁻¹ν`, with effective information
`I + LᵀΨ⁻¹L`. That is Fisher information, not a new quantity.

**One correction to the original framing.** The claim "attention is a weighted sum but
the split is a projection" is **false** and would be caught: the split's estimate is also
a weighted sum, `ŝ = Wν` with `W = (I+LᵀΨ⁻¹L)⁻¹LᵀΨ⁻¹`, just not constrained to the
simplex. The real differences are (a) attention weights are a function of the current
input, factor weights are a function of the noise covariance; (b) the split carries a
residual model and hence a test statistic, attention does not. Neither difference is a
contribution; both belong in related work.

---

## 2. Candidate 2 — "Evidence-Conditioned Degradation Transition"

### 2.1 What was checked

Non-negative degradation increment, rate-as-state (dual timescale), transition conditioned
on evidence strength, first-passage-time RUL distribution, no retraining for online update.

### 2.2 The decisive hit

**Si, Wang, Hu, Chen & Zhou (2013), *MSSP* 35(1–2):219–228, `10.1016/j.ymssp.2012.08.016`.**
State space: `x_k = x_{k−1} + θ_{k−1}Δ_k + σΔB_k`, `θ_k = θ_{k−1} + w_k`,
`y_k = x_k + v_k`. Kalman recursion updates `θ` **from the innovation**; RUL is the exact
first-passage density **integrated over θ's posterior**, explicitly contrasted by the
authors against plugging in a point estimate. Rate-as-state, innovation-driven update,
first-passage RUL, no retraining: four of the four elements, in 2013.

### 2.3 The rest of the hits

| Work | Element covered | Tag |
|---|---|---|
| Yu, Shao, Xu & Mechefske 2022, *RESS* 217:108099, `10.1016/j.ress.2021.108099` | generalized Wiener + adaptive drift + recursive Bayesian filter, closed-form RUL, no population data needed | `[V]` |
| Niu 2026, *RESS* 266(A):111666, `10.1016/j.ress.2025.111666` | **time-varying degradation rate + dual-timescale Kalman** + first-passage RUL PDF | `[V]` |
| Wu et al. 2026, *Appl. Math. Model.* 158:116771, `10.1016/j.apm.2026.116771` | bidirectional fast/slow coupling, rate depends on operating state, FPT RUL | `[V]` |
| Si et al. 2012, *IEEE Trans. Reliab.* 61(1):50–67, `10.1109/TR.2011.2182221` | nonlinear drift + closed-form FPT via time-space transform (the canonical reference) | `[V]` |
| Chen & Liu 2022, *RESS* 226:108651, `10.1016/j.ress.2022.108651` | `dX = f_LSTM(·)dt + σdB`, closed-form FPT PDF; online update by **fine-tuning**, not Bayesian recursion | `[S]` |
| Wan, van der Merwe & Nelson 2000, NeurIPS | **dual EKF** — fast state + slow parameter as random walk: the dual-timescale prototype | `[V]` |
| Mehra 1970, *IEEE TAC* 15(2):175–184, `10.1109/TAC.1970.1099422` | innovation covariance matching → rewrite `Q, R`; the original "innovation statistics modify the filter" | `[V]` |
| Revach et al. 2022, *IEEE TSP* 70:1532–1547, `10.1109/TSP.2022.3158588` | KalmanNet: GRU maps innovation features to the gain; `A` remains fixed | `[V]` |
| Li et al. 2023, *RESS* 235:109269, `10.1016/j.ress.2023.109269` | variational Bayesian joint inference of state + process-noise cov + **per-channel measurement-noise cov** | `[V]` |
| Roth, Özkan & Gustafsson 2013, ICASSP, `10.1109/ICASSP.2013.6638770` | Student-t process and measurement noise, closed-form recursion | `[V]` |
| Hafner et al. 2019/2020 (PlaNet/Dreamer) | latent transition conditioned on history summary (`h_t`) — "evidence changes the transition" is the default in deep SSMs | `[V]` |
| Xu, Peng & Chen 2021, arXiv:2106.02329 (DS³M) | **regime switch swaps the entire transition operator** — the strongest form of "evidence conditions the transition" | `[V]` |
| Chen, Ma, Zhao, Zhai & Mao 2022, *JDMD*, `10.37965/jdmd.2022.54` | `h_k = h_{k−1} + g_φ(x_k)`, `g_φ ≥ 0` — structurally non-negative increment | `[V]` |
| Li H. et al. 2026, *MSSP* 252 | multi-sensor CHI + FPT + Bayesian update, jointly trained weights/threshold | `[S]` |
| Zeng, Zhang, Zhang & Si 2026, *RESS* 275:112672, `10.1016/j.ress.2026.112672` | k-GNN CHI + Wiener, trained jointly (closed loop) | `[V]` |

### 2.4 Verdict

**Covered as a structure.** Every element exists, and three of the four pairwise
combinations exist. The combination is "known elements composed", which is exactly the
category IEEE TIM rejects.

There is, however, **one genuine computation gap**, and it is narrow but real: in *every*
first-passage RUL paper found, the drift is either a deterministic function of time
(`μ(τ;θ)`, `μ(t;LSTM)`) or a function of an **exogenous** state (`μ(t,s_t)`, operating
condition). **No paper makes the drift a function of the filtering innovation.** That is
not a cosmetic difference: once `μ_t = g(ν_t)λ_t` with `ν_t` being `F_t`-measurable, the
standard inverse-Gaussian first-passage density no longer applies, because the future
drift is a stochastic process rather than a fixed random variable. Stating
"we gate the drift by evidence" without confronting that fact would be a modelling slogan.

---

## 3. Repository mechanism evidence (why the problem is real)

From the code audit, three facts that motivate the architecture and constrain it:

1. **Age enters as an additive constant, after the temporal model.** `pooled = pooled +
   cycle_proj(cycle).unsqueeze(1)` (`src/candidates/model.py:336–337`) broadcasts a single
   vector over all 50 steps, then a LayerNorm partly removes it. It cannot modulate
   dynamics; it is a diluted offset.
2. **The window pipeline destroys the recurrence axis.** `build_windows` returns
   `(x, y, extras)` with no unit id (`scripts/run_candidate.py:292–365`); `WindowSet`
   shuffles. A state-space model cannot unfold over an engine's history without a change
   to this contract.
3. **The age channel saturates** at `min(c/250, 1)`, affecting 16.2% of FD003 and 14.0% of
   FD004 training windows, and evidence feature `e3` is a dead constant (0.9048) for every
   window with ≥5 early frames (`run_candidate.py:327–330`). First-passage models are
   sensitive to absolute age; a saturated channel is unusable.

These are the concrete reasons the problem "health evidence vs age prior" is not
already solved inside this codebase, independent of the literature.

---

## 4. What survives: two propositions and one operator

The audit leaves three things that are **not** in the literature as stated. They are
carried forward and each carries a falsification test.

### 4.1 Proposition 1 — the h-vs-bias confounding is broken *only* by a nonlinear observation model

If `y = Lh + b + ε` with `L` constant, the joint filter over `(h, b)` is rank-deficient:
`(h+δ, b−Lδ)` is observationally identical for every δ, so a common-mode sensor offset and
a health offset cannot be separated — a fact classical adaptive-`R` filtering inherits and
never escapes. Accumulating Fisher information over a trajectory,
`Σ_t [[g'(h_t)ᵀg'(h_t), g'(h_t)ᵀ],[g'(h_t), I]]`, leaves the null direction
`(1, −L)` intact when `g' ≡ L`. If `g` is nonlinear in `h`, the confounding direction
**rotates with the state**, and the only residual ambiguity is a bias component orthogonal
to the accumulated sensitivity subspace `span_t{g'(h_t)}` — which is genuinely unobservable
and should be reported as such.

**Consequence for design**: the observation model *must* be nonlinear in the health state,
and this is testable — a linear-`g_φ` ablation should exhibit drift of `b̂` along `g'` and
a biased `ĥ`, while the nonlinear version should not.

### 4.2 Proposition 2 — first passage under a filtered drift interpolates between two known laws

For `h_τ = h_0 + ∫_0^τ r_s ds + σW_τ` with `r_s` stationary, mean `μ`, variance `s²`,
correlation time `τ_c`, and `τ_f = (H−h_0)/μ`:

* `τ_c ≪ τ_f` → the FPT law converges to the inverse Gaussian with drift `μ` and
  **effective diffusion `σ_eff² = σ² + 2s²τ_c`** (rapid-fluctuation averaging);
* `τ_c ≫ τ_f` → converges to the **mixture of inverse Gaussians over `r ~ N(μ, s²)`**,
  which is exactly what Si 2013 / Chen & Liu 2022 compute.

Existing RUL work uses the second endpoint unconditionally. The correct object depends on
`τ_c/τ_f`, i.e. **on how far the engine is from failure** — which predicts that drift
uncertainty should inflate intervals near failure and *not* inflate them early in life,
the opposite of a uniform plug-in. This is a checkable numerical claim and is verified in
`scripts/cido/fpt_theory_check.py` before any network is trained.

### 4.3 Operator — the one that is not in the literature: bias states with their own dynamics

FDI tradition models faults as **hypotheses** (discrete, no continuous dynamics); factor
analysis models the specific component as **i.i.d. noise** (no state); deep SSMs put all
channels into one encoder and never ask which channel is wrong. The unattended middle is
a per-channel **bias state** with random-walk dynamics, filtered jointly with `h`, so that
`Cov(h_t, b_t) ≠ 0`. This changes attribution from threshold/direction matching to
posterior covariance, and it is falsifiable: inject a single-channel step offset; `b̂_i`
must absorb it while the increment of `ĥ` stays within noise.

---

## 5. Claims that are now forbidden

Written here so they cannot reappear in a draft:

| Claim | Status |
|---|---|
| "we decompose the innovation into shared degradation + sensor-specific perturbation" | **forbidden** — factor analysis / fault geometry (Apley & Shi 2001) |
| "per-channel credibility + effective information scalar" | **forbidden** — SVI (Dunia 1996); posterior precision is Fisher information |
| "sensor anomaly is auto-downweighted so it does not pollute the state" | **forbidden** — adaptive-`R` robust KF (Hajiyev 2025) |
| "the degradation rate is a state / dual timescale" | **forbidden** — Dual EKF (2000); Niu 2026 |
| "RUL from first passage time" | **forbidden** — Si 2012/2013; Chen & Liu 2022 |
| "attention is a weighted average but ours is a projection" | **forbidden and false** — both are weighted sums |
| "we combine shared/specific decomposition with first-passage RUL" | **weak; not sufficient** — composition of known elements |

---

## 6. Honest inventory of what would still be a composition

Even after the redesign, the model is a composition: a neural observation model, a
nonlinear filter, and a first-passage readout. The claim rests on the two propositions,
not on the composition. If Proposition 2 fails its numerical check, or if the
bias-state mechanism does not survive the single-channel-injection test, then the
architecture is `NO-GO` and this document says so in advance.

---

## Appendix: verification status of citations used above

Verified at publisher page (`[V]`): Apley & Shi 2001; Dunia et al. 1996; Kullaa 2011;
Lock 2013; Hajiyev 2025; Candès 2011; Si et al. 2012 (TR); Si et al. 2013 (MSSP);
Yu et al. 2022; Niu 2026; Wu 2026; Mehra 1970; Revach 2022; Li et al. 2023 (RESS);
Roth 2013; Xu 2021 (DS³M); Chen et al. 2022 (JDMD); Zeng 2026; Wan 2000; Hafner 2019/2020.

Bibliographic entry verified, mechanism inferred from abstract (`[S]`): Dunia & Qin 1998;
Han 2021 (TMC); Chen & Liu 2022; Li H. 2026 (MSSP); Wang et al. 2021 (RESS 210);
Bu et al. 2025 (RESS); Chen et al. 2026 (RESS 274, Tweedie).

Not verified (`[U]`, excluded from load-bearing claims): Zhuang et al. 2026
arXiv:2606.01894 (PC-MambaSDE, preprint); the 2025 *Appl. Soft. Comput.* state-space RNN
with monotonicity regulariser (PII only, no author/DOI); Discover Computing 2026
GAN-VAE paper (author list unavailable).
