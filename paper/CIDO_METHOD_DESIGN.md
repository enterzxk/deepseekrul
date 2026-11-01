# CIDO-RUL: Method Design (v2, post-audit)

**Branch**: `cido-rul`, forked from `ecs-rul`
**Prerequisite**: `paper/CIDO_NOVELTY_AUDIT.md` — two of the three originally proposed
operators are withdrawn as novel claims; this document designs around what survived.

---

## 1. Problem statement

Given an observation prefix `(y_{1:t}, u_{1:t}, m_{1:t})` of one engine — `M = 14` sensor
channels, `u` = three operating settings, `m` = observation mask — produce the conditional
distribution of remaining useful life `p(R | y_{1:t}, u_{1:t}, m_{1:t})`, such that:

* the distribution is **calibrated** (not merely a point estimate with an error bar);
* the estimate is **updated online from new observations without retraining**;
* when the sensor evidence is uninformative or a channel is faulty, the prediction
  **degrades gracefully toward the age-conditional prior** instead of being corrupted;
* the mechanism producing each of these behaviours is **identifiable and measurable**,
  not asserted.

The scientific question is not "can a bigger network lower RMSE". It is: *which part of a
degradation trajectory is evidence about the future, and how should the model's belief
change when that evidence is weak, contradictory, or stale?*

---

## 2. State, observation, transition

### 2.1 Latent state

$$z_t = (h_t,\ \rho_t,\ b_t),\qquad h_t\in\mathbb{R}_{\ge 0},\ \rho_t\in\mathbb{R},\ b_t\in\mathbb{R}^M$$

| symbol | meaning | role |
|---|---|---|
| `h_t` | degradation level in a normalized health coordinate, failure at `h = H_f = 1` | the quantity whose first passage defines RUL |
| `ρ_t = log r_t` | log degradation **rate**; `r_t > 0` by construction | second timescale; updated by evidence |
| `b_t` | per-channel bias state (sensor drift / offset) | absorbs channel-specific corruption instead of letting it enter `h` |

`b_t` is the operator that the audit found unattended (§4.3 there): FDI treats faults as
discrete hypotheses, factor analysis treats the specific term as i.i.d. noise, deep SSMs do
not model channels individually. Here `b_t` is a **state with its own dynamics**, filtered
jointly with `h_t`, so `Cov(h_t, b_t) ≠ 0` is available for attribution.

### 2.2 Transition (per cycle, `Δt = 1`)

$$
\begin{aligned}
h_{t+1} &= h_t + r_t\,\Delta t \;+\; \sigma_h\sqrt{\Delta t}\;\varepsilon_h,
&\varepsilon_h &\sim \mathcal N(0,1)\\
\rho_{t+1} &= \rho_t + \kappa(\mu_\rho-\rho_t)\Delta t \;+\; \sigma_\rho\sqrt{\Delta t}\;\varepsilon_\rho,
&\varepsilon_\rho &\sim \mathcal N(0,1)\\
b_{t+1} &= b_t + \sigma_b\sqrt{\Delta t}\;\varepsilon_b,
&\varepsilon_b &\sim \mathcal N(0,I_M)
\end{aligned}
$$

`ρ` follows a mean-reverting log-rate (so `r > 0` always, and the rate cannot run away);
`b` is a slow random walk — **slow relative to `h`**, which is what makes persistence the
discriminating statistic between "sensor drift" and "transient noise" (§3.3).

**Non-negative increments.** The drift term `r_t Δt` is non-negative by construction of
`r_t`; the diffusion term is the process noise of the health coordinate, not a physical
reversal. A strictly monotone variant
(`Δh = rΔt·G`, `G ~ Gamma(k, k)`, mean 1) is carried as an **ablation arm**, not the
default, because C-MAPSS health trajectories themselves are non-monotone at the sensor
level and a hard monotone constraint transfers the noise into the rate estimate.

### 2.3 Observation model (condition-aware)

$$
y_t = g_\phi(h_t, u_t) + b_t + \varepsilon_t,\qquad
\varepsilon_t\sim\mathcal N(0,\mathrm{diag}(\psi)),\quad
y_t\in\mathbb R^M
$$

`g_φ : R × R^3 → R^M` is a small MLP. **`g_φ` must be nonlinear in `h`** — this is not a
style choice, it is Proposition 1 (§4.1): with `g_φ` affine in `h`, the pair `(h, b)` is
rank-deficient and no filter can tell a common-mode sensor offset from a health offset.

The channel Jacobian

$$J_t \;=\; \frac{\partial g_\phi}{\partial h}\Big|_{h_t,u_t}\in\mathbb R^{M}$$

is the **state- and condition-dependent sensitivity vector**. It plays the role that a
loading matrix plays in factor analysis, with one decisive difference: it is **not a free
parameter**. Its value is pinned by the observation model, so the rotation/scale
non-identifiability of factor analysis never arises. This is the entire reason the
identifiability argument in §4.1 is available here and is not available to JIVE-type
decompositions.

---

## 3. The three operators

### 3.1 Operator A — condition-aware observation residual

Given the one-step prediction `(ĥ_{t|t−1}, b̂_{t|t−1})`:

$$\nu_t = y_t - g_\phi(\hat h_{t|t-1}, u_t) - \hat b_{t|t-1}\in\mathbb R^M$$

$$S_t = J_t P^h_t J_t^{\top} + P^b_t + \mathrm{diag}(\psi)\in\mathbb R^{M\times M}$$

with `P^h_t = Var(h_{t|t−1})`, `P^b_t = Cov(b_{t|t−1})`. Unlike a fixed noise covariance,
`S_t` has rank-one structure from `J_t P^h_t J_tᵀ` that changes with the state, and a
full-matrix block from `P^b_t` that changes with how much bias has been absorbed.

### 3.2 Operator B — Jacobian-anchored innovation splitting

Scalar GLS projection onto the sensitivity direction:

$$s_t = \frac{J_t^{\top}S_t^{-1}\nu_t}{J_t^{\top}S_t^{-1}J_t},
\qquad I_t = J_t^{\top}S_t^{-1}J_t \quad(\text{effective information about }h)$$

Orthogonal (specific) residual and its standardized form:

$$e_t = \nu_t - J_t s_t,\qquad
\zeta_{t,i} = \frac{e_{t,i}}{\sqrt{S_{t,ii} - J_{t,i}^2/I_t}}$$

**What is new here is not the split** (that is GLS, and the audit forbids claiming it);
it is that `J_t` is a *derivative of a learned observation model*, so:

* `I_t` is a genuine physical quantity — how many "sensor-equivalents" of information the
  current state is worth — and it changes with operating condition through `u_t`;
* the direction along which `h` is corrected, `J_t/I_t`, rotates as `h` moves, which is
  what breaks the `h`–`b` confounding (Prop. 1).

### 3.3 Operator C — persistence-based attribution and evidence-driven transition

The credibility of channel `i` is **not** a function of `|ζ_{t,i}|` alone (that is SVI, and
it is forbidden as a claim). It is a function of **persistence**, because that is the
statistic that separates the two failure modes:

$$\bar\zeta_{t,i} = (1-\lambda)\bar\zeta_{t-1,i} + \lambda\,\zeta_{t,i},
\qquad
c_{t,i} = \frac{1}{1 + \big(\bar\zeta_{t,i}/\tau\big)^2}$$

* a **transient** spike leaves `ζ̄` near zero → `c ≈ 1`, the sample is used but its weight
  in `S_t` already reflects the outlier through a robust (Huber/Student-t) update;
* a **persistent** offset drives `ζ̄` away from zero → `c → 0`, and the corresponding `b_i`
  absorbs the offset. The channel is not "thrown away"; it is **re-attributed**, which is
  what preserves observability (the audit's reconstructability objection:
  downweighting a channel that is the sole carrier of a degradation mode makes that mode
  unobservable — absorbing it into `b_i` does not).

The evidence-driven transition then modulates the **rate**, not the output:

$$\rho_{t+1|t} = \rho_{t|t-1} + \kappa(\mu_\rho - \rho_{t|t-1})\Delta t
\;+\; \underbrace{\gamma\,\frac{\sum_i c_{t,i}\,\zeta_{t,i}}{\sum_i c_{t,i}}}_{\text{shared evidence}}$$

with `γ` learned. The second term is the audit's "gap": the drift is a function of the
filtering innovation, which makes the future drift a stochastic process rather than a fixed
random variable — see §4.2 for what that costs and buys.

### 3.4 Update (one step, Laplace/UKF hybrid)

`h` and `ρ` are updated by a sigma-point step on the two-dimensional nonlinear subspace
(`h` enters `g_φ` nonlinearly; `ρ` enters only the transition); `b` is updated by the
linear-Gaussian conditional given `(h, ρ)`, which is exact because `b` enters the
observation linearly. This keeps the filter O(M) per step and avoids particle
degeneracy, while preserving the exact bias posterior.

---

## 4. Two propositions, each with a falsification test

### 4.1 Proposition 1 — nonlinearity of `g_φ` is necessary for `h`/`b` identifiability

**Statement.** For `y_t = g(h_t) + b_t + ε_t`, accumulate Fisher information over a
trajectory:
$$\mathcal I_{1:T} = \frac{1}{\psi}\sum_{t=1}^{T}
\begin{pmatrix} g'(h_t)^\top g'(h_t) & g'(h_t)^\top\\ g'(h_t) & I_M\end{pmatrix}.$$
If `g' ≡ L` constant, the vector `(1, −L)` lies in the null space, so `(h + δ, b − Lδ)` is
observationally equivalent for every `δ` and the filter is non-identifiable. If `g'` varies
with `h`, the confounding direction rotates; the residual null space is
`{(0, c) : c \perp span_t{g'(h_t)}}` — a bias component orthogonal to every sensitivity
direction, which is genuinely unobservable and must be reported rather than estimated.

**Falsification test (F1).** Train two otherwise identical models, one with `g_φ` affine in
`h` and one nonlinear. Inject a common-mode step offset into all channels at mid-life.
Prediction: the affine model's `b̂` drifts along `g'` and `ĥ` is biased by the same amount;
the nonlinear model absorbs the offset into `b̂` with `ĥ` unchanged within its posterior SD.
Both numbers are reported; if the nonlinear model does not separate them, Prop. 1 is
empirically empty and the bias-state operator loses its justification.

### 4.2 Proposition 2 — first passage under a filtered drift interpolates between two laws

**Statement.** For `h_τ = h_0 + ∫_0^τ r_s ds + σW_τ`, with `r_s` stationary, mean `μ`,
variance `s²`, correlation time `τ_c`, and `τ_f = (H − h_0)/μ`:

* if `τ_c ≪ τ_f`: `FPT → IG(μ, σ_eff²)` with `σ_eff² = σ² + 2s²τ_c` (rapid-fluctuation
  averaging — the drift's fluctuations act as **extra diffusion**);
* if `τ_c ≫ τ_f`: `FPT → ∫ IG(r, σ²) dN(r; μ, s²)`, the **mixture** that Si (2013) and
  Chen & Liu (2022) compute.

Existing RUL work uses the second endpoint unconditionally, i.e. it assumes the drift is
fixed for the whole remaining life. The correct object depends on `τ_c/τ_f`, which is
**large early in life and small near failure**. The practical consequence is falsifiable:
drift uncertainty should widen intervals near failure and should *not* widen them early —
the opposite of a uniform plug-in.

**Falsification test (F2) — RUN, RESULT: PASS with amended wording.** See
`paper/CIDO_F2_RESULT.md`; 8,000 paths per cell, ten `τ_c/τ_f` cells.

* The coefficient 2 is confirmed: the fitted `c` in `σ_eff² = σ² + c·s²τ_c` is **1.90,
  1.90, 2.20** for `τ_c/τ_f ≤ 0.03`, i.e. no free parameter is needed.
* Effective diffusion wins for `τ_c/τ_f ≤ 0.03` (KS ≤ 0.028), the mixture wins for
  `τ_c/τ_f ≥ 3` (KS ≤ 0.026), crossover at `τ_c/τ_f ≈ 1`.
* **The plug-in is never accurate**: KS 0.12–0.50 at every ratio.
* **Amended claim.** There is a gap band, `τ_c/τ_f ∈ [0.1, 3]`, where neither endpoint
  reaches KS 0.05 (worst 0.101). So Proposition 2 is not "a closed form for all regimes";
  it is "the two published families are the asymptotes, and the RUL-relevant range lies
  between them".

**Design consequence.** The FPT readout is **numerical**: sample the joint `(h, ρ)` forward
with no future observations, read the crossing time. The analytic laws are kept as
diagnostics that validate the sampler in their own regimes. Because `τ_f` is the remaining
life, `τ_c/τ_f` shrinks as an engine approaches failure, which predicts that the *shape* of
the predictive distribution changes with life stage — a checkable consequence on real data
(bin engines by observed age, test which law the held-out FPT distribution matches).

### 4.3 Why this is not "a gate on top of a state space model"

The audit's hardest objection is that evidence-gated transitions are the default in deep
SSMs (VRNN, RSSM, DS³M) and in adaptive filtering (Mehra 1970). The distinction that
survives is narrower and must be stated exactly:

* VRNN/RSSM condition the transition on a **deterministic summary of history**; the
  conditioning variable is not a statistic of the current innovation and carries no notion
  of "how much to trust this observation".
* Mehra rewrites `Q, R` from innovation **statistics** but keeps the drift exogenous; the
  state's future evolution is unaffected by what the sensor says.
* Here the **drift itself** is a function of the innovation (`§3.3`), which changes the
  process's future law — and that is precisely why Prop. 2 is needed and why the classical
  closed forms do not apply.

That distinction is a computation, not a name: it changes the transition kernel, hence the
predictive distribution, hence the FPT law. It is the only architectural claim in this
document that the audit could not locate a precedent for.

---

## 5. Training

All supervision uses only information available at or before the prediction time. No test
engine's termination point, no future sensor value, no future operating setting enters any
input; future `u` is marginalized with the training-set transition distribution.

**Joint objective**

$$\mathcal L = \underbrace{\mathcal L_{\text{FPT}}}_{\text{CRPS or censored NLL}}
+ \lambda_1 \underbrace{\mathcal L_{\text{obs}}}_{\text{next-step } y}
+ \lambda_2 \underbrace{\mathcal L_{\text{mono}}}_{\text{rate} \ge 0,\ \text{endpoint consistency}}
+ \lambda_3 \underbrace{\mathcal L_{\text{id}}}_{\text{Jacobian rank}}
+ \lambda_4 \underbrace{\mathcal L_{\text{age}}}_{\text{cross-age consistency}}$$

| term | definition | what it prevents |
|---|---|---|
| `L_FPT` | CRPS of the first-passage distribution; censored (`R > 125`) handled by the survival term, not by dropping samples | uncalibrated intervals; the ECS H2 failure |
| `L_obs` | Gaussian NLL of `y_{t+1}` given the filtered state | a latent state that fits RUL but not the sensors (the "CHI has no physical meaning" problem) |
| `L_mono` | hinge on `r_t ≥ 0` and on `median FPT` matching the observed trajectory's endpoint ordering | negative rates; trajectory/endpoint inconsistency |
| `L_id` | penalty when `span_t{J_t}` is rank-deficient within a batch | the confounding of Prop. 1 reappearing through a degenerate `g_φ` |
| `L_age` | agreement of the FPT distribution across observation ages of the same engine | state updates that depend on window length rather than on evidence |

**Identifiability of the split.** `(h, b)` is identified by Prop. 1's rank condition plus
the dynamics asymmetry (`b` is a random walk with no drift; `h` has a positive drift). The
training objective includes no term that rewards "small `b`" — an explicit sparsity penalty
on `b` would bias attribution toward blaming the state, which is the failure mode we are
trying to avoid.

---

## 6. Falsifiable hypotheses and gates

Recorded before any experiment. Any failure is reported with its number.

| # | hypothesis | test | GO threshold |
|---|---|---|---|
| **H1** | Prop. 1 holds | F1: common-mode offset injection, affine vs nonlinear `g_φ` | nonlinear `ĥ` bias < 0.3 × affine's; `b̂` absorbs ≥ 70% of the offset |
| **H2** | Prop. 2 holds | F2: FPT law vs MC over `τ_c/τ_f ∈ [0.03, 30]` | max KS distance < 0.05 in each regime; the crossover is located within one decade |
| **H3** | bias states suppress single-channel corruption | inject step offset / stuck-at / variance inflation into one channel at 60% life | RUL RMSE degradation < 1.5 RMSE, vs > 4 for the no-`b` arm |
| **H4** | common-mode degradation is still tracked | correlated drift across ≥ 6 channels | RUL update latency ≤ 5 cycles; `ĥ` increment within 1 SD of the no-injection case |
| **H5** | the mechanism is not parameter count | parameter-matched constant-gate control (see §7) | CIDO's advantage survives at equal parameters, ≥ 1.0 RMSE |
| **H6** | calibration | PIT histogram, 50/80/90% interval coverage on held-out engines | coverage within ±0.05 of nominal at all three levels |

`H1`/`H2` are theory gates and run first, before any network training. `H3`–`H6` run after
the architecture is fixed on the development folds.

---

## 7. Comparators (identical inputs, identical budget)

| arm | what it isolates |
|---|---|
| A (existing, additive cycle) | the current baseline |
| PE (existing, bounded encoding) | the ECS/PE baseline |
| ECS (Weibull prior × 32-bin likelihood) | the prior/likelihood fusion baseline from `ecs-rul` |
| plain neural SSM (no `b` states, no Jacobian split) | "is it just a state space model" |
| Kalman/robust filter on a hand-built HI | the classical competitor that the audit says will be cited against us |
| cross-sensor attention encoder + regression head | "is it just attention" |
| CIDO without Operator B (fixed `J`) | isolates the Jacobian anchoring |
| CIDO without `b` states | isolates the bias-state operator |
| CIDO with affine `g_φ` | isolates Prop. 1 |
| **parameter-matched constant gate** | isolates "more parameters" |
| **CIDO full** | — |

The parameter-matched control is mandatory: a learned scalar gate with the same parameter
count as the credibility module, receiving the same inputs.

---

## 8. What would make this NO-GO

Stated in advance so the decision is not made after seeing results.

1. `Prop. 1` fails F1 → the bias-state operator has no identifiable role → drop it, and the
   architecture becomes a state-space model with an FPT head, i.e. **composition only**.
2. `Prop. 2` fails F2 → no theoretical claim; the FPT readout is a known mixture →
   **composition only**.
3. `H5` fails (parameter-matched gate matches CIDO) → the credibility module is capacity,
   not mechanism → **NO-GO**.
4. `H3`/`H4` both fail → the bias states do not do what FDI theory says they should →
   **NO-GO**.
5. `H6` fails after one calibration attempt → the distributional output is unusable and the
   paper cannot claim probabilistic prediction.

Any two of 1/2 failing simultaneously means there is no architectural contribution and the
work reverts to the measurement-study framing already developed on `ecs-rul`.
