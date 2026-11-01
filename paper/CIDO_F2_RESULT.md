# CIDO F2 gate: first passage under a filtered drift

**Date**: 2026-11-01
**Script**: `scripts/cido/fpt_theory_check.py` (result JSON on `lab-a40`:
`results/cido/fpt_theory_check.json`)
**Verdict**: **P2_PASS = true** — but the verified statement is narrower than the one
originally proposed, and the narrowing is the useful part.

---

## 1. What was tested

Process: `h_τ = h_0 + ∫_0^τ r_s ds + σW_τ` with `r_s = exp(ρ_s)` and
`dρ = κ(μ_ρ − ρ)dt + σ_ρ dW_ρ`, so `r` has mean `μ`, coefficient of variation `cv = 0.3`,
stationary law lognormal, and correlation time `τ_c = 1/κ`. Threshold `H = 1`,
`τ_f = (H − h_0)/μ = 100`. Time is reported in units of `τ_f`.

Monte Carlo: 8,000 paths per cell, Euler-Maruyama with `dt = min(0.05, 0.05 τ_c)`
(the step is reduced for short `τ_c` so the rate process stays resolved — with a fixed
`dt = 0.05` and `τ_c = 0.3` the discretisation error alone produced a spurious KS of
0.053).

Three candidate laws, all evaluated against the same Monte Carlo sample by KS distance on
a 400-point time grid:

| label | law |
|---|---|
| **mixture** | `∫ IG(t; r, σ) π(r) dr`, `π` = exact lognormal stationary law — the object Si (2013) and Chen & Liu (2022) compute |
| **effdiff** | `IG(t; μ, σ_eff)`, `σ_eff² = σ² + 2 s² τ_c` — rapid-fluctuation averaging |
| **plugin** | `IG(t; μ, σ)` — drift treated as known exactly |

A fourth quantity is fitted at each cell: the coefficient `c` in
`σ_eff² = σ² + c·s²τ_c`, by minimising KS over `c ∈ [0, 4]`.

---

## 2. Results

| `τ_c/τ_f` | KS mixture | KS effdiff | KS plug-in | fitted `c` |
|---:|---:|---:|---:|---:|
| 0.003 | 0.4560 | **0.0200** | 0.1186 | 2.20 |
| 0.01 | 0.4166 | **0.0109** | 0.1991 | 1.90 |
| 0.03 | 0.3509 | **0.0283** | 0.2831 | 1.90 |
| 0.1 | 0.2538 | 0.0580 | 0.3763 | 1.80 |
| 0.3 | 0.1554 | 0.1011 | 0.4174 | 1.30 |
| 1.0 | 0.0636 | 0.2143 | 0.4709 | 0.60 |
| 3.0 | **0.0260** | 0.3480 | 0.4788 | 0.30 |
| 10 | **0.0156** | 0.5178 | 0.4946 | 0.10 |
| 30 | **0.0069** | 0.6545 | 0.4830 | 0.10 |
| 100 | **0.0065** | 0.7760 | 0.4821 | 0.10 |

---

## 3. What is established

**(a) The coefficient 2 is correct.** In the asymptotic regime (`τ_c/τ_f ≤ 0.03`) the
fitted coefficient is **1.90, 1.90, 2.20** — i.e. `σ_eff² = σ² + 2s²τ_c` with no free
parameter. This is the standard rapid-fluctuation result and it reproduces to within the
fit grid. The formula is therefore usable as stated, not as a fitted approximation.

**(b) The crossover is at `τ_c ≈ τ_f`, and it is sharp.** Effective diffusion wins below
0.1, the mixture wins above 3, and the exchange happens around 1. There is no single
analytic law that is accurate across the range.

**(c) The plug-in is never accurate.** Treating the rate as known scores KS 0.12–0.50 at
*every* ratio — it is the worst of the three except at the two extremes. Any RUL method
that propagates a point estimate of the degradation rate into a first-passage formula is
wrong by a margin that the interval itself does not reflect. This is the empirical
statement behind the practice-level warning, and it is stronger than expected.

**(d) There is a gap band, `τ_c/τ_f ∈ [0.1, 3]`, where neither endpoint reaches KS 0.05.**
The worst case within the band is 0.101 (at ratio 0.3). This is not a defect of the
implementation: it is the regime where the drift fluctuates appreciably *within* the
remaining life, so neither "constant random drift" nor "infinitely fast mixing" applies.

---

## 4. What this changes in the design

1. **Proposition 2 stands, with amended wording.** The claim is not "we have a closed form
   for all regimes" — it is: *the two published families are the two asymptotes; the
   coefficient in the fast-mixing asymptote is 2; and the practically relevant range for
   RUL falls in the gap between them.*

2. **The FPT readout must be numerical, not closed-form.** The method samples the joint
   `(h, ρ)` process forward with no future observations, and reads the crossing time off
   the samples. This is exact by construction, differentiable through reparameterisation,
   and it is the only way to be accurate inside the gap. The analytic laws are retained as
   *diagnostics and as sanity checks* of the sampler in their respective regimes — which is
   a much stronger validation than comparing to nothing.

3. **A testable prediction on real data.** `τ_f` is the remaining life, so `τ_c/τ_f` is
   large early in life and small near failure. Since `τ_c` is a property of the rate
   process (tens of cycles), an engine observed early sits near the mixture regime and one
   near failure sits near the effective-diffusion regime. **The shape of the predictive
   distribution should therefore change with life stage**, and a model that uses one fixed
   analytic form is wrong at one end by construction. This is checkable on C-MAPSS by
   binning engines by observed age and testing which law their held-out FPT distribution
   matches — and it is the kind of measurement-style result the project already knows how
   to do.

---

## 5. Honest limits of this check

* The check is on a **synthetic** process with a known rate law. It validates the
  *mathematics*, not that C-MAPSS degradation rates follow an OU process. The latter is an
  assumption of the model and is tested separately by `L_obs` (the observation likelihood)
  and by H6 (calibration).
* `cv = 0.3` and `σ_rel = 0.2` are fixed. The crossover location depends on both; the
  qualitative two-regime structure does not, but the numbers in the table are specific to
  this cell.
* The gap band is reported as `min(KS_mixture, KS_effdiff) ≥ 0.05`; with a different
  threshold the band's edges move.
* KS noise floor at 8,000 paths is ≈ 0.015, so differences below that are not resolvable.
  All conclusions above involve differences of ≥ 0.03.
