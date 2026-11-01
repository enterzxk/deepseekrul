# CIDO F1 gate: identifiability of the health state against per-channel bias states

**Date**: 2026-11-01
**Script**: `scripts/cido/f1_identifiability.py`
**Verdict**: **F1_PASS = false.** Proposition 1 is **false as stated**, and the bias-state
operator loses its theoretical justification. Recorded with all four test designs, three of
which failed for reasons that are themselves the finding.

---

## 1. Proposition 1 as written

> With `y_t = g(h_t) + b_t + ε_t`, the accumulated Fisher information
> `Σ_t [[g'ᵀψ⁻¹g', g'ᵀψ⁻¹],[ψ⁻¹g', ψ⁻¹]]` is rank-deficient when `g'` is constant, so
> `(h + δ, b − Lδ)` is observationally equivalent and the pair is **not identifiable**;
> when `g` is nonlinear the confounding direction rotates and identification is restored.

## 2. F1a (rank test): **passes**

Accumulated observation information over a 220-cycle trajectory, 14 channels:

| model | smallest eigenvalue | second | largest | condition number |
|---|---:|---:|---:|---:|
| affine `g` | **−7.1e−11** | 8.8e+04 | 5.4e+06 | 5.4e+36 |
| nonlinear `g` | **1.84e+04** | 8.8e+04 | 1.3e+06 | 70.5 |

The rank claim is exactly right: the affine observation model has a genuine null direction,
the nonlinear one does not. This is the part of Proposition 1 that survives.

## 3. F1b (consequence): **fails, in the direction opposite to the prediction**

Four designs were run. Each is recorded because the failure mode is informative.

| version | what it varied | affine | nonlinear | outcome |
|---|---|---:|---:|---|
| v1 | inject a **common-mode** offset | h-bias 0.019 | h-bias 0.050 | invalid: common mode is **not** the confounded direction (that is `L`); the two patterns are distinguishable by their spatial shape, so nothing was tested |
| v2 | inject along `L`, vary the **initial prior** on `b` (20×) | shift 0.0000 | shift 0.0188 | invalid: the filter forgets `t = 0` within ~30 cycles |
| v3 | inject along `L`, vary the **process-noise ratio** of `b` (10×) | shift 0.0040 | shift 0.0520 | invalid: the filter was handed the **true rate**, so `h` was pinned by the dynamics and the observation model was decorative |
| v4 | inject along `L`, give the filter a rate **15 % too high** | RMSE 0.0206 (vs 0.0194 at the true rate) | RMSE 0.0617 (vs 0.0517) | **valid, and the prediction is reversed**: the affine model is *more* accurate and *less* damaged by the wrong rate |

So where the test is finally valid (v4), the model that Proposition 1 says should be
crippled is the better one, by a factor of two in RMSE.

## 4. Why the proposition is wrong

Two errors, both instructive.

**(a) The rank test measured the wrong object.** F1a accumulates only the *observation*
information. The trajectory system also contains the transition model, and the two states
have different dynamics — `h` has a positive drift, `b` is a driftless random walk. A
perturbation `(δ, −Lδ)` that is invisible to every observation is therefore still
inconsistent with the *transition* over time: it moves `h` off its drift. The trajectory
Fisher information is `P_0⁻¹ + Σ_t H_tᵀψ⁻¹H_t` (with a constant-rate transition), which is
full rank whenever the prior precision has a component along the null direction — which it
always does. Rank deficiency at a single time step does not imply non-identifiability of the
trajectory.

**(b) The confounding is harmless because `b` can absorb it for free.** The null direction
is one-dimensional inside a 14-dimensional bias space. When the truth contains a bias along
`L`, the filter splits it by whatever prior it holds and the `h` estimate stays accurate,
because nothing else competes for that direction. "Non-identifiable" here means "the split
is arbitrary", not "the health state is wrong". Those are very different consequences, and
only the second would matter.

**(c) The nonlinear model is actively worse at tracking `h`**, because its sensitivity
`g'(h) = aλe^{−λh}` decays as the engine degrades — precisely when accuracy matters most.
The affine model's constant sensitivity keeps `h` well observed throughout. This is an
observability effect that dominates the identifiability effect, and it was not anticipated.

## 5. What this costs the design

Per the pre-registered rule in `CIDO_METHOD_DESIGN.md` §8.1: *"Prop. 1 fails F1 → the
bias-state operator has no identifiable role → drop it, and the architecture becomes a
state-space model with an FPT head, i.e. composition only."*

That rule now applies. Consequences, stated plainly:

1. **The per-channel bias-state operator is demoted from "core contribution" to
   "engineering component".** Joint filtering of a bias state with a health state is
   standard adaptive-Kalman practice (Hajiyev 2025, and the augmented-state literature
   behind it). Without an identifiability argument it has no claim to novelty.
2. **A nonlinear observation model is not required.** Since identification comes from the
   dynamics asymmetry, a linear observation model is sufficient and empirically better
   behaved. This simplifies the architecture and removes one ablation arm.
3. **What remains is Proposition 2 and the gap-band finding** (`CIDO_F2_RESULT.md`): the
   two published first-passage families are the asymptotes of a filtered-drift process, the
   coefficient in the fast-mixing asymptote is 2, and the RUL-relevant range falls between
   them so that neither closed form is accurate. That is a real result, verified
   numerically, but it is a **result about how to compute the predictive distribution**, not
   a new network operator.

## 6. Honest status of the architectural claim

Two of the three originally proposed operators were withdrawn by the literature audit.
The third — bias states with their own dynamics — is now withdrawn by this gate.

**There is currently no architecture-level novelty claim in CIDO-RUL.** What exists:

| asset | status | what it can support |
|---|---|---|
| Prop. 2 + gap band | verified numerically | a methodological contribution: correct FPT computation under an evidence-driven drift, with the two published closed forms located as asymptotes |
| the numerical FPT sampler | designed, not built | the vehicle for that contribution; also gives calibrated intervals, which most deep RUL papers do not report |
| `encode()` + `build_windows(meta)` | built and verified | infrastructure, no claim |
| H3/H4 (bias states empirically) | not run | could still justify the operator empirically, but not as a *novel* operator |

Per the falsification rule, the architecture claim is **NO-GO** until either H3/H4 produces
an effect large enough to be a mechanism rather than a component, or a different operator is
found that survives an audit of this kind.

## 7. Recommendation

Two defensible paths, in order of expected value:

1. **Reframe around the FPT result.** "How should the RUL distribution be computed when the
   degradation rate is driven by the observations?" — Prop. 2 answers a question the FPT
   literature assumes away, the gap band shows both published closed forms are wrong in the
   RUL-relevant range, and the fix (sample the joint process) is straightforward but nobody
   does it. This is a measurement-methodology contribution, which is what TIM publishes, and
   it does not require a novel network.
2. **Stop the architecture line** and consolidate the existing measurement work (E18–E22 +
   ECS) into a submission. This is the lower-risk option and remains available.

What is *not* recommended: continuing to search for an architecture claim by iterating on
operators. Two rounds of audit have now removed three candidate operators; the base rate
suggests the next one will also be a known method under a new name.
