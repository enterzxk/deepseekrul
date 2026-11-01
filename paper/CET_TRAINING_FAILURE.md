# CET-RUL: why the recursive-belief network does not train

**Date**: 2026-11-01
**Status**: **NO-GO for the architecture as specified.** Six structural defects found and
fixed by training; the seventh is not a bug but a property of the design.

## What was built

`src/cet/model.py`, 65,401 parameters: causal dilated-conv encoder → observation model
(forecasts the next sensor reading from the belief) → CEIO (whiten, GLS along a learned
degradation direction, e-process gate) → recursive belief over failure time → RUL readout.
Trained end to end on C-MAPSS with the true lifetime as the target.

## Defects found by training (all fixed, all real)

| # | defect | evidence | fix |
|---|---|---|---|
| 1 | evidence head emitted a **scalar** broadcast over the basis: a constant in log space, which softmax removes | gate had literally zero effect for 6 epochs | output a vector over the basis |
| 2 | betting fraction started at 0 → `d(loss)/d(score) ≡ 0` | gate never left 0 | floor at 0.3 (still predictable, martingale property intact) |
| 3 | `clip(gate, 0, 1)` is an **absorbing state**: gradient dies at the clamp | run froze at gate = 0 from epoch 2, predicted 125 forever | two-parameter sigmoid map |
| 4 | Gaussian tilt `−z²` reaches −4e4 on far bins → **softmax saturation**, zero gradient elsewhere | 16.7 → 113 RMSE in one epoch | Student-t (ν = 4) heavy tail |
| 5 | belief over **remaining life** shifted one bin per cycle: a broad prior shifted 150 times walks its tail into the low-R region | measured q[:6] ≈ 0.017 with sd 35 while the mean was nearly right | belief over **absolute failure time**, survival conditioning by truncation, no shift |
| 6 | `to_rul` clamped `age + R` to the last bin, **aliasing** hundreds of R values onto one and multiplying its mass | mean pinned to the horizon | explicit support with masking |
| 7 | observation model read `e[:, t]`, which already contains `y_t`, so `ŷ_t ≈ y_t` and the **innovation was identically zero** | — | forecast from `e[:, t−1]` |

## The remaining failure, and why it is structural

After all seven fixes the diagnostic is unambiguous:

```
q sum per step:            1.0000 (valid distribution)
q argmax at t = 5..150:    219, 221, 222, 223   (the prior, drifting by one bin per cycle)
RUL mean at t = 5..150:    212.8, 171.1, 127.4, 91.3   (it is counting down, not inferring)
gate mean = 0.1345, gate std = 1.5e-08          (constant across every engine and cycle)
prediction change when the input is time-reversed: 2.0e-05
```

The gate is constant to eight digits and the output does not depend on the input. The
mechanism is a **chicken-and-egg deadlock**: the belief only depends on the evidence
through `gate × tilt`; when `gate ≈ 0` the loss has no gradient with respect to the
evidence path (score, direction, whitening); with no gradient, `direction` decays under
weight decay and `score` stays at zero, so the gate never rises. Every gate design that
multiplies the update (clip, sigmoid, floor) reproduces this: the floor makes the gate
nonzero but it is a *constant*, which carries no information, so the belief still cannot
discriminate.

Two further observations that make this hard to escape inside the current design:

* A 320–600 step belief chain cannot be trained with full BPTT (gradient is clipped away)
  and cannot be trained with truncated BPTT either — truncation removed the only gradient
  that moved the belief (measured: it froze at 125).
* The belief must be sharp to score well (intervals of width 145 cycles give CRPS 45.6
  against a competitor range of 6.6–10.1), but sharpness comes from accumulating the
  tilt, which is exactly the path that has no gradient.

## What this rules out

The recursive distribution-over-failure-time formulation is **not trainable as a
multiplicative-gate recursion** on this data at this scale. That is a property of the
parameterisation, not of the team's effort: seven defects, each independently fatal, each
found only by running.

## What would still be defensible

Replace the recursion over a 600-bin distribution with a recursion over a **compact
state** (8–16 dims) that is updated by the same innovation, and decode the distribution
from the state at each step (a discrete-hazard or Weibull head, as WTTE-RNN does). The
recursion then has a short gradient path, and the survival semantics survive in the
decoder. This keeps the mechanism (evidence-driven state update) and drops the object
that cannot be trained.
