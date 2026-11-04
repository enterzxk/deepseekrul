# Module B/C Verdict Report

Date: 2026-11-04 (both modules complete). Script:
`scripts/summarize_modulebc.py` (engine-clustered bootstrap, 5000
reps, same statistics as Gate 0/1). Verdict file:
`experiments/ida/moduleBC/modulebc_verdict.json`.

## Design recap (condition isolation)

All seven arms share the identical M1p backbone and the identical context
injection (prior projection, +640 params); only the module-level condition
signal differs: `tau` (individualized phi(t/μ_T) ++ u), `abs` (absolute-time
phases, same dimension), `shuf` (engine-permuted tau, same marginal), and
`static` (constant, capacity-matched, Module C only). Modules are zero-init:
at initialization every arm is numerically identical to M1p (tested,
delta = 0.00e+00). Total overhead +1332 params = 3.1% of M1p.

## Module B — deg-conditioned attention bias: **DROP**

FD002, seed 101, all 5 outer folds pooled (n = 259 engines):

| Pair | RMSE A | RMSE B | ΔRMSE | 95% CI | Verdict |
|---|---|---|---|---|---|
| MB.1 MB_tau vs M1p | 12.221 | 12.217 | **+0.007** | [-0.205, +0.223] | FAIL |
| MB.2 MB_tau vs MB_abs | 12.221 | 12.092 | **+0.131** | [-0.119, +0.375] | FAIL |
| MB.3 MB_tau vs MB_shuf | 12.221 | 12.193 | **+0.030** | [-0.177, +0.242] | FAIL |

Frozen keep rule: tau arm must beat M1p with CI excluding 0 AND beat both
controls. All three comparisons fail (CIs all cross zero with comparable
widths; the tau point estimate is not even the best of the three arms).

## Interpretation (honest)

1. **No signal, not underpowered noise**: the three condition variants are
   statistically indistinguishable from each other. If degradation-stage
   attention modulation carried usable information, the tau arm would
   separate from shuf; it does not.
2. **Consistent with Gate 1**: the context-level prior injection (M1p, +640
   params) already gives the network all the lifetime-stage information it
   can exploit. A second, deeper injection path (attention bias) is
   redundant — the marginal value of the same information through a new
   channel is ~0.
3. **The zero-init + isolation design worked as intended**: a tempting
   "we added degradation-aware attention" claim would not have survived
   review; the pipeline removed it at a cost of 15 GPU runs.

## Module C — tau-gated sensor mixer: **DROP**

FD002, seed 101, all 5 outer folds pooled (n = 259 engines), 20/20 runs rc=0:

| Pair | RMSE A | RMSE B | ΔRMSE | 95% CI | Verdict |
|---|---|---|---|---|---|
| MC.1 MC_tau vs M1p | 11.952 | 12.217 | **-0.261** | [-0.544, +0.015] | FAIL |
| MC.2 MC_tau vs MC_abs | 11.952 | 11.979 | **-0.025** | [-0.261, +0.209] | FAIL |
| MC.3 MC_tau vs MC_shuf | 11.952 | 11.960 | **-0.007** | [-0.243, +0.222] | FAIL |

Frozen keep rule (same as Module B): all three comparisons fail. MC.1 comes
closest (upper bound +0.015) but still crosses zero.

## Interpretation (honest)

1. **The mixer's uniform shift is capacity, not information**: all three MC
   condition variants (tau / abs / shuf) land within 0.03 RMSE of each other
   (11.952–11.979) and all sit below both M1p (12.217) and every Module B arm.
   A condition-dependent mechanism that really consumed tau would separate
   tau from shuf; instead even the shuffled arm gets the full shift.
2. **The -0.26 shift vs M1p is an unconditional capacity/regularization
   effect**, attributable to the extra low-rank mixing path (+252 params) and
   its effect on optimization — not to degradation-stage sensing. Under the
   frozen rule it does not count as module evidence. It is recorded as a
   descriptive observation only.
3. **Attribution discipline held under temptation**: MC_tau's point estimate
   was the best of the night (-0.26 vs M1p); without the shuf/static controls
   and the CI-excludes-0 rule this would have been reported as a win. The
   controls show it is not one.

## Final disposition (frozen clause)

- **Module B: DROP. Module C: DROP. DROP_BOTH.**
- **The final architecture freezes as exactly M1p** (PE backbone + zero-init
  prior projection with tau-phase context, +640 params over M0). Gate 2
  (uncertainty calibration), official-test unsealing, and the manuscript all
  use M1p only.
- The paper reports both modules as one negative-ablation paragraph + one
  table (six comparisons, all CIs). This strengthens the mainline claim: the
  gain comes from the prior→clock→context path, and deeper injection channels
  add nothing.

## Presentation rule for the manuscript

- Report both module tables with all six comparisons and CIs.
- State: "Degradation-conditioned attention bias and tau-gated sensor mixing
  provided no benefit over absolute-time, shuffled, or no conditioning (all
  CIs cross zero), indicating the context-level injection already saturates
  the usable lifetime-stage information."
- Do NOT quote MC_tau's -0.261 as a module win; if the uniform MC shift is
  mentioned at all, label it explicitly as a capacity effect (shuffled
  conditioning reproduces it).
