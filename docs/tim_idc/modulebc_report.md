# Module B/C Verdict Report

Date: 2026-11-04 (Module B complete; Module C pending, ETA ~20:00).
Script: `scripts/summarize_modulebc.py` (engine-clustered bootstrap, 5000
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

## Disposition (frozen clause)

- **Module B is dropped from the architecture.** No further tuning, no
  re-interpretation, no subset search. The paper reports this as a negative
  ablation (one sentence + table row), which strengthens the mainline claim:
  the gain comes from the prior→clock→context path, not from generic
  extra conditioning.
- Module C (sensor mixer) verdict follows when its 20-run queue completes
  (~20:00). If C also drops, the final method is exactly M1p and the
  architecture freezes for Gate 2/3.

## Presentation rule for the manuscript

- Report the Module B table with all three comparisons and CIs.
- State: "Degradation-conditioned attention bias provided no benefit over
  absolute-time, shuffled, or no conditioning (all CIs cross zero),
  indicating the context-level injection already saturates the usable
  lifetime-stage information."
