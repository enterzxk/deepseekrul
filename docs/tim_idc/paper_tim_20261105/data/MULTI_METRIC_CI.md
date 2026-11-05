# Four-metric official-test results

Source: `multi_metric_ci.json`. Ensemble of 3 seeds; 95% CI from 5,000 engine-paired bootstrap resamples (seed 99). RMSE/MAE/bias in cycles; NASA score lower is better; bias = mean(pred - true), negative = earlier.

## Full budget (Table VIII) — FULL − M0

| Subset | ΔRMSE | ΔMAE | ΔBias | ΔNASA |
|---|---|---|---|---|
| FD001 | -0.409 [-0.850, +0.010] | -0.298 [-0.696, +0.104] | -0.083 [-0.490, +0.339] | -10.8 [-24.0, +0.9] |
| FD002 | +0.348 [-0.259, +0.948] | +0.512 [-0.033, +1.034] | **-2.145 [-2.632, -1.675]** | -3.1 [-97.9, +98.0] |
| FD003 | **-0.437 [-0.799, -0.060]** | -0.222 [-0.527, +0.066] | -0.009 [-0.328, +0.312] | **-24.7 [-54.3, -0.5]** |
| FD004 | -0.524 [-1.114, +0.080] | -0.102 [-0.580, +0.374] | -0.459 [-0.955, +0.013] | **-108.1 [-249.6, -0.4]** |

## Full budget — FULL − OPS, FD002 (the reversal)

| ΔRMSE | ΔMAE | ΔBias | ΔNASA |
|---|---|---|---|
| **+0.590 [+0.036, +1.148]** | **+0.709 [+0.286, +1.119]** | **-1.045 [-1.465, -0.622]** | +71.7 [-31.7, +183.9] |

## Lightweight (supplementary Table IX)

- FULL−OPS: NASA significant on all four (FD001 −30.5, FD002 −131.7, FD003 −161.4, FD004 −71.8).
- FULL−M0: NASA significant on FD001 (−251.9), FD003 (−270.8), FD004 (−71.4); FD002 +12.2 ns.

## Key reading

- RMSE: FULL−M0 significant only on FD003; FULL−OPS significant on all four, but worse on FD002.
- NASA: FULL−M0 significant on FD003 and FD004; FD001 borderline (−10.8, CI upper +0.9); FD002 flat.
- MAE: no significant FULL−M0 gain on any subset.
- Bias: FULL shifts predictions earlier on FD002 (−2.145) and FD004 (−0.459); the asymmetric NASA score rewards this shift.
- The prior trades accuracy for safety; on FD002 the age/condition prior is better on RMSE, MAE, and NASA.
