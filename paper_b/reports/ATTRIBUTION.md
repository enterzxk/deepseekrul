# Historical 10.91 -> 12.70 attribution (diagnostic only)

This experiment explains a historical discrepancy. It is **not** paper
evidence: the official test set is used, and the variants deliberately
re-introduce defects.

## Question

The same DSPT-Full architecture, same L=70, seed 101, fold 0, was reported at
FD001 test RMSE 10.91 before the Phase-1 fixes and 12.70 after them, while
validation RMSE barely moved (9.04 vs 9.15). Which change caused the test
regression?

## Design

Start from the protocol-fixed DSPT-Full and roll back one fix at a time:

| run | rollback |
|---|---|
| `ATTR_fixed` | none (reference) |
| `ATTR_legacy_ms_pad` | MS convolution sees zero padding, no replicate fill |
| `ATTR_legacy_ema` | causal EMA initialised at `x[:,0]` (zero for padded windows) |
| `ATTR_unmasked` | temporal Transformer does not mask left padding |
| `ATTR_legacy_rul` | RUL endpoint at n instead of n-1 |
| `ATTR_global_hi` | HI fitted on all engines (leaky) |
| `ATTR_legacy_hi_wrong` | HI monotone sign inverted |

Each run is trained and evaluated on FD001 with the same split and seed.

## Results

`[RESULT]`

## Interpretation

`[RESULT]`

## Caveat

The archived pre-fix code also differed in the early-stopping target (mode
mismatch) and in the evaluator. A perfect numerical reproduction of 10.91 is
not expected; the goal is to rank the mechanisms by effect size.
