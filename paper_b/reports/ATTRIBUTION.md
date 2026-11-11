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

FD001, seed 101, DSPT-Full, all runs trained on the same machine (a40) with the
same protocol (36 epochs, patience 6). Test numbers are diagnostic only.

| variant | validation RMSE | test RMSE | test vs fixed |
|---|---:|---:|---:|
| fixed (reference) | 11.43 | 10.83 | 0.00% |
| global HI (leaky) | 11.66 | **10.51** | **-2.96%** |
| legacy EMA init | **10.71** | 11.46 | +5.81% |
| inverted HI monotone | 11.27 | 10.89 | +0.53% |
| legacy MS zero-pad | 11.29 | 10.84 | +0.02% |
| legacy RUL endpoint | 11.06 | 10.94 | +1.01% |
| **unmasked temporal Transformer** | **11.22** | **10.25** | **-5.42%** |

Source: `paper_b/results/raw/ATTR_*/*/s101/{metrics,test_metrics}.json`.

## Interpretation

1. **The largest historical advantage came from leakage, not architecture.**
   Fitting HI on all engines (the pre-fix behaviour) improves FD001 test RMSE by
   3.0% (10.83 -> 10.51). This alone explains a substantial part of why the old
   numbers looked better. It is inadmissible.
2. **The EMA initialisation fix is beneficial on test but looks harmful on
   validation.** Fixing the zero-pad EMA initialisation improves test by 5.8%
   relative to the legacy behaviour, yet the legacy variant has the *better*
   validation RMSE (10.71 vs 11.43). This is a concrete validation/test
   disagreement and a warning against validation-only claims.
3. **The padding mask is the surprise.** Training without
   `src_key_padding_mask` improves both validation (-1.8%) and test (-5.4%).
   The effect is not confined to short test engines: RMSE improves from 11.95
   to 11.59 on engines shorter than 70 cycles and from 10.64 to 10.01 on full
   windows. Because full windows have no padding at all, this is a *training*
   effect (the unmasked model learns different weights), not an inference-path
   artefact. A 24-run control (`NM_base`, `NM_full`, 4 subsets x 3 seeds) tests
   whether it replicates under the frozen protocol.
4. MS zero-padding and the HI monotone sign are negligible on FD001 test; the
   RUL endpoint shift costs about 1%.

## Caveat

This experiment deliberately uses the official test set and single-seed runs.
It explains the past; it cannot select the paper model. The padding-mask
control is therefore re-run under the full protocol before any claim is made.

## Caveat

The archived pre-fix code also differed in the early-stopping target (mode
mismatch) and in the evaluator. A perfect numerical reproduction of 10.91 is
not expected; the goal is to rank the mechanisms by effect size.