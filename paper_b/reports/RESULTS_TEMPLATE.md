# Results table template (filled only from result files)

## Table 1. DSPT 8-config ablation, validation RMSE (mean +- std over 3 seeds)

| config | FD001 | FD002 | FD003 | FD004 | mean rel. vs base |
|---|---:|---:|---:|---:|---:|
| Base | | | | | 0.00 |
| MS | | | | | |
| SA | | | | | |
| HI | | | | | |
| MS+SA | | | | | |
| MS+HI | | | | | |
| SA+HI | | | | | |
| FULL | | | | | |

Source: `results/tables/ablation8_by_subset.csv`, `ablation8_overall.csv`.

## Table 2. MS mechanism (validation, relative to base)

| variant | FD001 | FD004 | mean rel. | wins/2 |
|---|---:|---:|---:|---:|
| single_proj | | | | |
| multi_noproj | | | | |
| multi3 | | | | |

## Table 3. SA mechanism (validation, relative to SA)

| variant | FD001 | FD002 | FD003 | FD004 | mean rel. | wins/4 |
|---|---:|---:|---:|---:|---:|---:|
| identity | | | | | | |
| cond | | | | | | |

## Table 4. HI mechanism (validation, relative to HI)

| variant | FD001 | FD004 | mean rel. | wins/2 |
|---|---:|---:|---:|---:|
| hf_wrong | | | | |
| hf_nomono | | | | |
| hB_correct | | | | |

## Table 5. Vanilla Transformer vs DSPT (validation RMSE)

| model | FD001 | FD002 | FD003 | FD004 | params |
|---|---:|---:|---:|---:|---:|
| Vanilla (raw) | | | | | 72,257 |
| Vanilla (DSPT features + cycle) | | | | | |
| DSPT Base | | | | | 77,058 |
| DSPT FULL | | | | | 115,651 |

## Table 6. Frozen official test (mean +- std over 3 seeds)

| model | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| DSPT Base | | | | |
| DSPT FULL | | | | |
| Vanilla (raw) | | | | |
| Vanilla (features) | | | | |
| Final model | | | | |

## Table 7. Historical attribution (FD001, seed 101, diagnostic)

| variant | val RMSE | test RMSE | delta test vs fixed |
|---|---:|---:|---:|
| fixed | | | 0 |
| legacy_ms_pad | | | |
| legacy_ema | | | |
| unmasked | | | |
| legacy_rul | | | |
| global_hi | | | |
| legacy_hi_wrong | | | |
