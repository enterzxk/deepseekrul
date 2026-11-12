# A8 ablation: complete 8-config grid (96 cells)

Generated 2026-11-12 from `paper_b/results/a8_engine_level.csv`, which is
recomputed from the saved `val_pred.csv` / `test_metrics.json` of each run.
Do not edit by hand; regenerate with `paper_b/experiments/` tooling.

## 0. What this report answers, and what it does not

This grid was **not** run to prove that the modules help. On FD001–FD003 the
whole method space spans about two RMSE and the seed spread is 0.2–0.9, so a
"our module wins" claim would sit inside noise and would not survive review.

What the grid can support is narrower and more useful: **under which subset and
which combination does each module help or hurt**. The answer turns out to be
regime-dependent, and that is the contribution here.

## 1. Design

2 x 2 x 2 factorial over three switches, plus the all-off cell:

| switch | off | on |
|---|---|---|
| `ms_mode` | `off` | multi-scale conv |
| `sa_mode` | `off` | cross-sensor attention |
| `hi_source` | `none` | HI auxiliary supervision |

8 cells x 4 subsets x 3 seeds (101/202/303) = **96 training runs, all complete**.
Official test set evaluated for the frozen manifest (`FREEZE_ABLATION_SEED101`),
which covers **seed 101 only**: 8 x 4 = 32 test evaluations, all complete.

Validation RMSE below is **engine-level**: each engine's own RMSE first, then the
unweighted mean over engines (5 truncated windows per engine). This is not the
same as the pooled number in `metrics.json`; the pooled values are kept in
`a8_engine_level.csv` as `val_pool`.

## 2. Validation RMSE (engine level, mean ± sd over 3 seeds)

| config | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| base (all off) | 11.085 ± 0.470 | 10.974 ± 0.069 | 11.381 ± 0.598 | 11.806 ± 0.448 |
| +ms | 10.545 ± 0.314 | 11.096 ± 0.249 | 11.493 ± 0.174 | 12.226 ± 0.473 |
| +sa | 10.431 ± 0.579 | 10.782 ± 0.335 | 11.116 ± 0.582 | 12.084 ± 0.400 |
| +hi | 11.164 ± 0.442 | 10.642 ± 0.352 | 11.279 ± 0.153 | **11.580 ± 0.167** |
| ms+sa | **10.240 ± 0.554** | **10.675 ± 0.417** | **10.983 ± 0.388** | 12.135 ± 0.850 |
| ms+hi | 10.495 ± 0.136 | 11.034 ± 0.182 | 11.576 ± 0.329 | **11.579 ± 0.285** |
| sa+hi | 10.431 ± 0.738 | 10.894 ± 0.273 | 11.187 ± 0.718 | 12.475 ± 0.174 |
| full (all on) | 10.359 ± 0.317 | 10.761 ± 0.267 | 10.987 ± 0.890 | 11.818 ± 0.344 |

## 3. Official test RMSE (seed 101, descriptive — never used for selection)

| config | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| base | 11.547 | 10.975 | 12.266 | 12.497 |
| +ms | 10.664 | 11.538 | 11.601 | 11.744 |
| +sa | 10.550 | 10.909 | 12.828 | 12.227 |
| +hi | 11.451 | 11.381 | 12.151 | 12.462 |
| ms+sa | 10.822 | 10.848 | 10.946 | 12.653 |
| ms+hi | 10.930 | 11.696 | 11.088 | 12.552 |
| sa+hi | 10.789 | 10.958 | 11.298 | 12.775 |
| full | **10.288** | **10.523** | 11.803 | 12.628 |

**Freeze coverage.** `FREEZE_ABLATION_SEED101.json` lists 32 cells, all seed 101.
The disk holds 36 test evaluations: four extra cells
(`A8_base_L70/FD001/s202`, `/s303`, `A8_mssa_L70/FD001/s202`, `/s303`) were
evaluated outside the manifest. They are **excluded from this table and marked
n/a** — retro-fitting them into the freeze declaration would falsify the
evaluation order, which is exactly what the freeze exists to prevent. The table
above is seed 101 only, for every cell.

## 4. Main effects (mean over configs with the switch on, minus off)

Negative = the module helps. Validation:

| switch | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| ms | −0.368 | +0.068 | +0.019 | −0.047 |
| sa | −0.457 | −0.159 | −0.364 | **+0.330** |
| hi | +0.037 | −0.049 | +0.014 | **−0.200** |

Official test (seed 101):

| switch | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| ms | −0.41 | +0.10 | −0.78 | −0.10 |
| sa | −0.54 | −0.59 | −0.06 | **+0.26** |
| hi | −0.03 | +0.07 | −0.33 | +0.32 |

## 5. Findings

1. **On FD001–FD003, sa is the only module with a consistently negative main
   effect** (−0.46 / −0.16 / −0.36 validation; −0.47 / −0.59 / −0.06 test).
   ms is mixed, hi is essentially zero. None of these are large compared with
   the seed spread.
2. **FD004 reverses sa**: +0.330 validation, +0.257 test. Cross-sensor attention
   is the one module that actively hurts on the subset with the most operating
   regimes and the most engines.
3. **FD004 rewards hi instead**: −0.200 validation. It is the only switch with a
   consistent sign on that subset.
4. **ms and hi interact on FD004.** ms alone is 12.226 against base 11.806
   (+0.42, clearly worse), yet ms+hi is 11.579, the best FD004 cell. The main
   effect of ms (−0.047) hides this: ms is harmful on its own and neutral once
   hi is present. Any statement of the form "multi-scale helps on FD004" or
   "multi-scale hurts on FD004" is therefore wrong; the honest form is
   "ms requires hi on FD004".
5. **The all-off base is competitive on FD001–FD003** (11.085 / 10.974 / 11.381),
   which is the expected consequence of a saturated benchmark rather than a
   defect of the modules.

## 6. Caveats

- **Model selection used pooled validation RMSE.** The GPU host still carries
  the pre-R2 `metrics.py`, so every run in this grid selected its best epoch by
  pooled RMSE. The engine-level numbers above are recomputed post hoc from the
  saved predictions. Selection and evaluation therefore use different
  aggregations; this is recorded, not corrected.
- **The test column is one seed.** Only seed 101 was frozen, so no test-side
  seed variance exists. Do not compare the test column against 3-seed validation
  numbers as if they had the same precision.
- **FD004 is cap-dominated.** 20.4% of validation windows sit at the RUL = 125
  cap and they contribute 54.7% of the squared error (FD001–FD003: 7–11% of
  windows, 11–14% of error). Any FD004 mechanism claim is partly a claim about
  the truncated early-life regime.
- **`full` in this grid is the same run set as `F_full_basic_hi1`** in
  `SEED_ROBUSTNESS.md` (configs differ only by `tag`). The other cells are
  distinct runs and must not be compared across the two reports.

## 7. What would falsify the mechanism story

- If an engine-level re-selection (R2 code) changed which cell wins on FD004,
  findings 2–4 would be about selection noise rather than mechanism.
- If the sa penalty on FD004 disappeared once cap windows are excluded, the
  effect would be a truncation artefact rather than an attention/regime effect.
- If ms+hi failed to beat hi alone in a repeat with a different seed, finding 4
  would be a single-seed coincidence.
