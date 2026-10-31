# E22a — Full test-set evaluation with a frozen padding rule

Status: pre-registered 2026-10-31T23:10+08:00, before any E22a number was computed.
sha256 of this file is recorded in the E22 ledger after freezing.

## Why

E21 evaluated only engines with at least 50 observed cycles (FD001 93/100, FD002
235/259, FD003 97/100, FD004 228/248; 653 of 707 engines). The excluded 54 are the
longest-remaining-life cases. Two questions remain open and are the point of E22a:

1. What do the models actually score on **all 707 official test engines**, with short
   trajectories handled by a rule that does not silently drop them?
2. How much do the capped and uncapped truth conventions differ **as analyses**, not
   as an error to be corrected? Capped truth measures agreement with the training
   target definition (the convention under which cross-paper RMSE comparison is
   possible). Uncapped truth measures error against the real remaining life, which is
   the quantity a maintenance decision actually depends on, and it is the convention
   under which the NASA score's late-penalty has its operational meaning. Both are
   reported; neither is called meaningless.

## Frozen padding rule

Short trajectories are windowed with `_last_window_with_edge_padding`
(`src/dataset.py:298-309`, the function the training-side test-window builder already
uses): when a trajectory has L < 50 cycles, the first frame is repeated 50 − L times
and prepended, then the final 50 frames are taken. No other rule is permitted. The
rule is fixed **before** any E22a number is seen; changing it afterwards voids the
experiment.

The hand-built extras (baseline / EMA / deviation features, `build_test_features` in
`scripts/dsh_e21_test_all.py`) are computed on the padded 50-frame window exactly as
the training pipeline computes them; `ref_count` therefore reaches 1.0 for every
engine.

## Protocol

* Models: the same 180 frozen checkpoints as E21 (5 folds × 3 seeds × 3 arms × 4
  subsets, config-driven architecture from each run's `metrics.json`). No retraining,
  no selection, no hyperparameter change.
* Per engine: last observable window (padded if L < 50), each fold's training scaler
  applied unchanged, prediction clipped to [0, 125].
* Aggregation: per-model metrics over the 15 models, and the ensemble (mean of the 15
  predictions per engine) with a 20k-replicate engine bootstrap 95% CI.
* Metrics, both truth conventions: RMSE, MAE, mean NASA score per engine and the
  cumulative NASA score over all evaluated engines, late fraction (err > 0), severe
  fraction (err > 10). Capped = min(true, 125); uncapped = true as given by the
  official RUL files.
* Outputs: `results/dsh/e22a/full_test.json`, per-engine ensemble CSVs
  `results/dsh/e22a/preds_<subset>_<arm>.csv` (columns: unit, y_pred, y_true_capped,
  y_true_raw), and `scripts/dsh_e22a_full_test.py`.

## Pre-registered expectations (to be checked, not tuned)

P1. Full-set capped RMSE exceeds the E21 retained-set value on FD002 and FD004 by
    roughly the bound computed in §6.10 (small, ±5%).
P2. Uncapped RMSE is systematically larger than capped on FD002/FD004 and nearly
    identical on FD001/FD003, because the uncapped penalty is concentrated in engines
    with true RUL > 125.
P3. The BE−A contrast sign and approximate size on the full set agree with E21 within
    the interval widths; FD003 remains positive at test.
P4. Cumulative NASA score is dominated by a small number of engines with true RUL far
    above the cap; its per-engine mean is the stabler summary.

A failed prediction is reported as a failed prediction.

## Prohibitions

No model, checkpoint, fold, seed or hyperparameter may be chosen using E22a numbers.
No padding-rule variants are explored. If the rule produces a degenerate feature for
some engine, that is reported, not fixed by switching rules.
