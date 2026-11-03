# Dataset & Protocol (frozen facts — v0.1)

> Every number in this section is a protocol constant or a dataset property,
> not an experimental result. Sources: `src/dataset.py`,
> `src/lifetime_prior/nested_prior.py`, `gates.json`, `docs/FINAL_PROTOCOL.md`.

## 3.1 Dataset

NASA C-MAPSS turbofan benchmark, training splits only (`train_FD00x.txt`).
The official test split is *never* used for development, model selection, or
gate decisions; it is unsealed exactly once after architecture freeze
(section 6 of the gated spec).

| subset | train engines | op. conditions | fault modes | sensors used |
|--------|--------------|----------------|-------------|--------------|
| FD001  | 100 | 1 | 1 | 14 |
| FD002  | 260 | 6 | 1 | 14 |
| FD003  | 100 | 1 | 2 | 14 |
| FD004  | 249 | 6 | 2 | 14 |

14 sensors = `SELECTED_SENSOR_COLUMNS` (the standard constant/flat-channel
pruning shared by most baselines).

## 3.2 Targets and windows

- Target: piecewise-linear RUL, capped at 125 (`build_piecewise_linear_rul`,
  max_rul=125) — identical for every arm and every compared method.
- Window: length 50, stride 1. Evaluation windows end at cycles 50..T of each
  engine (dev engines run to failure; this includes end-of-life windows with
  RUL=0, unlike the official test protocol — stated explicitly wherever
  development-val numbers appear).

## 3.3 Evaluation splits (development cross-validation)

- Outer 5-fold split **by engine**, `KFold(5, shuffle=True,
  random_state=20261104)` over sorted unit ids — identical partitions for
  the lifetime prior, all model arms, H1, and every compared baseline that
  we run ourselves.
- Within each outer fold: scaler fit on outer-train only.
  - FD001/FD003: `SensorMinMaxScaler` on the 14 sensors.
  - FD002/FD004: `OperatingConditionScaler` (6 condition clusters,
    random_state=7+fold) — per-condition affine scaling fit on outer-train.
- No engine contributes windows to both sides of any comparison; engines are
  the statistical units for every confidence interval.

## 3.4 Lifetime prior construction (for context; details in Section 5 [GATED])

- Estimator: `HistGradientBoostingRegressor` (max_iter=200, lr=0.05,
  max_leaf_nodes=15, min_samples_leaf=12, l2=10.0, early_stopping=False),
  5-seed ensemble (11/22/33/44/55) → μ_T = mean, σ_T = std.
- Features: causal prefix statistics at cutoff c (age; last/mean/std of ops;
  late/recent/early sensor moments + slopes) — rows 1..c only.
- Nested cross-fitting: inner 4-fold inside outer-train produces OOF priors
  for training arms; the outer-val ensemble prior is produced by estimators
  that never saw outer-val engines, labels, or total lifetimes.
- Causal staircase: a window ending at t consumes the latest cutoff
  prediction with c ≤ t (no interpolation to future cutoffs); t < 50 (below
  the minimum cutoff) uses the outer-train median lifetime and is flagged;
  flagged-window fraction is reported.
- Leakage audit: four unit tests (perturb-future-sensors; perturb-val-label →
  train prior unchanged; perturb-val-label → own prior unchanged; cutoff ≤ t
  assertion) — `tests/test_prior_causality.py`, all passing on synthetic and
  real data paths.

## 3.5 Training budget (identical across all neural arms)

| item | value |
|------|-------|
| backbone | CandidateModel: depthwise temporal conv → sensor attention → 1-layer Transformer (d=64) → MLP head |
| params (M0) | 42,306 |
| optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| schedule | cosine annealing to 1e-5 over max epochs |
| max epochs / early stop | 80 / 20 (checkpoint = best val RMSE) |
| batch | 256 |
| loss | MSE + 0.5·mean(asymmetric late penalty, weight 1.8) |
| seeds | 101 / 202 / 303 (development); identical for every arm |

## 3.6 Metrics

Primary: RMSE (dev protocol: pooled over outer-val windows of all 5 folds;
per-subset reported separately). Deltas between arms: engine-clustered paired
bootstrap, 5,000 resamples, 95% CI — windows are never treated as independent
samples. Secondary (reported for final candidates): MAE, NASA exponential
score, bias, late fraction (pred > true), severe-late fraction, and
short/mid/long lifetime tercile stratification by engine true lifetime.
