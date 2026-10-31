# E21 — Unified official-test evaluation, FD001-FD004

Status: **FROZEN 2026-10-31T11:50+08:00**, before any test file other than the already-reported
FD002 one-shot is read. Authorisation: `paper/FD0034_PROTOCOL_CHANGE.md`.

## Protocol (one rule for all four subsets)

* One prediction per engine: the last observable 50-cycle window.
* Sensor scaling: the fold's training scaler (OperatingConditionScaler, 6 clusters, fitted on
  that fold's 80% training engines), applied unchanged to the test trajectory.
* Extras: built exactly as `WindowSet` builds them (14-dim reference baseline from the
  trajectory's own first cycles, the scaled age channel, 6 evidence features = 21 dims).
* Prediction clipped to [0, 125]; truth is the official RUL file, uncapped.
* Metrics: RMSE, MAE, asymmetric NASA score, late (e > 0), severe (e > 10).
* Engines with fewer than 50 observed cycles are excluded and counted (FD001 ?, FD002 24/259,
  FD003 3, FD004 20); their RUL distribution is reported with the results.

## Models

A, BE, PE only; 15 frozen checkpoints per arm per subset (3 seeds x 5 folds), from
`e3_fd001`+`e18` (FD001), `e3`+`e10/e10b` (FD002), `e20` (FD003, FD004). No other
architecture is evaluated. No model is retrained.

## What is computed

1. Per-model metrics, then the mean over the 15 models (comparable to the FD002 one-shot).
2. An ensemble prediction per engine (mean over the 15 models), with a 95% interval from an
   engine-level bootstrap - the interval the FD002 one-shot lacked.
3. A training-side vs test-side retention table: for every contrast, the training-side
   estimate divided by the test-side gap.
4. The FD002 one-shot (23.958) is retained as history; the unified-protocol FD002 number is
   reported beside it as a same-protocol cross-check, expected to agree to within the
   model-set difference (12 vs 15 checkpoints).

## Prohibitions

No architecture, seed, fold or hyperparameter may be chosen from these numbers. If a model is
changed later for any reason, the resulting test numbers are labelled exploratory and are
never presented as independent confirmation.

## Artifacts

`results/dsh/e21/test_all.json`, `scripts/dsh_e21_test_all.py`; per-engine ensemble
predictions for every subset.
