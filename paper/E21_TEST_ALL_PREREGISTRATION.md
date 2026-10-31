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

## Amendment 1 (2026-10-31T19:30+08:00) — truth capping

The registration's phrase "truth = official RUL files" was ambiguous about capping, and
the first implementation compared 125-capped predictions against **uncapped** official
truth. That manufactures an unavoidable error of (true − 125) on every engine whose
remaining life exceeds 125 — 40 of 235 FD002 engines (17%) and 49 of 228 FD004 engines
(21%) — and inflated FD002 from 11.96 to 23.39 and FD004 from 12.27 to 22.40 RMSE while
barely affecting FD001/FD003 (few engines above the cap). The community-standard
C-MAPSS convention, used by every paper this project compares against, caps the truth
at 125 exactly as the training target is capped.

**Remedy.** `evaluate_subset()` now clips truth to MAX_RUL before computing all metrics
(pred, late, severe, NASA all inherit the capped error). No model, checkpoint, fold or
seed is touched; predictions are bit-identical; only the evaluation target definition
changes, to the one registered for training. Found by a bucketed error analysis that
showed an RMSE of 50 on the cap segment of FD002 against a segment mean prediction of
116 — arithmetically impossible under a capped-truth reading. All E21 numbers in the
paper and CLAIMS (C41–C43) are recomputed under the capped convention. The uncapped
figures are retained in `results/dsh/e21/test_all_uncapped.json` for provenance, and
the cross-paper comparison table (§4.3.6) states the convention explicitly.
