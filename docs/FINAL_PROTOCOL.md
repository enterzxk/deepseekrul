# STG-MambaNet Final Protocol

Status: P0 draft, frozen after P2  
Date: 2026-10-20  
Owner: Methodologist (Gemini decision), implementation by DeepSeek

This document is the single source of truth for data, model selection, loss
selection, ablation, and final test reporting. If code and this document
disagree, this document wins and the code must be changed.

## 1. Label protocol

- Main protocol: clip125 = True.
- Training targets: min(125, last_cycle - cycle).
- Validation targets: same piecewise-linear cap of 125.
- Official test targets: min(125, RUL_FDxxx.txt).
- Appendix only: raw (unclipped) test metrics must also be stored under
  appendix_raw in metrics.json and in the final appendix table.
- Main tables use clip125 only. Raw numbers must never be mixed into the main
  table or used for model selection.

## 2. Single production codebase

- The only legal production code is src/dataset.py, src/model.py, src/loss.py,
  scripts/run_ablation.py, scripts/summarize_ablations.py.
- The former root-level duplicates model.py, loss.py, dataset.py,
  sanity_check.py are archived under archive_root_v0/ and must not be imported.
- The archived root implementation is a reference for trajectory/monotonic
  logic only; all future changes happen in src/.

## 3. Test hold-out policy

- The 12 runs with seeds 42, 67, 123 are archived as Development Pilot.
- During P1 and P2, the official test set must not be read, printed, scored, or
  referenced. Model selection, early stopping, and hyperparameter search use
  only engine-level validation splits (20% of training engines) and the
  validation metrics val_rmse_clip125 / val_nasa_clip125.
- P1/P2 scripts must not instantiate a test DataLoader.
- After P1/P2 freeze, P3 uses exactly five pre-registered seeds:
  101, 202, 303, 404, 505.
- The official test set is evaluated once per final run, after loading the
  validation-selected checkpoint. No further model changes are allowed after
  the first P3 test result.

## 4. Fair ablation design

### 4.1 Parameter matching

- The w/o SSM LSTM control must be within +/-3% of the Full Model parameter
  count. The matched hidden width and any output projection must be recorded
  in config.json.
- Parameter counts are checked automatically before training.

### 4.2 Graph topology controls

AdaptiveSpatialBranch supports graph_mode:

- learnable: A = (1 - alpha) I + alpha * softmax(ReLU(E1 E2^T)),
  alpha initialized to 0.1 and learned.
- identity: A = I.
- random_fixed: a fixed, row-stochastic random matrix stored as a buffer.

The identity-graph control distinguishes graph propagation from extra
parameters. The random-fixed control checks whether any fixed mixing is
sufficient.

### 4.3 Final P3 variants

1. Full Model: learnable graph + SSM + full physics loss.
2. w/o A-GCN: independent per-sensor branch + SSM + full physics loss.
3. Identity-Graph: A = I + SSM + full physics loss.
4. w/o SSM: learnable graph + parameter-matched LSTM + full physics loss.
5. w/o Physics Loss: learnable graph + SSM + pure MSE.
6. w/o Mono Loss: learnable graph + SSM + MSE + asymmetric term only.

## 5. Metrics and gates

Main metrics:

- RMSE (clip125)
- NASA asymmetric score (clip125)
- MAE
- Mean signed error (positive = late prediction)
- Late_Fraction: fraction of samples with y_pred > y_true
- Severe_Late_Count: number of samples with y_pred - y_true > 10
- Mono_Violation_Rate: fraction of valid adjacent trajectory pairs with
  y_hat_t > y_hat_{t-1}

Appendix metrics:

- RMSE and NASA on raw (unclipped) test targets.

No hard-coded target such as "force test RMSE <= 13.0" is allowed. The
validation objective is to minimize mean val_rmse_clip125 and reduce seed
variance. Final results must report all five seeds, mean, standard deviation,
and paired deltas versus Full. No seed may be dropped after seeing test.

## 6. Physics loss protocol

The production loss supports four pre-registered variants:

1. Loss-MSE: pure MSE.
2. Loss-Asym: MSE + asymmetric late-penalty.
3. Loss-Mono: MSE + trajectory monotonic penalty.
4. Loss-Full: MSE + asymmetric + monotonic.

The total objective is:

L = MSE + lambda_asym * L_Asym + lambda_mono * L_Mono

with L_Asym using weight 2.0 for late errors and 1.0 for early errors.
gamma/late_weight, lambda_asym, and lambda_mono are selected on validation
only. The trajectory for L_Mono is the per-cycle RUL trajectory returned by
return_trajectory=True. If no trajectory is supplied, L_Mono is exactly zero
and monotonic_available is recorded as 0.

## 7. Multi-condition extension

FD001 remains the primary development set. After FD001 is frozen and if
compute allows, FD002 is an external validity check only. FD002 preprocessing
must use condition-aware normalization (K-Means operating-condition clusters,
per-cluster sensor statistics fitted on training engines only). FD002 must not
be used to retroactively change the frozen FD001 design.

## 8. Reproducibility and audit

Every run stores:

- config.json (all hyperparameters, graph_mode, loss flags, seed)
- split_seed_<seed>.json (train/validation/test engine ids)
- scaler_stats.json (training-only statistics)
- history.csv (per-epoch validation metrics)
- best_model.pt (validation-selected checkpoint)
- test_predictions.csv (per-engine y_true, y_pred, signed error)
- metrics.json (main clip125, appendix_raw, safety diagnostics)

Every final table must be generated from metrics.json / predictions.csv by
scripts/summarize_ablations.py. Manual numbers are forbidden.

## 9. Freeze record

P1 (validation-only) selected:

- window_size = 50
- dropout = 0.2
- weight_decay = 1e-4
- d_model = 64, batch_size = 256, lr = 1e-3, grad_clip = 1.0
- epochs = 80, patience = 20
- graph_mode = learnable, identity_prior_alpha = 0.1
- temporal = SSM, kernels = 3/5/7
- validation split = 80/20 engine-level, seeds 42/67/123
- Compute-limited execution: P1/P2 were completed as a seed-42 screen; the
  five-seed P3 is the confirmatory evidence. Recorded as a limitation.

P2 (validation-only) selected:

- Loss-Asym weight = 1.0, gamma (late_weight) = 2.0
- Loss-Full uses asymmetric weight 1.0 and lambda_mono = 0.05
- Loss-MSE for w/o Physics; Loss-Asym for w/o Mono
- No test score was used for this selection.

P3 freeze hashes (SHA256, first 16 hex chars):

- src/model.py: B5EF67DF7C440FDF
- src/loss.py: 94A3B8E293CD925E
- src/dataset.py: F7D84E4E10DEE333
- scripts/run_ablation.py: 8C61EC632BEB0F67
- scripts/summarize_ablations.py: 4BBB919F2C04A2A0

P3 seeds: 101 202 303 404 505
P3 variants: full, wo_agcn, identity_graph, wo_ssm, wo_physics, wo_mono
