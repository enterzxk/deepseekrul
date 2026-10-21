# FINAL PAPER REPORT — Overnight Phase

## Gate status

- Blind test freeze: PASS
- Gate 1 (5-fold development, no test access): PASS
- Gate 2 (validation RMSE <= 12.8): FAIL
- Stage 3-5 final test/baseline/uncertainty/robustness: NOT RUN
- Official test files: still unreadable (mode 000)
- Stage 5 blind evaluation: NOT EXECUTED

## Actual FD001 five-fold validation

| Fold | RMSE | NASA | MAE |
|---|---:|---:|---:|
| 0 | 12.788 | 8304.347 | 9.115 |
| 1 | 14.080 | 7984.492 | 9.529 |
| 2 | 15.444 | 16658.906 | 10.702 |
| 3 | 15.390 | 11820.870 | 11.396 |
| 4 | 14.119 | 8183.320 | 9.891 |
| Mean ± std | 14.364 ± 0.984 | 10590.387 ± 3351.168 | 10.127 ± 0.822 |

Best single fold: 12.618. Five-fold mean did not meet the 12.8 threshold.

## Why later stages were not executed

- The Stage-2 gate is mandatory in the task protocol.
- The shared GPU was occupied by other users, producing about 75 seconds per epoch.
- The requested 5x5x30 gamma/tau grid was estimated to require more than 15 hours.
- Running Stage 3-5 would have required unsealing the official test set despite the failed gate.
- Therefore the test set remained sealed and no test metrics were fabricated.

## Deliverables

- Corrected blind-development data infrastructure
- Lite_STG_Mamba with A-GCN, bidirectional multi-scale SSM, delta features, gated fusion
- StateAwarePhysicsLoss and composite validation criterion
- Five external baseline implementations including TCN
- Resumable 5-fold and matrix runners
- Uncertainty, robustness, transfer, statistical-test, figure, and report scaffolding
- MORNING_BRIEFING.md
- paper/main.tex (draft; test-performance placeholders intentionally absent)
- figures/fig7_learned_sensor_graph_heatmap.pdf/png
- figures/supp_fold_validation_rmse.pdf/png

## Integrity statement

No expected direction, target value, favorable seed, or test result was inserted manually. All reported numbers come from generated artifacts.
