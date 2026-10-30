# MORNING REPORT - STG-MambaNet Overnight Run

Generated: 2026-10-21T04:32:32

## Frozen hyperparameters

~~~json
{
  "phase": "P2",
  "selected_tag": "p1_w50_d015_wd1e-4",
  "window_size": 50,
  "dropout": 0.15,
  "weight_decay": 0.0001,
  "d_model": 64,
  "batch_size": 256,
  "lr": 0.001,
  "epochs": 80,
  "patience": 20,
  "seeds": [
    42,
    67,
    123
  ],
  "val_rmse_mean": 11.917152847594515,
  "val_rmse_std": 0.5343541951707039,
  "val_nasa_mean": 7659.008124478806,
  "full_loss_tag": "p2_full_l20_w10_m005",
  "full_loss": {
    "mode": "full",
    "asym_weight": 1.0,
    "lambda_mono": 0.05,
    "late_weight": 2.0,
    "val_rmse_mean": 12.390976841797167,
    "val_rmse_std": 0.7215494248617551,
    "val_nasa_mean": 8538.207735579464
  },
  "wo_mono_loss_tag": "p2_asym_l20_w10",
  "wo_mono_loss": {
    "mode": "asym",
    "asym_weight": 1.0,
    "late_weight": 2.0,
    "val_rmse_mean": 11.917152847594515,
    "val_rmse_std": 0.5343541951707039,
    "val_nasa_mean": 7659.008124478806
  },
  "wo_physics_loss": {
    "mode": "mse"
  },
  "selection_rule": "P1: min(val_rmse_mean + 0.25*std); P2: best full/asym by validation RMSE"
}
~~~

## FD001 five-seed confirmatory ablation (clip125)

| Variant | RMSE | NASA | MAE | Late_Fraction | Severe_Late_Count | Mono_Violation_Rate | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| w/o A-GCN | 14.104 +/- 1.200 | 369.635 +/- 102.242 | 10.352 | 0.530 | 24.60 | 0.465 | 143,816 |
| Identity-Graph (A=I) | 14.689 +/- 0.683 | 392.977 +/- 86.252 | 10.583 | 0.504 | 21.80 | 0.463 | 143,816 |
| w/o SSM-Block (matched LSTM) | 14.405 +/- 0.329 | 359.656 +/- 22.626 | 10.379 | 0.468 | 18.60 | 0.313 | 142,806 |
| w/o Physics Loss | 14.123 +/- 0.868 | 357.238 +/- 56.818 | 10.194 | 0.510 | 21.60 | 0.451 | 144,265 |
| w/o Mono Loss | 14.133 +/- 0.538 | 324.621 +/- 40.663 | 10.246 | 0.450 | 17.80 | 0.446 | 144,265 |
| Full Model (Ours) | 13.423 +/- 0.487 | 327.119 +/- 59.184 | 9.719 | 0.522 | 18.60 | 0.453 | 144,265 |

## Raw-label appendix

| Variant | RMSE | NASA |
|---|---:|---:|
| w/o A-GCN | 15.122 +/- 1.053 | 395.962 +/- 99.756 |
| Identity-Graph (A=I) | 15.667 +/- 0.534 | 423.210 +/- 83.312 |
| w/o SSM-Block (matched LSTM) | 15.428 +/- 0.355 | 389.163 +/- 21.521 |
| w/o Physics Loss | 15.086 +/- 0.938 | 384.123 +/- 53.159 |
| w/o Mono Loss | 15.321 +/- 0.633 | 360.064 +/- 37.463 |
| Full Model (Ours) | 14.456 +/- 0.468 | 354.360 +/- 55.360 |

## Paired deltas vs Full

| Ablation | Mean delta RMSE | Mean delta NASA |
|---|---:|---:|
| Identity-Graph (A=I) | +1.267 | +65.859 |
| w/o A-GCN | +0.682 | +42.516 |
| w/o Mono Loss | +0.710 | -2.498 |
| w/o Physics Loss | +0.700 | +30.119 |
| w/o SSM-Block (matched LSTM) | +0.982 | +32.538 |

## FD002 multi-condition external validity (clip125)

| Variant | RMSE | NASA | MAE | Late_Fraction | Severe_Late_Count | Mono_Violation_Rate | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| w/o A-GCN | 16.126 +/- 1.037 | 1385.647 +/- 88.676 | 11.631 | 0.463 | 52.33 | 0.494 | 143,816 |
| w/o SSM-Block (matched LSTM) | 14.363 +/- 1.059 | 1001.130 +/- 248.916 | 10.538 | 0.422 | 43.33 | 0.447 | 142,806 |
| w/o Physics Loss | 15.512 +/- 0.784 | 1513.909 +/- 215.770 | 10.958 | 0.528 | 59.00 | 0.491 | 144,265 |
| Full Model (Ours) | 14.955 +/- 0.715 | 1456.569 +/- 248.479 | 10.580 | 0.474 | 53.33 | 0.480 | 144,265 |

## Raw-label appendix

| Variant | RMSE | NASA |
|---|---:|---:|
| w/o A-GCN | 28.595 +/- 2.274 | 13182.474 +/- 5361.914 |
| w/o SSM-Block (matched LSTM) | 26.776 +/- 0.771 | 8642.804 +/- 1556.024 |
| w/o Physics Loss | 26.520 +/- 0.320 | 7719.224 +/- 1377.675 |
| Full Model (Ours) | 26.568 +/- 0.929 | 9678.185 +/- 5507.393 |


## Key findings

- FD001 Full has the best RMSE (13.423) and MAE (9.719), but not the best NASA score: w/o Mono is slightly better (324.621 vs 327.119). Do not call Full uniformly best.
- A-GCN has the strongest positive evidence on FD001: removing graph propagation costs +0.682 RMSE / +42.516 NASA on average. Fixed identity graph loses +1.267 RMSE / +65.859 NASA.
- SSM beats the matched LSTM on FD001 (+0.982 RMSE / +32.538 NASA when removed), but the LSTM has much lower RMSE variance (0.329 vs 0.487).
- The physics objective beats pure MSE on FD001 (+0.700 RMSE / +30.119 NASA when removed), but the effect is modest.
- L_Mono improves FD001 RMSE by 0.710 when removed, but slightly improves NASA by 2.498 and lowers late fraction from 0.522 to 0.450. Mono helps accuracy, not the safety-side metric.
- FD002 reverses the temporal conclusion: Full is 14.955 RMSE / 1456.569 NASA, while the matched LSTM is 14.363 / 1001.130. A-GCN remains useful (w/o A-GCN 16.126 / 1385.647).
- FD002 raw-label errors are high (Full 26.568 RMSE / 9678 NASA); condition-aware FD002 is not yet competitive on raw labels.
- FD002 uses K-Means condition-aware normalization fitted on training engines only.

## Artifacts

- docs/FINAL_PROTOCOL.md
- docs/STAGE3_REPORT.md
- results/final_FD001_5seeds/
- results/final_FD002_3seeds/ (if present)
- results/final_tables.tex
- figures/final/
- logs/overnight_progress.log

## Limitations

- All official test scores in P3/P4 are confirmatory; do not use them for further model selection.
- FD002 raw-label errors are high; clip125 is the main protocol, raw is appendix only.
- FD002 condition clusters are data-driven, not physically labeled.
- Full is not uniformly best; the paper narrative must be narrowed to the modules with consistent evidence (A-GCN) and must report SSM/Mono tradeoffs honestly.
- The identity-prior graph may be close to identity; report the learned alpha and off-diagonal mass before claiming dynamic coupling.
