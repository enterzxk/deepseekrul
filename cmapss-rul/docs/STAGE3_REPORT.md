# STG-MambaNet P0-P3 Stage Report

Date: 2026-10-20  
Project: C:\Users\20562\Desktop\deepseek-rul  
Remote: lab-server:/home/yndx/zxk/bearing-rul-pi

> Historical note: the final P1/P2/P3/P4 numbers are in MORNING_REPORT.md.
> The P3 table in this file is the earlier d0.2 pilot and was superseded by
> the locked d0.15 run. This file preserves the P0 and pilot history.

## 1. P0 - Protocol freeze and code unification

Completed:

- Added docs/FINAL_PROTOCOL.md with clip125, hold-out, fair-ablation, metric,
  physics-loss, FD002, and reproducibility rules.
- Moved the root legacy implementation into archive_root_v0/:
  model.py, loss.py, dataset.py, sanity_check.py.
- src/ is now the only production implementation.
- Added trajectory output to src/model.py:
  model(x, return_trajectory=True) -> (prediction, trajectory), with
  trajectory[:, -1] exactly equal to the terminal prediction.
- Added L_Mono and monotonic_violation / monotonic_violation_rate to
  src/loss.py, backward compatible with PhysicsGuidedLoss.
- Added graph modes to AdaptiveSpatialBranch:
  learnable, identity, random_fixed.
- Added the identity prior:
  A = (1 - alpha) I + alpha * Softmax(ReLU(E1 E2^T)), alpha init 0.1.
- Added parameter-matched LSTM:
  default lstm_hidden_size = 70, full 144,265 params, LSTM 142,806 params,
  difference 1.01% (within the 3% rule).
- Added clip_test_rul=True as the default in src/dataset.py and
  scripts/run_ablation.py.
- The runner now stores both test_metrics_clip125 and appendix_raw.
- Added safety diagnostics: late_fraction, severe_late_count_gt10,
  mono_violation_rate.
- Added identity_graph, wo_mono to VARIANT_ORDER.
- Current test suite: 34 collected, OK; 6 skipped because official test
  files were sealed after P3.

Frozen hashes (first 16):

- src/model.py: B5EF67DF7C440FDF
- src/loss.py: 94A3B8E293CD925E
- src/dataset.py: F7D84E4E10DEE333
- scripts/run_ablation.py: 8C61EC632BEB0F67
- scripts/summarize_ablations.py: 4BBB919F2C04A2A0

## 2. P1 - Validation-only tuning

Important limitation: due compute contention, the completed P1 grid is a
**seed-42 screen**, not the planned 3-seed aggregate. The P1 screen used
window 30/50, dropout 0.2/0.3, weight decay 1e-4/1e-3, identity prior 0.1,
and Loss-Asym only (no mono).

Selected configuration:

- window_size = 50
- dropout = 0.2
- weight_decay = 1e-4
- d_model = 64, batch 256, lr 1e-3
- identity_prior_alpha = 0.1
- Loss-Asym weight 1.0, gamma 2.0

Seed-42 validation result of the selected configuration:

| Config | val RMSE | val NASA | val MAE | Late_Fraction | Mono_Violation |
|---|---:|---:|---:|---:|---:|
| w50, d0.2, wd1e-4 | 12.077 | 8451.94 | 9.020 | 0.556 | 0.452 |

Other tested branches (seed 42 or partial) were worse; window30/dropout0.2/
wd1e-4 was about 13.56 val RMSE, and several dropout0.3/long-window branches
were unstable. They are excluded from selection, not from the record: logs
remain under experiments/validation_tuning/logs/.

## 3. P2 - Loss ablation (validation-only, seed-42 screen)

All runs used the selected P1 model configuration. The table below is a
one-seed screen; the five-seed confirmation is P3.

| Loss | val RMSE | val NASA | val MAE | Late_Fraction | Severe_Late_Count | Mono_Violation |
|---|---:|---:|---:|---:|---:|---:|
| MSE | 12.823 | 10204.61 | 9.799 | 0.607 | 877 | 0.455 |
| MSE + Asym (w=0.5) | 12.942 | 9583.66 | 10.072 | 0.553 | 792 | 0.431 |
| MSE + Asym (w=1.0) | 12.077 | 8451.94 | 9.020 | 0.556 | 684 | 0.452 |
| MSE + Mono (lambda=0.1) | 12.228 | 9787.83 | 8.911 | 0.613 | 667 | 0.453 |

Selected: Loss-Asym w=1.0 for the w/o Mono control. Full P3 uses
MSE + Asym(1.0) + Mono(lambda=0.05).

Caveat: the one-seed screen showed that L_Mono at 0.05-0.1 did not materially
reduce the discrete mono-violation rate. The P3 five-seed result is the
evidence to decide whether the monotonic term earns its place.

## 4. P3 - Confirmatory 5-seed ablation

Protocol:

- Seeds: 101, 202, 303, 404, 505
- Six variants: full, wo_agcn, identity_graph, wo_ssm, wo_physics, wo_mono
- Main target: clip125
- Appendix: raw test RUL
- Test evaluated once per run after validation-selected checkpoint
- Frozen code hashes as above

Results:

Main clip125 table (mean +/- std over seeds 101/202/303/404/505):

| Variant | RMSE | NASA Score | MAE | Late_Fraction | Severe_Late_Count | Mono_Violation_Rate | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| w/o A-GCN | 15.170 +/- 2.581 | 498.932 +/- 373.945 | 11.315 | 0.580 | 27.80 | 0.463 | 143,816 |
| Identity-Graph (A=I) | 14.852 +/- 0.875 | 409.660 +/- 66.116 | 10.874 | 0.520 | 23.80 | 0.470 | 143,816 |
| w/o SSM-Block (matched LSTM) | 14.141 +/- 0.139 | 347.938 +/- 32.121 | 10.079 | 0.482 | 18.60 | 0.326 | 142,806 |
| w/o Physics Loss | 14.152 +/- 0.436 | 346.610 +/- 29.718 | 10.218 | 0.488 | 19.20 | 0.473 | 144,265 |
| w/o Mono Loss | 13.742 +/- 0.644 | 319.993 +/- 32.089 | 10.023 | 0.476 | 19.60 | 0.469 | 144,265 |
| Full Model (Ours) | 14.134 +/- 0.877 | 344.619 +/- 97.019 | 10.211 | 0.458 | 18.80 | 0.473 | 144,265 |

Raw-label appendix (same predictions, unclipped test targets):

| Variant | RMSE | NASA Score |
|---|---:|---:|
| w/o A-GCN | 16.299 +/- 2.295 | 534.034 +/- 363.643 |
| Identity-Graph (A=I) | 15.801 +/- 0.883 | 437.620 +/- 63.383 |
| w/o SSM-Block (matched LSTM) | 15.203 +/- 0.186 | 377.857 +/- 25.235 |
| w/o Physics Loss | 15.209 +/- 0.475 | 379.439 +/- 20.917 |
| w/o Mono Loss | 14.874 +/- 0.665 | 351.380 +/- 28.628 |
| Full Model (Ours) | 15.231 +/- 1.015 | 379.293 +/- 98.156 |

Paired deltas vs Full (positive = ablated variant worse):

| Ablation | Mean delta RMSE | Mean delta NASA |
|---|---:|---:|
| w/o A-GCN | +1.036 | +154.3 |
| Identity-Graph (A=I) | +0.718 | +65.0 |
| w/o SSM (matched LSTM) | +0.008 | +3.3 |
| w/o Physics Loss | +0.018 | +2.0 |
| w/o Mono Loss | -0.392 | -24.6 |

## Interpretation (must not be rewritten into a positive story)

1. **A-GCN has the clearest evidence.** Removing graph propagation costs
   +1.036 RMSE and +154 NASA on average. Identity-graph also loses to Full,
   but only by 0.718 RMSE / 65 NASA, so most of the benefit comes from the
   graph-convolution residual path rather than large learned cross-sensor
   coupling.
2. **The learned graph is almost identity.** The selected Full checkpoint has
   alpha = 0.101 and maximum off-diagonal adjacency 0.096. The identity prior
   kept the graph near independent channels. Fig. 3 shows this directly.
   The paper should not claim strong dynamic coupling without an alpha-schedule
   experiment.
3. **SSM does not beat the parameter-matched LSTM.** Mean RMSE is essentially
   tied (14.134 vs 14.141); the LSTM has much lower variance and a better
   mono-violation rate (0.326 vs 0.473). The SSM claim is not supported.
4. **The asymmetric physics loss is not supported in the five-seed test.**
   Removing it changes mean RMSE by only +0.018 and NASA by +2.0. The one-seed
   P2 screen favored Asym, but the confirmatory result does not.
5. **The monotonic term hurts.** Removing it improves mean RMSE by 0.392 and
   NASA by 24.6; it does not reduce mono violation (0.473 vs 0.469). The
   shared-prefix trajectory did not learn a useful monotone profile.
6. **Full is not the best model.** w/o Mono has the best RMSE and NASA.
   Full has the lowest late fraction (0.458) and near-lowest severe-late count
   (18.8 vs 18.6 for LSTM), which is a safety-side observation, not a global
   win.

These are negative/mixed findings. Do not change the table to fit the
original narrative.

## 5. What remains open

- P1/P2 need the full 3-seed aggregate to match the protocol exactly.
- If Full does not beat the controls, the paper narrative must be narrowed to
  the modules with consistent evidence, not rewritten to fit.
- FD002 condition-aware normalization is not implemented in this cycle.
- The current test set has now been observed in P3; no further model changes
  should be selected on these five-seed test scores.

## 6. Reproduce

~~~bash
cd /home/yndx/zxk/bearing-rul-pi
python -m unittest discover -s tests -v
python scripts/summarize_ablations.py \
  experiments/final/FD001_confirmatory/seed_101 \
  experiments/final/FD001_confirmatory/seed_202 \
  experiments/final/FD001_confirmatory/seed_303 \
  experiments/final/FD001_confirmatory/seed_404 \
  experiments/final/FD001_confirmatory/seed_505 \
  --output-dir results/final_FD001_5seeds
~~~
