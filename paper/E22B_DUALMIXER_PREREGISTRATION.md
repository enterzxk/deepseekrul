# E22b — Fair re-implementation of a strong published baseline (Dual-Mixer)

Status: pre-registered 2026-10-31T23:35+08:00, before any E22b training run started.
sha256 recorded in `paper/E22_LEDGER.md` after freezing.

## Why this experiment

E21/E22a compare six architectures we implemented ourselves, and a descriptive table of
seven 2025–26 published numbers. Neither answers the reviewer question: *does the
resolution picture survive when a recent, independently published, strong model is run
under the identical protocol?* E22b re-implements one such model with its own code and
its own hyperparameters, changing nothing except the protocol elements that must be
shared for the comparison to mean anything.

## Selected method

**Dual-Mixer + FSGRI** (Fu, Hu, Peng, Chu, *Reliability Engineering & System Safety*,
2024, doi 10.1016/j.ress.2024.110398; code `github.com/fuen1590/PhmDeepLearningProjects`,
commit cloned 2026-10-31). Selection rationale: independent peer-reviewed venue, code
public and runnable, configuration readable in `models/RULPrediction/experiments.py`,
and it reuses the same 14 informative sensors (`dataset/cmapss.py:DEFAULT_SENSORS`)
that our pipeline uses — so "unified input information" is achievable without
reinterpreting the method.

Runner-up, not run in this round: the IEEE TIM 2025 two-phase model
(`github.com/mjliu-advertising/RUL`) — its repository exposes only two scripts with no
configuration file, so its original hyperparameters cannot be verified from the code.
Recorded as a candidate for a later round if a reviewer requests a second strong model.

## What is kept from the original (frozen, not tunable)

* Architecture: `DualMLPMixer`, `hidden_dim=32`, `num_layers=6`, `dropout=0`. FSGRI
  contrastive training **off** (`contra_training=False`, the repository default in
  `experiments.py`); the plain MSE path is the published model without the auxiliary
  loss, and turning FSGRI on would add a second design change we cannot verify.
* Optimiser Adam, `lr=1e-3`, `batch_size=1024`, `epochs=100`, `early_stop=5`,
  loss MSE, on the repository's own training loop (`train/trainable.py`).
* Label handling: RUL target capped at 125 (repository `rul_threshold` mechanism with
  `label_norm=False`; the repo default normalises to [0,1], which is a loss-scaling
  choice — we report RUL in cycles and therefore keep the label in cycles).

## What is unified with our protocol (the point of the experiment)

* **Split**: the same engine-level 5-fold partition as every other experiment
  (`SPLIT_SEED 7 + 20261020`, fold k holds out its own engines; fold sizes as in
  `E20_FD0034_CV_PREREGISTRATION.md`). The repository's own 20% random-window
  validation is replaced — a window-level split leaks engines between train and val,
  and would not be comparable with any of our numbers.
* **Inputs**: the same 14 sensors, the same two scalers (`SensorMinMaxScaler` for
  FD001/FD003, `OperatingConditionScaler` with 6 clusters for FD002/FD004), fitted on
  the fold's training engines only, applied unchanged to validation and test.
* **Window**: 50 cycles (repository default 30). Recorded as a deliberate deviation:
  the window is part of the input definition and must match our protocol; a 50-cycle
  window is within what a mixer can process unchanged.
* **Test protocol**: the E22a protocol — all 707 test engines, last observable window
  with first-frame padding below 50 cycles, truth capped at 125 for the headline
  metric, uncapped reported alongside, one prediction per engine, ensemble over the
  five fold models.
* **Seeds**: 2027, 2028, 2029 (the same three as every other experiment).

## Prohibitions

No hyperparameter may be changed after seeing any validation or test number. No
architecture variant, no FSGRI variant, no window variant. The official test files are
read once, after all 15 training runs (5 folds × 3 seeds) are complete and the
checkpoints archived. If Dual-Mixer loses, wins, or ties, all three are reportable
outcomes and none triggers a re-run with different settings.

## Pre-registered expectations

P1. Dual-Mixer's dev-side RMSE will land within 1.5 RMSE of arm A on FD001 and FD004,
    and the pair will not be separable at the 1.5 margin on any single subset — the
    resolution argument predicts exactly this for two independently designed models.
P2. On the official test (capped, 707 engines) the ordering will differ from the dev
    side on at least one subset, again by the resolution argument.
P3. Dual-Mixer's parameter count will exceed arm A's 41,858 by at least 3× while its
    test RMSE stays within the same 1.5-RMSE band.

Failed predictions are reported.

## Outputs

`experiments/e22b/{arm}_s{seed}_f{fold}/{subset}/` checkpoints and metrics;
`results/dsh/e22b/dev_metrics.json`; `results/dsh/e22b/test_full.json` and per-engine
predictions; `scripts/dsh_e22b_dualmixer.py` (adapter) and
`scripts/dsh_e22b_analyze.py`.

## Amendment 1 (2026-10-31T00:20+08:00) — input normalisation, decided before any test read

The first adapter run (FD001 fold 0, seed 2027) failed to fit: train RMSE 37 cycles,
val RMSE 43, over 16 epochs with the registered hyperparameters. Diagnosed as an input
conditioning problem, not a tuning problem: the project's `SensorMinMaxScaler` maps
several near-constant FD001 channels to values in [0,1] with a range of ~1e-2, which the
mixer's two sigmoid gates per layer (initialised at 0.5, six layers deep) attenuate to
nothing. The same model, same hyperparameters, on per-channel z-score input converges
normally: train RMSE 9.2, val RMSE 15.1 after 28 epochs.

**Decision, taken before any test number was read:** E22b uses per-channel z-score
normalisation, computed from the fold's training engines only and applied unchanged to
validation and test. This is the C-MAPSS convention used by the majority of published
models (including the Dual-Mixer repository's own `cmapss.get_data` scalar) and it is
applied identically to both sides of the comparison: a **paired candidate arm A-zscore**
(identical architecture and training budget to arm A, only the input scaler changed to
z-score) is trained alongside Dual-Mixer for every (subset, fold, seed) cell. The E21 /
E22a result tables are unaffected — they use their own registered scaler and are not
compared cell-by-cell with E22b. No hyperparameter of Dual-Mixer changes.

This is an input-convention decision, not model selection: it was forced by a failed
fit, was resolved by a mechanism diagnosis, and is reported with both the failed and
the corrected numbers.
