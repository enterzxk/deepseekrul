# DSPT negative ablation: mechanism analysis

Status: analysis in progress. Numbers marked `[RESULT]` are filled only from
`paper_b/results/raw/*/metrics.json` and `test_metrics.json`.

## 1. The observation to explain

The historical 32-cell ablation (single seed, fold 0, pre-fix code) reported:

| subset | base test RMSE | FULL test RMSE | FULL vs base |
|---|---:|---:|---:|
| FD001 | 10.60 | 12.70 | +19.8% |
| FD002 | 11.40 | 11.42 | +0.2% |
| FD003 | 11.95 | 10.23 | -14.4% |
| FD004 | 12.36 | 12.17 | -1.6% |

Source: `archive/legacy_2026_11/results/abl_test2.csv`. These numbers are
**not** paper evidence: the evaluator that produced them cannot load the
checkpoints it is paired with (see `PROTOCOL_AUDIT.md` P1-4), and the test set
was observed repeatedly. They are used only to define the question.

## 2. Hypotheses

### SA (cross-sensor Transformer)

- **H-SA1 (condition heterogeneity).** A single condition-agnostic cross-sensor
  attention mixes sensors whose correlation structure changes across operating
  conditions. This should hurt FD002/FD004 more than FD001/FD003.
  Probe evidence: mean absolute difference between per-condition sensor
  correlation matrices divided by mean within-condition correlation is
  **FD001 0.025, FD003 0.032, FD004 0.257, FD002 0.468** (10-19x larger on the
  multi-condition subsets).
- **H-SA2 (missing sensor identity).** The SA input tokens carry no sensor
  identity: the shared `sensor_in` projection and the attention are
  permutation-equivariant, so the layer cannot learn sensor-specific roles.
  Adding a learned per-sensor embedding should recover part of the loss.

### HI (health-indicator auxiliary supervision)

- **H-HI1 (noisy target).** The PCA-1 HI is globally monotone but locally
  noisy. Probe: per-engine Spearman(HI, cycle) = -0.92/-0.88/-0.88/-0.79
  (FD001/2/3/4), but **~48% of adjacent steps move against the global trend**;
  R^2(HI ~ RUL) = 0.61-0.78. A hard monotone penalty therefore fights the
  target itself, and the auxiliary task is largely redundant with RUL.
- **H-HI2 (representation mismatch).** The main RUL output uses `h_f` (a
  gate between the short-span `h_A` and the full-span `h_B`), while in
  `mode=long` the evaluated output is `h_B`. If HI reads `h_f` it optimises a
  representation that the evaluated head does not use. Reading HI from `h_B`
  should align the gradients.
- **H-HI3 (monotone sign).** The legacy penalty `relu(h_t - h_{t-1})`
  punishes increases. Because HI is a *health* index (decreasing), this
  direction is correct; the defect is the lack of masking, not the sign. An
  inverted-sign run is included only to quantify the sensitivity.

### MS (multi-scale convolution)

- **H-MS0 (the scales are redundant).** The degradation signal is smooth on a
  ~25-cycle scale: the first-PC autocorrelation is 0.76-0.82 at lag 7 and
  0.53-0.60 at lag 20, crossing 0.5 at lag 23-27 (FD001/2/3/4). The kernels
  k=3/5/7 therefore all sit well inside the correlation length and see almost
  the same low-frequency content. "Multi-scale" is local smoothing, not
  multi-scale in the degradation sense; the temporal Transformer already has a
  global receptive field.
- **H-MS1 (capacity/projection, not scale).** The legacy "MS on" branch uses
  three kernels **and** an extra linear projection, while "MS off" uses a
  single k=3 convolution. The comparison confounds multi-scale with capacity.
  Controls: `single_proj` (one kernel + projection), `multi_noproj` (three
  scales, no projection), `multi3` (three k=3 kernels + projection,
  parameter-matched to full).

### The archived ranking survives the P0 evaluation fixes

Recomputing the archived checkpoints on all four subsets with the frozen
evaluator (same checkpoints, P0-fixed windows/EMA/MS):

| subset | archived base | archived FULL | recomputed base | recomputed FULL | ranking |
|---|---:|---:|---:|---:|---|
| FD001 | 10.60 | 12.70 | 10.92 | 12.33 | FULL worse |
| FD002 | 11.40 | 11.42 | 11.19 | 11.20 | tie |
| FD003 | 11.95 | 10.23 | 11.54 | 9.88 | FULL better |
| FD004 | 12.36 | 12.17 | 12.94 | 12.30 | FULL better |

Source: `results/legacy_all_repro.csv`. The absolute values move (the fixes are
not neutral), but the **ranking does not flip**. Therefore the historical
negative ablation is not explained by the P0 evaluation fixes alone. The
remaining candidate explanation is the training/model-selection protocol
(window-level vs engine-level early stopping), which the new 8-config grid
tests directly.

### Where the historical FULL fails

Reconstructing the archived FD001 test predictions (same checkpoints, frozen
evaluator) shows the FULL deficit is concentrated in the high-RUL regime:

| true RUL bin | n | base RMSE | base bias | FULL RMSE | FULL bias |
|---|---:|---:|---:|---:|---:|
| 0-50 | 30 | 2.61 | +0.91 | 3.07 | -2.21 |
| 50-80 | 15 | 14.18 | +5.17 | 10.85 | +0.46 |
| 80-100 | 21 | 12.49 | +5.52 | 14.90 | -0.15 |
| 100-125 | 23 | 13.87 | -2.38 | 14.84 | -5.69 |
| 125+ | 11 | 9.57 | -6.78 | 17.63 | -12.41 |

Source: `results/legacy_preds_bins.csv`. FULL systematically under-predicts
high-RUL engines (bias -12.4 at 125+). This is consistent with the asymmetric
loss (late predictions penalised 1.8x) plus a validation metric that
under-weights the high-RUL regime (see `PROTOCOL_AUDIT.md` §3a).

### Protocol effects

- **H-P0.** Part of the historical 10.91 -> 12.70 change is protocol, not
  model: unmasked temporal Transformer, legacy RUL endpoint, global (leaky) HI,
  zero-padded MS convolution, and EMA initialised at the zero pad. The
  attribution grid toggles each fix one at a time on FD001/seed 101.

## 2b. Active parameter counts

Modules are built conditionally, so these are active parameters (no dead
branches), unlike the historical 115,716 count that included disabled modules.

| config | params | delta vs base |
|---|---:|---:|
| base | 77,058 | - |
| MS | 82,114 | +5,056 |
| SA | 110,530 | +33,472 |
| HI | 77,123 | +65 |
| MSSA | 115,586 | +38,528 |
| MSHI | 82,179 | +5,121 |
| SAHI | 110,595 | +33,537 |
| FULL | 115,651 | +38,593 |

Source: `metrics.json` of each run. SA accounts for 87% of the FULL delta, so
any SA comparison must also report the capacity difference.

## 3. Design

All runs: L=70, 4 subsets, seeds 101/202/303, 40 epochs, patience 12, batch 512,
engine-level validation (5 cuts), no EMA selection. Grids:
`ablation8.json`, `ms_mech.json`, `sa_mech.json`, `hi_mech.json`,
`attribution.json`.

Controls:

| question | comparison |
|---|---|
| does MS help? | base (k=3, no proj) vs ms (k=3/5/7 + proj) |
| is it multi-scale or capacity? | single_proj, multi_noproj, multi3 vs ms |
| does SA help? | base vs sa (one cross-sensor layer) |
| why? | sa vs sa+identity, sa vs sa+condition embedding |
| does HI help? | base vs hi (h_f, correct monotone) |
| why? | hi vs h_f no-mono, h_f inverted sign, h_B source |
| do they interact? | mssa, mshi, sahi, full |

## 4. Results

`[RESULT: ablation8 table, validation, mean +- std over 3 seeds, relative to base]`

`[RESULT: MS mechanism table]`

`[RESULT: SA mechanism table]`

`[RESULT: HI mechanism table]`

`[RESULT: attribution table]`

## 5. Decision rules

- A module is **supported** only if it beats its reference on at least 3 of 4
  subsets by >=1% mean validation RMSE and is not worse than base overall.
- A module that fails twice (ablation + mechanism) is dropped from the model.
- No module is kept because it is already in the code.
- The official test is not used for any of these decisions.

## 6. What would falsify the mechanism story

- SA: if `sa+identity` and `sa+condition` are both no better than `sa`, the
  identity/heterogeneity explanation is wrong; the cost is optimisation noise.
- HI: if `h_B` and `no-mono` are both no better than `h_f`, the
  representation/noise explanation is wrong; the auxiliary task is simply
  uninformative.
- MS: if `multi3` (parameter-matched, single scale) matches `ms`, the
  multi-scale claim is falsified; if `multi_noproj` matches, the projection is
  unnecessary.
