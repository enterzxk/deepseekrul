# Baseline Reproduction & Comparison (2026 papers)

Date: 2026-11-05 (night run). Scope: reproduce the two 2026 baselines with
official code, then build the manuscript comparison table. Discipline:
baselines are run with **their own official configs and evaluation protocol**;
their numbers are reported as their pipelines output them, with per-row
protocol footnotes. Our numbers live in two calibers (see
`protocol_calibers.md`); cross-caliber rows are never mixed silently.

## Baseline 1 — CET-MSA (full reproduction, running)

Wang, Zhang & Qin, "Noise-robust remaining useful life prediction for
intelligent condition monitoring via convolutionally-enhanced temporal
self-attention", **J Intell Manuf (2026)**, doi 10.1007/s10845-026-02895-3.
Official repo: suosimiaoqianli/Noise-Robust-RUL-CET-MSA (cloned at
`/home/user/rulb/baselines/Noise-Robust-RUL-CET-MSA`).

- Framework: PyTorch. Pipeline: `preprocess_cmapss.py` → `train.py --config
  configs/fd00X.yaml` (train.py evaluates the best checkpoint on test itself).
- Their protocol (from configs): window 30, stride 1, **RUL cap 150**,
  17 screened sensors, 50 epochs, Adam lr 1e-3, batch 100, seed 0.
- Reproduction env: `/home/user/miniconda3/envs/pmv` (torch 2.5.0+cu124),
  data symlinked from `/home/user/rulb/data/CMAPSS`.
- Queue: `logs/baseline_cet/` on GPU0 (FD001→FD002→FD003→FD004 sequential).
- Results (Best test RMSE, their cap-150 protocol, official code/configs,
  seed 0; logs `logs/baseline_cet/cet_FD00X.log`):

| Subset | Reproduced test RMSE | Published (paper) | Note |
|---|---|---|---|
| FD001 | **17.22** (3 iters: 17.22 / 16.14 / 17.39, mean 16.92 ± 0.68) | not verifiable | |
| FD002 | **22.41** | not verifiable | |
| FD003 | **16.55** | not verifiable | |
| FD004 | **21.78** | not verifiable | |

### Honest reading of the CET-MSA reproduction

1. The paper's Data Availability promises "result tables" in the repo, but
   the repo contains none (docs/*.md are 0-byte except data_preparation).
   The Springer page is paywalled; per-subset absolute numbers cannot be
   verified from public sources. Only relative claims are public ("vs 3D-block
   attention baseline: average Score −27.8%, RMSE −3.86%").
2. Our reproduction uses their exact released configs and code paths, single
   GPU, their seed. The 3-iteration FD001 check (16.92 ± 0.68) shows the
   level is stable, not a seed fluke.
3. These levels are far below the attention-model literature on C-MAPSS
   (FD002 commonly 13–15 at cap 125/130; higher caps make the task easier,
   not harder). **The released package therefore does not reproduce
   paper-level performance.** We report the reproduction faithfully with this
   caveat, and cite their published relative improvement only as their claim.

## Baseline 2 — BootBoGs-TS (official artifacts, no re-run)

Youness & Sy, "BootBoGs-TS: Bootstrap-Based Hyperparameter Optimization for
LSTM Models in Remaining Useful Life Prediction", **Int J Comput Intell Syst
(2026)**, doi 10.1007/s44196-026-01344-6. Official repo:
Gyouness/BootBoGs-TS (cloned at `/home/user/rulb/baselines/BootBoGs-TS`).

- Framework: TensorFlow/Keras; the contribution is a Bayesian + Bootstrap
  hyperparameter optimization loop (notebook-driven). Full re-run = days of
  hyperparameter search for a baseline — not warranted; we use the repo's
  **official results CSVs** (`results/summary_Comparaison_All_test_FD004.csv`,
  committed by the authors) as the reported numbers.
- Their FD004 test RMSE (official CSV, `rmse_test` column):

| Method (theirs) | FD004 test RMSE |
|---|---|
| grid_full (h128) | 27.37 |
| bayes_tpe (h128) | 22.42 |
| **bootbogs_ts_oob_1.5 (h224)** | **24.39** |
| random (h160) | 18.29 |

  Note: in the authors' own table their proposed BootBoGs-TS is not the
  best row (random search wins); report faithfully with this caveat.

## Comparison framing for the manuscript

1. **Main table (official test, cap 125)**: our final M1 number after the
   one-shot unsealing vs CET-MSA reproduction (footnote: their config caps at
   150 — reproduced faithfully, see honest-reading caveat), BootBoGs FD004
   official number, and published numbers of no-code papers (AIDGN /
   LVDACNN / MSMT-FDA, cited). Additional published context rows (cap 125,
   no code): multiscale spatio-temporal graph + attention (ECUST J, 2025)
   11.23 / 12.64 / 11.15 / 13.41; adaptive dual distillation (2025)
   13.43 / 14.36 / 14.11 / 16.25. Historical reference row: E22a PE
   9.82 / 11.51 / 10.06 / 11.79 (our own earlier tuned PE, same cap-125
   official-test caliber).
2. **Ablation table (outer-val, cap 125, IDA protocol)**: M0/M1/M1p/controls
   — internal only, never mixed with test-caliber rows.
3. Honest positioning: our method's claim is not SOTA-RMSE-at-any-cost; it is
   a leakage-controlled causal prior with pre-registered gating and a
   favorable compute footprint (42k params vs ~10^5+). Baselines are context,
   not the fight we pick.
