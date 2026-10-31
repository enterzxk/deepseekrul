# E20 — FD003 and FD004 five-fold cross-validation

Status: **FROZEN 2026-10-31T11:40+08:00**, before any FD003/FD004 training run exists.
Authorisation: `paper/FD0034_PROTOCOL_CHANGE.md`.

## Design (identical to FD002/FD001 in every respect)

| item | FD003 | FD004 |
|---|---|---|
| engines | 100 (1 operating condition) | 249 (6 operating conditions) |
| fold size | 20 | round(0.2 x 249) = 50 (fold 4 = 49) |
| arms | A, BE, PE | A, BE, PE |
| seeds | 2027, 2028, 2029 | 2027, 2028, 2029 |
| budget | 80 epochs, patience 20, d_model 64, num_layers 1, ref-mode global | same |
| runs | 45 | 45 |
| condition-aware preprocessing | no | yes (as FD002) |

Estimator: `dsh_e16_arch_corrected.py` convention — seed 2027 for intervals, 100k clustered
bootstrap within fold, three RNG seeds, delta = RMSE(first) - RMSE(second), TOST at 1.0.

## Pre-registered predictions

| # | prediction | if it fails |
|---|---|---|
| 1 | FD003 (single condition, like FD001): BE - A positive; may or may not clear zero at 100 engines | sign reversal means the time effect is regime-dependent |
| 2 | FD004 (six conditions, like FD002): BE - A positive and significant, magnitude 1-3 RMSE | the FD002/FD001 agreement was a coincidence |
| 3 | PE - A is sub-resolution on both (CI includes zero) | a bounded encoding wins somewhere; that becomes a positive result and changes 4.3 |
| 4 | pooled over the four subsets, BE - A is significant with the CI narrowing as 1/sqrt(n) | if it widens, the subset effects are heterogeneous and must be reported separately |

## Prohibitions

No tuning after seeing a fold; no dropping an arm or seed; no changing the margin; FD004's
OperatingConditionScaler stays as implemented; test files are not touched until the
validation analysis is complete and the models are frozen.

## Artifacts

`experiments/e20/{FD003,FD004}/{arm}_s{seed}_f{fold}/`, `results/dsh/e20/E20_cv.json`,
`scripts/dsh_e20_fd34_analyze.py`.

## Amendment 1 (2026-10-31T14:45+08:00)

Two launch defects, both found before any FD003/FD004 result was analysed.

1. FD003 batch ran one run instead of 45: the launcher passed all 45 four-field job
   lines to a single bash -c invocation because xargs -a was used without -n. One valid
   run exists (A, seed 2027, fold 0); the other 44 are relaunched with the corrected
   three-field form.
2. FD004 fold 4 failed 9/9 on an assertion: prepare() asserted every fold has exactly
   n_val engines, but 249 engines cannot be split into five folds of 50. The design
   table above already declared fold 4 = 49, so the code contradicted the registration
   and the assertion is relaxed to allow a smaller final fold. FD004 folds 0-3 (36 runs,
   all rc=0) are unaffected; fold 4 is rerun with 49 validation engines.

No completed run is invalidated: the permutation, the scaler seeds and every fold 0-3
partition are unchanged. A third defect (a corrupted character introduced while editing
run_candidate.py, caught by py_compile before any job started) left the queue idle for
about 25 minutes and cost no runs.

## Amendment 2 (2026-10-31T17:30+08:00)

Architecture deviation found and fixed, discovered during E21 preflight (checkpoint
loading), after all E20 analysis outputs existed.

**Deviation.** Exactly one of the 90 e20 runs — FD003/A/seed2027/fold0, the single run
produced by the Amendment-1 xargs defect — was trained with `num_layers=2` (75,330
parameters): its launcher line predated the corrected queue form and omitted
`--num-layers 1`, which defaults to 2. The other 89 runs carry the registered 1-layer
architecture (A 41,858 / BE 41,730 / PE 42,306). Found by parameter-count comparison
against `metrics.json` on every checkpoint.

**Impact on the original analysis.** None of the E20 numbers were computed from
checkpoints: `dsh_e20_fd34_analyze.py` reads each run's own
`matched_cutoff_predictions.csv`, so every original number reflects the architecture
that actually produced it. The deviation contaminated 1 of 45 runs pooled into the
FD003 estimate.

**Remedy.** The deviant run directory was preserved at
`experiments/e20/FD003/A_s2027_f0_LAYERS2_DEVIATION/` and the run retrained with the
registered configuration (identical command plus `--num-layers 1`; deterministic split
SPLIT_SEED 7+20261020, seed 2027, so the same train/validation partition). Retrained
run: val RMSE 11.772, 41,858 params. FD003 was then re-analysed from the regenerated
predictions. Fixed-architecture FD003 results: A 10.934 / BE 9.619 / PE 9.982 (pooled
RMSE, seed 2027); BE−A −1.315 [−2.949, +0.316]; PE−A −0.952 [−2.438, +0.276]. The
qualitative conclusions are unchanged: no significant BE−A or PE−A effect on FD003,
direction reversed relative to the other three subsets. E20_cv.json now contains only
fixed-architecture numbers.
