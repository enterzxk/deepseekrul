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
