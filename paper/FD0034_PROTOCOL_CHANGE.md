# Protocol change — unsealing FD003/FD004 and the official test files

**Authorised 2026-10-31T11:28+08:00** by explicit human instruction:
"调整实验策略：不再长期封存官方测试集……完成 FD001–FD004 的完整训练、验证与官方测试".
This supersedes the "FD003/FD004 remain sealed" clauses of
`paper/TEST_SET_PROTOCOL_CHANGE.md` and `paper/SUPPLEMENT_DATA_USE.md`; those files are kept
as the record of the earlier boundary and are not rewritten.

## What changes

1. **Training side.** FD003 and FD004 training engines enter the five-fold cross-validation
   with the existing arms (A, BE, PE), seeds (2027-2029), budget (80 epochs, patience 20,
   d_model 64, num_layers 1) and estimator. No new architecture, no tuning
   (`paper/E20_FD0034_CV_PREREGISTRATION.md`).
2. **Test side.** `test_FD001..FD004` and `RUL_FD001..FD004` are read under one unified
   protocol (last observed window per engine, cap 125, RUL truth from the official files).
   Models are frozen before any test read; no architecture or hyperparameter may be chosen
   from test results (`paper/E21_TEST_ALL_PREREGISTRATION.md`).
3. **History.** The 2026-10-31 one-shot FD002 result (A 23.958 on 235 engines) is retained as
   the historical record and is not overwritten; the unified-protocol numbers are reported
   alongside it and labelled as such.

## What does not change

* Window 50, cap 125, asymmetric NASA score, severe = e > 10.
* The matched-cutoff validation protocol and the 5-fold design.
* The rule that test results cannot select anything. If a model is later changed for any
  reason, the resulting numbers are labelled exploratory and never presented as independent
  confirmation.

## Known exclusions under the 50-cycle window rule

| subset | engines | excluded (< 50 cycles) | retained |
|---|---:|---:|---:|
| test_FD001 | 100 | to be counted at run time | |
| test_FD002 | 259 | 24 (documented, §6.10) | 235 |
| test_FD003 | 100 | 3 | 97 |
| test_FD004 | 248 | 20 | 228 |

Excluded engines are disclosed per subset together with their RUL distribution, as done for
FD002 in §6.10.
