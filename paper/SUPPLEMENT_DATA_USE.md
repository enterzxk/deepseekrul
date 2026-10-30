# Supplement — Data use (approved wording, 2026-10-29)

Status: wording approved by the human on 2026-10-29; to be placed in the
Supplement, with one sentence carried into Limitations.

## Data use

> **Data use.** All reported results were obtained from the official C-MAPSS
> training trajectories only. The official test files (`test_FD001`–`FD004`,
> `RUL_FD001`–`FD004`) were not used for model selection, hyperparameter
> tuning, calibration, or any metric reported in this paper. For transparency,
> we note that one exploratory script read `test_FD001` and `test_FD002` once
> to count engines; it produced no predictions and does not enter any result.
> FD003 and FD004 were not used at all.

## Sentence to carry into Limitations

> One exploratory script read the official FD001 and FD002 test files once to
> count engines; it produced no predictions and no reported number depends on
> it (Supplement, Data use).

## Implementation notes

* The claim must remain literally true. Any future script that touches a test
  file for any purpose must be added to this disclosure before submission.
* FD003/FD004 remain sealed; the 509-engine projection in §4.2 is an
  arithmetic extrapolation, not a measurement, and must be labelled as such.
