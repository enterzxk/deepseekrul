# Test-set deviation requested on 2026-11-11

The pre-registered rule opened the official test set only for the final model
and two baselines. At the user's explicit request, the test set is now also
evaluated for **all eight ablation configs on all four subsets, seed 101**.

Status of this deviation:

- It is **descriptive**, not selective: the final model remains MS+SA, chosen
  on validation before this evaluation.
- The test sets are no longer pristine for any future model selection. No
  further choice (module, window, hyperparameter) may be made from these test
  numbers.
- The ablation test table must be reported with this caveat, and the
  pre-registered FD001 final comparison remains the primary evidence.
- Manifest: `paper_b/results/FREEZE_ABLATION_SEED101.json`.
