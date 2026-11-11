# FD001-only freeze rule (scope reduction, 2026-11-11)

The user asked to stop the multi-subset grid and keep a single dataset. The
FD001 8-config ablation was already complete (24/24 cells), so no new ablation
is run. The rule is the same as `FREEZE_RULE.md`, restricted to FD001:

1. Final model = the DSPT config with the lowest mean validation RMSE over
   seeds 101/202/303 on FD001, provided it beats DSPT-base.
2. The official FD001 test set is opened once for the final model, DSPT-base,
   and the two Vanilla baselines (raw, DSPT-features+cycle).
3. No FD002/FD003/FD004 number is used for selection or reported as a main
   result in this round.
4. The full ablation table is reported as validation evidence for FD001 only.
