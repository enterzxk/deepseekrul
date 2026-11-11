# Pre-registered freeze rule

Written 2026-11-11 **before** any protocol-fixed test number was computed.

1. **Eight-config DSPT ablation.** A config is eligible for the primary model
   iff (a) it beats DSPT-base on at least 3 of the 4 subsets, using the mean
   validation RMSE over seeds 101/202/303, and (b) its equal-weight mean
   relative validation RMSE across the four subsets is <= 0.
2. **Mechanism variants.** An MS/SA/HI variant is supported iff it beats its
   own reference on at least 3 of 4 subsets by a mean of at least 1%, and it is
   not worse than DSPT-base overall.
3. **Final model.** The eligible candidate with the lowest equal-weight mean
   relative validation RMSE; ties broken by fewer parameters.
4. **Fallback.** If nothing is eligible, the final model is DSPT-base.
5. **Test scope.** Only the final model, DSPT-base, DSPT-full, the two Vanilla
   baselines, and the supported mechanism variants are evaluated on the
   official test set. The full ablation table remains validation-based.

Rationale: the official test sets have been observed repeatedly in this
project. Restricting test evaluation to a small, pre-specified set prevents a
second round of test-based selection, while still giving a comparable headline
number and the Vanilla comparison.
