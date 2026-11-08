
## K. A pre-registered tail guard

The span schedule reduces the pooled error on both fleets, but on FD004 the asymmetric
score is not stable: in one seed and one fold a single engine is over-estimated by about
96 cycles, and because the score grows exponentially on the over-estimation side that
engine alone contributes 14695 to a fleet mean of 313, against 23 for the reference.
The maximum absolute error on that fold is the same for both arms (124.6 against 124.5);
only the sign differs.

We therefore pre-registered a guard before computing any guarded result (protocol 8.8):
when the two models disagree by more than tau cycles, keep the short-window prediction.
We fix tau = 50 cycles, about 3.7 times the typical FD004 error, and report tau in
{30, 50, 75} as a sensitivity check.

With tau = 50 and two seeds pooled, FD002 improves by 7.20% (MAE 8.312 against 9.158,
asymmetric score 3.179 against 3.649) and FD004 by 3.89% (MAE 8.369 against 8.807,
asymmetric score 8.682 against 9.943); the stratified interval is -5.53% with bounds
[-0.881, -0.589]. The gain is positive for every tau in the sensitivity set.

Two disclosures. First, the unguarded result is reported alongside: FD004 -2.92% with an
asymmetric score of 37.76, which is the failure the guard is designed to remove. Second,
the four-metric clause holds in the pooled comparison but not for every seed separately:
on seed 202 and FD004 the asymmetric score rises from 8.63 to 9.71 while the RMSE falls
by 3.87% and the MAE by 6.1%. We do not claim that the guard makes the asymmetric score
monotone across seeds; we claim that it removes the catastrophic tail and that the pooled
comparison is then no worse than the reference on both metrics.
