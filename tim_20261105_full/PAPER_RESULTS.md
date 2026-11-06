# PAPER — RESULTS SECTION (draft with final seed-101 numbers, 2026-11-05 21:45)

All numbers from `RESULTS.csv`, `ABLATIONS.csv`, `MEASUREMENT_CONDITIONS.csv`,
`observability.json`, `PER_ENGINE.csv`, `paper_tables/per_fold.csv`.
Protocol: full budget (80 epochs, all training windows, fixed final epoch), five
engine-disjoint outer folds, RUL capped at 125, engine-equal-weight pooled metrics,
engine-clustered paired bootstrap (5000).

## A0. Cross-subset summary (seed 101, M0 vs FULL)

| Subset | Engines | RMSE M0 | RMSE FULL | RMSE rel | MAE rel | NASA rel | Bias M0 | Bias FULL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 100 | 10.274 | 10.235 | -0.38% | -0.70% | -4.34% | +0.801 | -0.329 |
| FD002 | 260 | 13.505 | 13.054 | -3.34% | -2.48% | -12.98% | +1.208 | +0.288 |
| FD003 | 100 | 10.863 | 10.158 | -6.49% | -5.63% | -13.58% | -1.724 | -1.125 |
| FD004 | 249 | 13.398 | 13.253 | -1.08% | -1.68% | **+25.29%** | -0.171 | -1.038 |

FULL improves RMSE on all four subsets, but the NASA score improves on three and
degrades by 25% on FD004. The single-condition subsets (FD001, FD003) show the guardrail
is satisfied with margin. Source: \`paper_tables/cross_subset.csv\`.

## A. Main comparison (seed 101)

**FD002** (260 engines, 41,019 windows). Reference arm (lowest outer-validation RMSE):
FULL, 13.054.

| Arm | RMSE | MAE | NASA (per-window mean) | Bias | Delta vs reference [95% CI] |
|---|---:|---:|---:|---:|---|
| M0 | 13.505 | 9.329 | 4.012 | +1.208 | +0.451 [+0.071, +0.823] |
| OPS | 13.376 | 9.291 | 3.904 | +0.776 | +0.322 [+0.060, +0.583] |
| **FULL** | **13.054** | **9.098** | **3.492** | **+0.288** | — |

FULL is better than M0 on 5/5 folds (per-fold deltas -0.03 to -0.98).

**FD004** (249 engines, 49,048 windows). Reference arm: FULL, 13.253.

| Arm | RMSE | MAE | NASA | Bias | Delta vs reference [95% CI] |
|---|---:|---:|---:|---:|---|
| M0 | 13.398 | 8.810 | **8.980** | -0.171 | +0.145 [-0.240, +0.518] |
| OPS | 13.676 | 8.721 | 30.644 | -0.108 | +0.422 [+0.123, +0.756] |
| **FULL** | **13.253** | **8.662** | 11.251 | -1.038 | — |

FULL is better than M0 on 3/5 folds only, the pooled interval crosses zero, and FULL's
NASA score is worse than M0's. The additive prior is therefore not uniformly better on
FD004.

## B. Ablations (each vs M0 on the folds where both exist)

| Variant | Subset | Folds | Delta RMSE [95% CI] | Verdict |
|---|---|---:|---|---|
| NORM (cycle coordinate replaced by c/mu_T) | FD002 | 4 | +1.034 [+0.595, +1.468] | significantly worse |
| NORM | FD004 | 4 | +1.535 [+0.989, +2.116] | significantly worse |
| PHASE (RUL = phi * mu_T) | FD002 | 2 | +3.909 [+2.383, +5.421] | significantly worse |
| DIFF (window minus early-life reference) | FD002 | 2 | -0.038 [-0.534, +0.472] | neutral |
| EQBAL (engine-balanced sample weights) | FD002 | 1 | -0.092 [-0.873, +0.704] | neutral |
| GRAPH (sensor-wise attention, +40% params) | FD002 | 1 | -0.019 | neutral |

## C. Observability of the lifetime scale

| Subset | 50 | 100 | 150 | 190 |
|---|---:|---:|---:|---:|
| FD002 corr(mu_T, T) | 0.26 | 0.75 | 0.92 | 0.94 |
| FD002 MAE (cycles) | 35.5 | 22.4 | 11.1 | 9.9 |
| FD004 corr | 0.53 | 0.74 | 0.88 | 0.87 |
| FD004 MAE (cycles) | 49.1 | 36.2 | 21.1 | 19.9 |

The curve is computed on the two multi-condition fleets, where the operating-condition
normalisation is the object of study. For the single-condition fleets the same prior
reaches a rank correlation of 0.65 (FD001) and 0.68 (FD003) at a 100-cycle cutoff and
0.92 and 0.90 at 150 cycles, with MAE 25.9/47.0 and 13.4/25.6 cycles respectively.

## D. Identifiability of adaptive correction (FD002, 4 folds)

Six observable families are essentially uncorrelated with the individual-versus-population
advantage (|rho| < 0.08). Quantile and age gates give 13.03-13.62 against a fixed 0.5
fusion at 12.89; the per-window oracle is 10.97. The sensitivity coefficient correlates
0.37-0.39 with the absolute error but 0.02 with the relative advantage.

## E. Measurement conditions (FD002, folds 0 and 2, weights unchanged)

| Condition | M0 | FULL |
|---|---:|---:|
| clean | 13.372 | 13.221 |
| MC1 (20% missing, forward fill) | 15.289 | 14.862 |
| MC2 (1 sigma offset on three channels) | 36.937 | 39.692 |

MC3 (terminal-condition support tertiles) is flat (13.0-13.6 for all arms).

## F. Endpoint and budget dependence

On FD002 the three endpoints select three different winners: all windows -> FULL 13.054;
one truncation per engine -> M0 11.662; official test -> OPS 11.842. A leave-one-fold-out
inverse-variance fusion reaches 12.657 (-3.0% vs FULL) on all windows but 11.967 on the
official test, worse than OPS. NORM is better than M0 at 24 epochs (-4.7%, three seeds)
and worse at 80 epochs (+7.5%).

## G. Per-engine win rates

The aggregate improvement is not a broad improvement. FULL is better than M0 on 144 of
260 FD002 engines (55%) and on 133 of 249 FD004 engines (53%); the mean per-engine RMSE
falls from 12.74 to 12.30 (FD002) and from 11.84 to 11.51 (FD004). The pooled gain comes
from a few large improvements (up to -8.5 cycles) outweighing a few large regressions
(up to +8.1 cycles). Reporting only the pooled RMSE hides the fact that roughly half of
the engines get worse.

## H. Matched-fold multi-seed check

On the four folds where seeds 101 and 202 are both complete, the sensor-history prior
improves the pooled RMSE by 3.20% (seed 101) and 2.27% (seed 202); the two-seed mean is
2.73%, below the 3% threshold. The larger single-seed figure is partly due to a favourable
initialisation on fold 1 (12.04 against 13.28 cycles). Any claim above 3% must therefore
rest on the three-seed mean, not on a single seed.

## I. Metric behaviour

NASA score on FD004 is dominated by 1-5 engines (OPS: 90% of the total from five engines,
one engine contributing 5256), so the NASA ranking is unstable at this fleet size. The
95% interval half-width of the RMSE difference is 0.376 cycles on FD002, i.e. about 2.9%
relative, so a 3% effect sits at the edge of detectability.

## J. Observation span as a first-order design variable

The protocol fixes the observation window at 50 cycles, a convention inherited from the
C-MAPSS literature. We tested whether that convention is leaving accuracy on the table.

**Effect of the span.** On the windows where a longer history is available, replacing the
50-cycle window by a 100-cycle window reduces the pooled error from 11.03 to 7.50 cycles
(FD002, windows with at least 100 observed cycles); a 150-cycle window reaches 4.97 on the
windows where it applies. The comparison is made on identical windows, so it is not an
artefact of a changing evaluation set.

**Schedule.** Because a longer window is only available once the engine has been observed
long enough, we use a schedule: windows ending before 100 cycles use the 50-cycle model,
windows ending at or after 100 cycles use the 100-cycle model. Both models are trained
with the frozen budget (80 epochs, all available windows, fixed final epoch, no selection).

**Controls.** Three controls separate the span from competing explanations. (i) A late-window
specialist trained at the same 50-cycle window on the same late windows reaches 10.65
against 11.03 for the generalist and 7.50 for the 100-cycle model, so specialisation alone
is worth 3.5% and the span is worth the rest. (ii) A span control with identical parameter
count (50 positions sampled every second cycle over 100 cycles) reaches 9.62, so both the
span and the temporal resolution contribute. (iii) A three-seed ensemble of the 50-cycle
reference, which has the same model count as the schedule, reaches 12.53 on the fold where
the schedule reaches 11.80, so the gain is not a model-count effect.

**Boundaries.** The gain is concentrated in long-lived engines: -10% to -11% on engines
above the first lifetime quartile against -2% below it, because short-lived engines reach
100 observed cycles only near the end of their life. The pooled error is dominated by early
windows, where no longer history exists, so the whole-window gain is necessarily smaller
than the late-window gain.

**A leakage trap we avoided.** Enforcing a monotone decreasing RUL along the trajectory by
isotonic regression lowers the pooled error by 12.5%, but the fit uses windows that occur
after the one being predicted. The causal version, which fits only on the prefix available
at each prediction time, is worth 2.5% on FD002 and is slightly negative on FD004. We report
the causal result only.

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
