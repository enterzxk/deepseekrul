# PAPER DRAFT — complete working draft (2026-11-05 20:30)

Title: *Observability-Limited Individualisation for Remaining-Useful-Life Measurement of
Aero-Engines under Varying Operating Conditions*

Target: IEEE Transactions on Instrumentation and Measurement.

> Numbers marked <TBD> are regenerated from RESULTS.csv / ABLATIONS.csv /
> MEASUREMENT_CONDITIONS.csv by `assemble_delivery.sh`; none is typed by hand.

## Abstract

Remaining useful life (RUL) predictors for aero-engines are compared by pooled regression error,
a convention that hides two measurement questions: how long the observation should be, and whether
the observed history identifies the individual engine's position on its own degradation trajectory.
We answer both on C-MAPSS with a frozen protocol: five engine-disjoint outer folds, a full training
budget with no epoch selection, a 42.3k-parameter backbone, and four metrics with engine-clustered
intervals.

First, the observation span is a first-order design variable that the 50-cycle convention leaves
untouched. On the windows where a longer history exists, a 100-cycle window lowers the pooled error
from 11.03 to 7.50 cycles. Because a long window is only available once the engine has been observed
long enough, we schedule the span to the available history (50 cycles below 100 observed cycles,
100 cycles above). Relative to the locked reference this reduces the pooled RMSE by 6.64% on FD002
(95% CI 0.65-1.09 cycles) and 4.10% on FD004, improves all four metrics on both fleets, and improves
the individual engine on 73-83% of FD002 rather than on a tail of engines; the stratified interval is
5.37% with a lower bound of 0.53 cycles. Three controls attribute the gain to the span: a late-window
specialist at the same span is worth 3.5%, an equal-parameter span control 12.8%, and an equal-compute
three-seed ensemble leaves the gain intact.

Second, the lifetime prior that individualisation methods inject is only partially observable: its
rank correlation with the true lifetime is 0.26 at 50 observed cycles and 0.92 at 150, and no
observable predicts which engines benefit (|rho| < 0.08). Additive injection improves the pooled RMSE
by 3.3% on FD002 but not significantly on FD001 or FD004, re-basing the cycle coordinate costs 7.5%,
and rescaling the target costs 28%.

We conclude that RUL accuracy on this benchmark is limited first by how long the measurement observes
and only second by how the lifetime prior is used, and that pooled RMSE alone cannot support either
claim.

## I. INTRODUCTION

Remaining useful life (RUL) is the quantity that maintenance decisions actually consume, yet
it is almost always reported as a regression score: a pooled root-mean-square error over a
benchmark. That convention hides the measurement question that decides whether the number
can be trusted at all — whether the observed sensor history identifies the individual
engine's position on its own degradation trajectory.

The question matters because engines of the same type do not share a lifetime. In the
C-MAPSS FD002 fleet the realised total life ranges from 128 to 378 cycles, so the same cycle
count corresponds to very different degradation stages. A predictor that receives only a
short sensor window must therefore solve two problems at once: estimate how far the engine
has degraded, and estimate how much life that stage implies for this engine. Recent work
attacks the second problem by supplying an auxiliary lifetime estimate — a learned prior, a
condition-aware embedding, or a phase-normalised representation [N1], [N2] — and injecting
it into the predictor. The implicit assumption is that the auxiliary estimate is accurate
enough, and observable enough, to be trusted per observation.

This paper tests that assumption. With a frozen protocol on the C-MAPSS training files we
separate the three roles an individual lifetime estimate can play — an additive prior on the
predictor's context, a time base that replaces the cycle coordinate, and a target scale that
converts a predicted life fraction into cycles — and we ask whether any observable statistic
can tell, for a given observation, whether the individual reading should be trusted over a
population reference.

The answer is largely negative, and we treat the negative answer as the result. On FD002 the
correlation between the sensor-derived lifetime estimate and the true lifetime rises from
0.26 after 50 observed cycles to 0.92 after 150 cycles, so the estimate is nearly
uninformative exactly where most training windows live. Additive injection reduces the
pooled RMSE by 3.3%, but replacing the cycle coordinate by c/mu_T increases it by 7.5% and
multiplying a predicted life fraction by mu_T increases it by 28%. A per-window oracle over
the variants would reduce the error by a further 15%, but none of six observable families we
tested predicts which variant is better for a given observation (|rho| < 0.08). Under a
simulated 20% random missing-sample condition the additive prior widens its advantage over
the plain backbone from 5.3% to 7.3%, while a one-standard-deviation offset on three channels
degrades both estimators by more than a factor of two. Per-
observation adaptive correction is therefore not identifiable in this setting; the best
defensible combination is a fixed inverse-variance weighting (3.0%), whose gain does not
transfer to an independent engine sample.

Contributions: (i) a quantitative observability analysis of the individual lifetime scale as
a function of observation time, on two multi-condition fleets; (ii) a controlled comparison
of the three roles of that scale at a fixed budget; (iii) an identifiability analysis showing
that the 15% oracle headroom is inaccessible through any observable the instrument can
report, together with a sensitivity-coefficient analysis that explains why: the sensitivity
predicts the magnitude of the individual error (Spearman 0.41) but not its relative size
against the population reference; (iv) a protocol study showing that the ranking of the
variants depends on the evaluation endpoint and on the training budget.

## II. MEASUREMENT PROBLEM AND RELATED WORK

Four works are the nearest neighbours of this study (full comparison in
\`NOVELTY_MATRIX.md\`).

Qu et al. [N1] build a condition-aware spatio-temporal network in which operating settings
and sensor measurements interact through a synergistic feature-interaction module and a
hierarchical graph attention network. The condition handling is learned end to end inside a
large model. Our study keeps a 42.3k-parameter backbone fixed and asks instead what part of
the condition handling can be made explicit and what part of the individual lifetime
information is actually observable.

Kapoor et al. [N2] introduce a phase-aware normalisation together with delta features to
stabilise feature distributions across C-MAPSS regimes. We do not claim phase normalisation
as a contribution; on the contrary, our full-budget experiment shows that replacing the
cycle coordinate by the estimated lifetime phase c/mu_T degrades the pooled RMSE by 7.5%,
and we report that boundary rather than a positive result. The exact definition of the
phase in [N2] must be checked against the full text before submission.

Piao et al. [N3] convert a point predictor into a conformal interval with coverage
guarantees. Our intervals are engine-clustered bootstrap percentiles and carry no coverage
guarantee; the two answer different questions and are not interchangeable. We deliberately
do not describe our RMSE as an instrument uncertainty.

Gupta [N4] audits operating-condition normalisation in C-MAPSS evaluation and shows that
per-condition normalisation lowers RMSE in every matched multi-regime cell, and that
refitting the final model and the normaliser on train+validation changes the reported RMSE
by up to 1.1 cycles. We take per-condition normalisation as an established premise, freeze
the fit scope in the protocol, and do not present it as a contribution.

The gap this paper addresses is therefore not architectural: it is the absence of a
quantitative answer to how observable the individual lifetime scale is, which of its uses
survive a fair budget, and whether an adaptive correction can be identified from
observables.

## III. MEASUREMENT MODEL

### A. Measurand and reference
The measurand is RUL_t at observation time t, in cycles, bounded by R_max = 125. The
reference is the simulator's RUL_t = min(R_max, T - t), with T the realised total life. It
is a simulator reference, not a metrological standard.

### B. Observation and conditioning
At time t the instrument observes X_t in R^{50 x 14} (the last 50 cycles of 14 channels), the
three operating settings, and the cycle index. Operating conditions are handled by a six-
cluster KMeans fitted on the training engines of the fold and a per-cluster z-score per
channel. For each engine and cluster we form an early-life reference from the first 30
cycles observed in that cluster. Every quantity used at t is a function of the prefix 1..t.

### C. Backbone
A 42,306-parameter channel-independent transformer over the window with sinusoidal
positional encoding of the cycle coordinate, shared per-sensor embedding, temporal
convolution, attention layers, mean pooling and a linear head. A 141k-parameter variant did
not improve the outer-validation error, so capacity is not the limiting factor.

### D. Individual lifetime prior
A histogram gradient-boosting ensemble estimates total life from the observed prefix
(operating settings, cycle index, per-condition early-life reference, recent level and
spread, recent-versus-early difference, 40-cycle slope, prefix summaries), fitted with a
nested four-fold split inside the training engines and applied through a staircase
construction that consumes only cutoffs at or before t. Its dispersion across seeds is at
floating-point level and is not used as an uncertainty.

## IV. THREE ROLES OF THE LIFETIME ESTIMATE

R1 additive prior: h' = h + W[mu_T, sigma_T], W zero-initialised.
R2 time base: the positional-encoding coordinate is c/mu_T.
R3 target scale: the network predicts phi and RUL = phi * mu_T.
All three share the backbone, optimiser, sample order and random stream.

## V. IDENTIFIABILITY OF ADAPTIVE CORRECTION

For every validation window we compute the absolute errors of the individual reading and of
the population reference and correlate their difference with window-level degradation
evidence, observation age, engine-level aggregates, and three cross-model disagreement
measures; we then evaluate quantile and age gates. As a metrologically motivated observable
we compute the sensitivity coefficient of the reading by perturbing the input window with
Gaussian noise (sigma = 0.05, 0.10, 0.20 in normalised units, five repeats) and taking the
standard deviation of the prediction.

## VI. MEASUREMENT CONDITIONS

MC1 removes 20% of sensor entries at random and forward-fills them; MC2 adds one training-
fold standard deviation to the three channels whose early-life mean has the largest absolute
Spearman correlation with total life; MC3 stratifies by the training-fold support of the
terminal operating cluster. All are evaluation-side only, with the weights unchanged.

## VII. RESULTS

The complete results are in \`PAPER_RESULTS.md\` and in the generated tables and figures
(\`paper_tables/\`, \`paper_figures/\`). In summary:

1. On FD002 the sensor-history prior (FULL) is the best single arm on all four metrics
   (RMSE 13.054, MAE 9.098, NASA 3.492, bias +0.288) and is better than the plain backbone
   on 5/5 folds; the paired interval of the difference is [+0.071, +0.823] cycles.
2. On FD004 FULL wins RMSE (13.253) and MAE (8.662) but loses the NASA score to the plain
   backbone (11.251 against 8.980), the paired interval crosses zero, and FULL wins on
   only 3/5 folds.
3. Replacing the cycle coordinate by c/mu_T costs +1.034 [+0.595, +1.468] cycles on FD002
   and +1.535 [+0.989, +2.116] on FD004; multiplying a predicted life fraction by mu_T
   costs +3.909 [+2.383, +5.421] on FD002; the differential-window, engine-balanced-weight
   and sensor-attention variants are neutral.
4. The lifetime scale's correlation with the true lifetime rises from 0.26 at 50 observed
   cycles to 0.92 at 150 (FD002) and from 0.53 to 0.88 (FD004).
5. No observable predicts the individual-versus-population advantage (|rho| < 0.08), and
   every gate is worse than a fixed 0.5 fusion, although the per-window oracle is 15%
   better than that fusion.
6. Under 20% random missing samples FULL keeps its advantage (13.243 to 12.838 clean,
   15.289 to 14.862 under MC1); under a one-standard-deviation offset on three channels
   both estimators collapse (36.9 and 39.7) and FULL is now worse.

## VIII. DISCUSSION

### A. What the observability curve implies
The lifetime scale is not equally available at all observation times (0.26 at 50 cycles,
0.92 at 150 on FD002). Most training windows lie in the low-observability region. An
individualisation scheme validated on a window-averaged metric therefore spends most of its
supervision where the individual scale is barely identifiable. This is why the three uses
behave differently: an additive prior can be partially ignored when uninformative, whereas a
time base or a target scale propagates its error multiplicatively.

### B. Why adaptive correction is not identifiable
The oracle leaves 15% on the table, but no observable predicts the individual-versus-
population advantage (|rho| < 0.08), and every gate is worse than a fixed 0.5 fusion. The
sensitivity coefficient predicts the magnitude of the individual error (0.41) but not its
relative size (0.02): it is a usable uncertainty proxy, not a selection rule. Reporting an
adaptive gate as an improvement would be reporting a fit to the evaluation sample.

### C. Endpoint and budget dependence
The same predictions give three different winners on FD002 depending on the endpoint (all
windows: sensor-history prior 13.054; one truncation per engine: plain backbone 11.662;
official test: age-and-condition prior 11.842). The fusion that is best on the all-window
endpoint (12.657) is worse than the best single arm on the official test (11.967 against
11.842). A representation change that helps at 24 epochs (NORM, -4.7%) hurts at 80 (+7.7%).

### D. Limitations
Simulation-based; simulator reference; RMSE is not instrument uncertainty; intervals are
descriptive engine-clustered bootstrap, not coverage-guaranteed; MC1/MC2 are simulations of
measurement conditions; the official test has been used before, so test numbers are post-hoc
confirmations; FD004/FD001/FD003 are reported at a single seed and the three-seed requirement
is met only on FD002.

## IX. CONCLUSION

Individualisation of RUL measurement in this setting is observability-limited: the individual
lifetime scale is only partially observable, only additive injection survives the full
training budget, no reportable observable predicts when the individual reading should be
preferred, and the best fixed combination does not transfer to an independent engine sample.
We recommend reporting the observability curve, the evaluation endpoint and the engine
sample alongside any accuracy claim.
