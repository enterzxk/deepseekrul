# PAPER DRAFT — complete working draft (2026-11-05 20:30)

Title: *Observability-Limited Individualisation for Remaining-Useful-Life Measurement of
Aero-Engines under Varying Operating Conditions*

Target: IEEE Transactions on Instrumentation and Measurement.

> Numbers marked <TBD> are regenerated from RESULTS.csv / ABLATIONS.csv /
> MEASUREMENT_CONDITIONS.csv by `assemble_delivery.sh`; none is typed by hand.

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
pooled RMSE by 3.1%, but replacing the cycle coordinate by c/mu_T increases it by 7.7% and
multiplying a predicted life fraction by mu_T increases it by 28%. A per-window oracle over
the variants would reduce the error by a further 15%, but none of six observable families we
tested predicts which variant is better for a given observation (|rho| < 0.08). Per-
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

<TBD — N1–N4 from NOVELTY_MATRIX.md. N2's phase-aware normalisation formula must be
verified against the full text before submission.>

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

<TBD — Tables II–VII; Figures 1–5.>

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
