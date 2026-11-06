# PAPER — METHODS SECTIONS (draft, 2026-11-05 19:45)

## III. MEASUREMENT MODEL

### A. Measurand and reference

The measurand is the remaining useful life RUL_t of one engine at observation time t,
expressed in cycles and bounded by the reporting cap R_max = 125 cycles. The reference
value is produced by the C-MAPSS simulator: RUL_t = min(R_max, T - t), where T is the
engine's realised total life. Because T is the simulator's own realisation and not a
metrological standard, the resulting errors are estimator errors, not instrument
uncertainties, and they are reported as such.

### B. Observation and conditioning

At time t the instrument observes a window X_t in R^{50 x 14} of the last 50 cycles of 14
sensor channels, together with the three operating settings o_t and the cycle index t.
Operating conditions are handled by fitting, on the training engines of the fold only, a
six-cluster KMeans model on the operating settings and a per-cluster z-score for each
sensor channel. For each engine and each cluster we additionally form an early-life
reference r_m, the shrunk mean of the first 30 cycles observed in that cluster, which
serves as the engine's own baseline at that condition. All quantities used at time t are
functions of the prefix 1..t only.

### C. Backbone

The predictor is a 42.3k-parameter channel-independent transformer over the 50-cycle
window with sinusoidal positional encoding of the cycle coordinate, a shared per-sensor
embedding, a temporal convolution, two attention layers, mean pooling over time, and a
linear regression head. It is deliberately small: a 141k-parameter variant did not
improve the outer-validation error, so capacity is not the limiting factor.

### D. Individual lifetime prior

A histogram gradient-boosting ensemble estimates the engine's total life from the observed
prefix. Features are the operating settings, the cycle index, the per-condition early-life
reference, the recent level and spread, the recent-versus-early difference, the 40-cycle
slope and prefix-wide summaries. The model is fitted with a nested four-fold split inside
the training engines and applied through a staircase construction: a window ending at
cycle t consumes only the most recent cutoff c <= t, so no future information enters. The
prior outputs a point estimate mu_T; its dispersion across seeds is at floating-point
level, so we do not use it as an uncertainty.

## IV. THREE ROLES OF THE LIFETIME ESTIMATE

The same scalar mu_T can enter the measurement in three places.

**R1, additive prior.** The pooled context h is augmented by a zero-initialised linear map
of (mu_T, sigma_T): h' = h + W[mu_T, sigma_T], with W initialised to zero so that the arm
starts exactly at the backbone and the difference is attributable to the injected
information rather than to re-initialisation.

**R2, time base.** The cycle coordinate that feeds the positional encoding is replaced by
the dimensionless phase c / mu_T, so that the encoder is asked to represent the degradation
stage in units of the individual life rather than in cycles.

**R3, target scale.** The network predicts a life fraction phi and the reported RUL is
phi * mu_T, so that the individual scale converts a scale-free prediction into cycles.

Each role is implemented behind the same backbone, the same optimiser, the same sample
order and the same random-number stream, so that a difference between arms is attributable
to the role and not to the training recipe.

## V. IDENTIFIABILITY OF ADAPTIVE CORRECTION

If the individual reading were more accurate than a population reference for some
observations and less accurate for others, and if that difference were predictable from an
observable statistic z, then a per-observation weight w(z) would approach the per-window
oracle. We test this directly. We compute, for every validation window, the absolute errors
of the individual reading and of the population reference, and we correlate their
difference with (i) window-level degradation evidence (deviation from the early-life
reference, jitter, trend-to-noise ratio), (ii) the observation age, (iii) engine-level
averages of the same statistics, and (iv) the disagreement between the arms. We then
evaluate quantile gates and age gates built from each statistic.

A second, metrologically motivated observable is the sensitivity coefficient of the
reading with respect to the observation: we perturb the input window with Gaussian noise of
standard deviation 0.05, 0.10 and 0.20 in normalised units, repeat the forward pass five
times and take the standard deviation of the prediction. This is the GUM-style sensitivity
of the estimate to the input, and it can be reported per observation.

## VI. MEASUREMENT CONDITIONS

Three conditions are imposed on the evaluation side only, identically for every arm, with
the model weights unchanged: MC1 removes 20% of the sensor entries at random (per window,
cycle and channel) and forward-fills them along the window; MC2 adds one training-fold
standard deviation to the three channels whose early-life mean has the largest absolute
Spearman correlation with total life in the training fold; MC3 stratifies the validation
windows by the support of their terminal operating cluster in the training fold. These are
simulated measurement conditions, not bench validations.

## VII. RESULTS

<TBD — tables from RESULTS.csv / ABLATIONS.csv / MEASUREMENT_CONDITIONS.csv.>
