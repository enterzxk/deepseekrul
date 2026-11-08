# PAPER — DISCUSSION AND CONCLUSION (draft, 2026-11-05 20:25)

## VIII. DISCUSSION

### A. What the observability curve implies for individualisation

The lifetime scale that individualisation methods inject is not equally available at all
observation times. On FD002 the correlation between the sensor-derived estimate and the
true lifetime is 0.26 at 50 observed cycles and 0.92 at 150; on FD004 it is 0.53 and 0.88.
Most training windows in a C-MAPSS trajectory lie in the low-observability region, and the
same is true of the early part of any real run-to-failure record. An individualisation
scheme that is validated on a window-averaged metric therefore spends most of its
supervision on observations for which the individual scale is barely identifiable. This
is, in our view, the main reason why the three uses of the same scale behave so
differently: only the additive prior can be partially ignored by the network when it is
uninformative, whereas a time base or a target scale propagates the estimate's error
multiplicatively and cannot be switched off.

### B. Why adaptive correction is not identifiable here

The oracle over the three variants leaves 15-16% of the pooled error on the table. That
headroom is real, but it is not addressable by an observable rule. Six families of
observables — window-level degradation evidence, observation age, engine-level aggregates,
and three cross-model disagreement measures — are essentially uncorrelated with the
individual-versus-population advantage (|rho| < 0.08), and every gate we built from them
was worse than a fixed 0.5 fusion. The sensitivity coefficient behaves differently: it
predicts the magnitude of the individual error (Spearman 0.41) but not its relative size
against the population reference (0.02). In metrological terms, the sensitivity coefficient
is a usable uncertainty proxy for the individual reading, but it does not tell the
measurement system which of two readings to trust.

We regard this as the central measurement result of the study: in this task, the
information needed to decide *when* to trust the individual reading is not contained in the
observables that the instrument can report. Reporting an adaptive gate as an improvement
would therefore be reporting a fit to the evaluation sample.

### C. Endpoint and budget dependence

The same predictions give three different winners on FD002 depending on the endpoint: all
windows selects the sensor-history prior (13.054), one truncation per engine selects the
plain backbone (11.662), and the official test engines select the age-and-condition prior
(11.842). The fixed fusion that is best on the all-window endpoint (12.657, 3.0% below the
reference) is worse than the best single arm on the official test (11.967 against 11.842).
A representation change that helps at 24 training epochs (NORM, -4.7% on FD002) hurts at
80 epochs (+7.7%). Both observations are, in our view, reasons to report the endpoint, the
engine sample, and the training budget together with any RUL accuracy claim, and to prefer
engine-clustered paired intervals over pooled point estimates.

### D. Limitations

The study is simulation-based. C-MAPSS is a thermodynamic simulation, and the reference RUL
is the simulator's own realised lifetime, not a metrological standard; the reported RMSE
values are estimator performance metrics, not instrument uncertainties, and the intervals
are descriptive engine-clustered bootstrap intervals, not coverage-guaranteed uncertainty
statements. The missing-data and offset conditions are simulations of measurement
conditions, not bench validations. The official test engines have been used by this project
before, so the test numbers are post-hoc confirmations, not a blind validation. Finally,
the FD004, FD001 and FD003 matrices are reported at a single seed, and the three-seed
requirement is met only on FD002; this is stated wherever those numbers appear.

## IX. CONCLUSION

We examined whether the individual lifetime scale that individualisation methods inject into
an RUL predictor is observable enough to be used adaptively. On two multi-condition
C-MAPSS fleets, under a frozen engine-disjoint protocol, the scale is only partially
observable and its observability grows with the observation time; of the three ways it can
enter the measurement, only additive injection survives at the full training budget, and
none of the observables that the instrument can report predicts when the individual reading
should be preferred to a population reference. The best defensible combination is a fixed
inverse-variance weighting, and even that gain does not transfer to an independent engine
sample. We conclude that individualisation of RUL measurement in this setting is
observability-limited, and we recommend reporting the observability curve, the endpoint and
the engine sample alongside any accuracy claim.
