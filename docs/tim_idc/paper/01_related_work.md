# Related Work (draft v0.1 — writing-gate compliant, no results claimed)

> Citation policy: only references verified to exist (checked 2026-11-03).
> [VERIFY-BIB] tags mark entries needing exact bibliographic extraction
> before submission. Method positioning paragraphs are branch-agnostic.

## 2.1 Deep sequence models for RUL prediction

Deep architectures dominate recent C-MAPSS literature. Convolutional and
recurrent backbones established the sliding-window regression paradigm
[BABU2016CNN, LI2018DCNN, ZHENG2017LSTM — VERIFY-BIB]. Transformer encoders
were adopted for long-horizon dependency modeling [MO2021TransformerEncoderGCU
— VERIFY-BIB], and subsequent work refined the attention block itself:
LECformer adds local-context enhancement and channel weighting to the
self-attention of vanilla Transformers [LECformer — VERIFY-BIB], and CET-MSA
(convolutionally-enhanced temporal multi-head self-attention) preconditions
key/value streams with neighborhood-aware convolutions plus pyramidal
key/value reduction, reporting Score/RMSE gains on FD001–FD004 with
throughput/latency analysis [CETMSA2026 — verified: Springer J Intell Manuf,
DOI 10.1007/s10845-026-02895-3, official code
github.com/suosimiaoqianli/Noise-Robust-RUL-CET-MSA].

These works condition the attention on *what to attend to* but keep the time
coordinate itself generic: positions are encoded by absolute cycle index
(possibly normalized by a constant horizon), which is identical across
engines at the same cycle. Our proposal changes the *coordinate system*:
attention and sensor interactions are conditioned on an individualized
degradation stage, estimated per engine. This is orthogonal to block-level
architectural improvements and composable with them.

## 2.2 Handling operating-condition heterogeneity

FD002/FD004 mix six operating conditions and two fault modes; per-condition
normalization and clustering are standard mitigations [ OperatingCondition-
Scaler family — VERIFY-BIB: e.g. conditionwise z-score used by DCNN/TensorFlow
baselines], and domain-adaptation pipelines target cross-condition shift
[MSMT-FDA, RESS 2026 — existence verified, no official code; VERIFY-BIB].
We adopt per-condition scaling for the backbone and, critically, treat
condition heterogeneity as a *driver of lifetime heterogeneity*: engines in
different regimes have different total lives, which is exactly what the
lifetime prior must capture causally.

## 2.3 Similarity-based and lifetime-aware prognostics

Before deep regression, RUL was estimated by *similarity to fleet reference
trajectories*, where a library of run-to-failure histories provides the time
axis [WANG2008Similarity — VERIFY-BIB]. These methods implicitly exploit the
quantity our clock makes explicit: an engine's degradation stage is
meaningful only relative to its own total life. Modern deep models dropped
this relativity in favor of absolute age. Two recent lines partially recover
it: (i) health-indicator construction with phase/stage labels
[phase-based HI — VERIFY-BIB], and (ii) remaining-useful-life prediction via
degradation-point estimation [VERIFY-BIB]. Our work differs in three
respects: the per-engine lifetime scale is produced by a *strictly causal,
nested-cross-fitted* estimator (no future information, audited by leakage
unit tests); it enters the network as a *conditioning signal* (attention
bias, sensor mixing, gated residual), not as a feature concatenation alone;
and its *uncertainty* controls how much the model trusts the individualized
coordinate.

## 2.4 Uncertainty in prognostics

Prognostic uncertainty quantification spans Bayesian recurrent networks,
Monte-Carlo dropout, ensembles, and conformal prediction
[CONFORMAL-PDM — verified repo Al-Moccardi/Conformal-Predictive-Maintenance;
VERIFY-BIB for journal version]. These methods quantify uncertainty *of the
RUL output*. We instead use the uncertainty of an *upstream state estimator*
(lifetime prior) as a control signal: high prior uncertainty down-weights the
individualized pathway and reverts to the strong backbone. To our knowledge
this "uncertainty as a mixing gate between a data-driven and an
identity-driven time coordinate" is not present in prior RUL work
[VERIFY-BIB via targeted search at writing time].

## 2.5 Hyperparameter and protocol fragility

BootBoGs-TS recently argued that RUL benchmark rankings are sensitive to
hyperparameter selection and proposed bootstrap-based unit-level resampling
for tuning [BOOTBOGS — verified: Springer, DOI 10.1007/s44196-026-01344-6,
official code github.com/Gyouness/BootBoGs-TS]. We share the concern about
protocol fragility and go further on the evaluation side: engine-level
clustered bootstrap for all deltas, pre-registered gates, and shuffled /
constant / oracle controls to attribute gains to the mechanism rather than
to capacity.
