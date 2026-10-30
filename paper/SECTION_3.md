# 3 Study design

Draft. Numbers here are the implementation of `paper/E6_STATS_AUDIT.md`; no result is
reported in this section.

## 3.1 Scope and data

We use the official C-MAPSS **training trajectories only**, on two subsets: FD001
(100 engines, one operating condition, one fault mode) and FD002 (260 engines, six
operating conditions, one fault mode). FD003 and FD004 are sealed by the protocol and
are not read, counted or probed at any point; no claim in this paper extends to them,
and the official test files are not used for model selection, calibration or any
reported metric (Supplement, Data use).

Within a subset we use an 80/20 **engine-level** split with a fixed split seed
(`split_seed = 7`), so that the same engines are held out across methods, seeds and
arms. Windows are 50 cycles, stride 1 for training, one matched cutoff per held-out
engine for evaluation. Because the split is engine-level, no window of a held-out
engine appears in training; because the split seed is fixed, every contrast in this
paper is paired at the engine level.

The label is the standard capped remaining life (cap 125 cycles). The cap is held
constant across every experiment; §4.5 quantifies what happens when it is not.

## 3.2 Units, metrics and what is reported together

The evaluation unit is the **engine**, not the window. Each held-out engine
contributes one prediction at a matched cutoff, so an interval over engines is an
interval over independent units. For every arm we report, together:

| quantity | definition |
|---|---|
| RMSE / MAE | over the held-out engines |
| NASA score | asymmetric score of Saxena et al., **averaged over engines** (never summed; the two differ by the engine count) |
| late | fraction of engines with positive error |
| severe | fraction of engines with error > 10 cycles — the frozen definition, not a late-and-short-RUL proxy |

The legacy proxy (`error > 0 and RUL ≤ 30`) understates the severe rate by roughly
five times on our data; §4.5 reports both so that the discrepancy is on the record.

## 3.3 What varies, and what is treated as noise

Three factors are varied deliberately: the method (or arm), the held-out engine set,
and the random seed. A variance decomposition over a 3×3 design (method × engine set)
and a 3×3×3 design (adding seed) attributes RMSE variation to each factor; the result
(§4.1) is that the engine set dominates the method, which is the empirical
justification for treating engines as the resampling unit everywhere else.

Seeds are treated as a random factor with three levels. We do not select the best
seed, and every reported mean is over all three.

## 3.4 Statistics

**Effect size.** Δ = RMSE(first) − RMSE(second); negative means the first arm is
better. One convention, used in every table.

**Intervals.** Engine-clustered bootstrap: engines are resampled with replacement
within a seed, the statistic is recomputed, and the replicates are averaged over
seeds. We use 100,000 resamples and three fixed RNG streams (7000–7002), and report
the SD of the bound across streams as the Monte-Carlo error of the interval
(typically ±0.01 RMSE). An interval whose Monte-Carlo error is comparable to the
margin it is being compared against cannot adjudicate that margin — a point §4.3 makes
concrete.

**Resolution.** The minimum detectable effect at n engines is
`MDE(n) = (z_{0.975} + z_{0.80}) · sd_i(ΔRMSE_i) / √n`, with the per-engine RMSE
difference as the unit and the two-sided 5% / 80% power convention. This is a paired
quantity: it asks how precisely a *re-run on the same engines* can be distinguished,
and it is therefore much smaller than the between-engine spread. We report it at the
engine count actually available, and scale it in n only when a statement is explicitly
about a different corpus size.

**Multiplicity.** Comparisons within a family are reported with Holm-adjusted
intervals; an unadjusted interval that crosses zero cannot survive Holm, so we state
which comparisons are unadjusted.

## 3.5 Pre-registration and the amendment policy

Every experiment that could produce a number used in the paper has a frozen
pre-registration with a recorded sha256, written before the first run: E2 (monotone
health index), E3 (bounded encoding, FD002), E3b (FD001 replication), E3c
(scrambled-time control), E4 (reproduction). The pre-registration fixes the arms, the
seeds, the budget, the decision rule and what a failure would mean.

Amendments are appended, never substituted: the original text is immutable, and a
change is recorded with a timestamp in the pre-registration's amendment section and in
the experiment's result note. We made three such corrections during this work — a
validation-time leak, a non-converged bootstrap, and a claim whose artefact did not
exist — and each is reported as a withdrawal rather than silently repaired
(§4.4, §4.2, §4.3).

## 3.6 Compute, artefacts and provenance

All neural runs execute on a single NVIDIA A40 (46 GB) with PyTorch 2.5.1 / CUDA 12.1
/ Python 3.9. Each run writes `metrics.json` (metrics, parameter count, wall-clock
bounds, `prereg_sha`, and the sha256 of the runner and model files),
`matched_cutoff_predictions.csv` (one row per held-out engine), the best checkpoint
and the training history. Code is versioned by hash, not by commit message, because
the project has no repository history to rely on.

Reproducibility is measured, not asserted: §7 reports a replay from stored weights
(max |Δprediction| 7 × 10⁻⁶) and a from-scratch rerun in an empty directory
(ΔRMSE = 0.00000, same best epoch).
