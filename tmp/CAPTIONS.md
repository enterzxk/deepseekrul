# 图注与表注（draft）

## Figures

**Fig. 1 — Observability of the individual lifetime scale.**
Pearson correlation between the sensor-derived total-lifetime estimate mu_T and the true
total life T, as a function of the number of observed cycles at the prediction cutoff, for
the two multi-condition fleets (FD002, FD004). Each point pools the outer-validation
engines of the five engine-disjoint folds at a single cutoff (n = 240-260 engines per
point, decreasing at long cutoffs because engines must survive to the cutoff). The
estimate is nearly uninformative below 80 observed cycles (rho = 0.26 on FD002) and
approaches saturation above 150 cycles (rho = 0.92 on FD002, 0.88 on FD004). Source:
`observability.json`; plotted by `make_paper_assets.py`.

**Fig. 2 — Pooled RMSE by arm and fleet.**
Engine-equal-weight pooled RMSE (bars, mean over available seeds; error bars, standard
deviation over seeds) for the plain backbone (M0), the age-and-condition prior (OPS), the
sensor-history prior (FULL), and the lifetime-normalised time base (NORM), on the
outer-validation engines. Source: `RESULTS.csv`.

**Fig. 3 — Error versus observation age.**
Pooled RMSE as a function of the cycle at the window end, for M0 and FULL on FD002 and
FD004. The error falls sharply beyond 150 observed cycles, which is the same region in
which the lifetime scale becomes observable (Fig. 1). Source: `make_extra_figures.py`.

**Fig. 4 — Error by realised lifetime quartile.**
Pooled RMSE and signed bias for engines in the four quartiles of realised total life. The
shortest-lived quartile carries the largest error and a large positive bias (over-
estimation of the remaining life), which the additive prior reduces but does not remove.
Source: `PER_ENGINE.csv`.

**Fig. 5 — Measurement conditions.**
Pooled RMSE of the same trained model under the clean condition, 20% random missing
samples with forward fill (MC1), and a one-standard-deviation offset on the three
channels with the strongest lifetime association (MC2). Source:
`MEASUREMENT_CONDITIONS.csv`.

## Tables

**Table I — Frozen protocol.** Folds, budget, cap, metrics, intervals, test-use history.
See `PROTOCOL.md`.

**Table II — Main four-metric table.** RMSE, MAE, NASA score (per-window mean and sum) and
signed bias for every arm, with the engine-clustered paired bootstrap interval of the RMSE
difference against the locked reference. Source: `RESULTS.csv`.

**Table III — Ablations.** Each variant against M0 on the folds where both exist, with the
paired interval. Source: `ABLATIONS.csv`.

**Table IV — Observability.** MAE, RMSE, correlation and dispersion of mu_T at each cutoff.
Source: `paper_tables/observability.csv`.

**Table V — Identifiability.** Correlation of each candidate observable with the
individual-versus-population advantage, and the RMSE of the corresponding gate. Source:
`IDENTIFIABILITY.md`.

**Table VI — Measurement conditions.** Four metrics under clean, MC1 and MC2. Source:
`MEASUREMENT_CONDITIONS.csv`.

**Table VII — Endpoint dependence.** The same predictions scored at all windows, at one
truncation per engine, and on the official test engines. Source:
`ENDPOINT_DEPENDENCE.md`.
