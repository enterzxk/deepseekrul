# N-CMAPSS DS02 dataset protocol

Status: **protocol fixed, data dictionary pending measurement** (owner: zxk).
Base commit: `ac94c29`. Nothing in this file may be relaxed later without a
recorded decision.

## 1. Why this dataset is in the paper

C-MAPSS is saturated (see `docs/paper_b_baselines/BASELINE_PROTOCOL_AUDIT.md`
§6): the whole method space spans ~1.9 RMSE and the cycle feature carries most
of the signal. DS02 is a different regime — multi-flight, variable operating
conditions, real degradation trajectories — and is used as a **cross-dataset
check**, not as a new benchmark to tune on.

## 2. Data dictionary (PENDING — zxk)

Must be measured from the HDF5 file and reconciled against the official
documentation, **not** inferred from field names. One row per field:

| field | shape | dtype | physical meaning | observable at deployment? | source |
|---|---|---|---|---|---|
| `X_s` | | | | | |
| `X_v` | | | | | |
| `T` | | | | | |
| `Y` | | | | | |
| `A` | | | | | |
| `W` | | | | | |

`X_s` / `X_v` are sensor arrays at 1 Hz; `T` is the ambient/operating context;
`Y` and `A` are derived targets/auxiliary quantities whose observability is the
crux of §3.

## 3. Observability boundary (hard rule)

At inference time the model may see **only** what a deployed system sees. The
following are forbidden as model inputs under every variant:

- `T` (ambient temperature / scenario descriptor) — it identifies the scenario,
  which encodes the degradation regime.
- `Y` (the RUL-like target) and anything derived from it, in any form.
- Any future trajectory of `X_s` / `X_v`.
- Simulation internal state (health parameters, fault magnitudes, model
  coefficients) that is not recoverable from sensors.
- Any label computed from the **complete** flight, when the claim is a
  prediction made *before* that flight ends.

Violating any of these invalidates the run, not just the claim.

## 4. Flight-level representation

- Group by `(unit, cycle)` and compute per-flight statistics: mean, standard
  deviation, quantiles, and flight-phase descriptors. Operating conditions are
  preserved as a separate conditioning input, never folded into the statistics.
- **`cycle` is the cumulative flight cycle index, never the 1 Hz sampling
  index.** A 1 Hz row counter is a within-flight clock; using it as `cycle`
  would leak the flight duration and destroy the meaning of the unit axis.
- **Prediction instant = end of flight.** If any feature is a statistic over the
  complete flight, the model cannot claim in-flight prediction. Every result
  table must state which of the two it is.

## 5. Splits (immutable)

- dev units: `[2, 5, 10, 16, 18, 20]`
- official test units: `[11, 14, 15]`

These are fixed. They may not be re-drawn, re-ordered, or augmented, and the
official test units may not be evaluated before the model and the analysis rule
are frozen.

## 6. Evaluation protocol

- **LOEO (leave-one-engine-out) over the six dev units**: six folds, each
  holding out one unit entirely. This is the only cross-validation scheme used
  for development and model selection.
- Selection uses validation folds only. The three official test units are opened
  once, after the freeze manifest is written and committed.
- Report per-fold values, the mean and dispersion across folds, and the paired
  deltas with intervals. A single aggregate number is not a result.

## 7. Label convention

C-MAPSS reports are capped at 125 and that is the literature-consistent
convention (see the audit §2b). For DS02 the convention must be stated
explicitly in every table, and **capped and raw results may never be placed in
the same column, table, or sentence**. If DS02 has no natural cap, say so and
report raw only — do not invent a cap to make the two datasets look alike.

## 8. Prohibitions

- No tuning on the official test units.
- No re-decision of the MS / SA / HI switches based on DS02 outcomes (N3 is
  frozen by construction — see `BASELINE_RESULTS.md`).
- No comparison of a DS02 number against a C-MAPSS number without stating both
  protocols.
- No 1 Hz index used as a cycle, and no within-flight row count used as a unit.
