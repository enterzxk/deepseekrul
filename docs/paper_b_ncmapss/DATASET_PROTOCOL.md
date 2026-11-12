# N-CMAPSS DS02 dataset protocol

Status: **protocol fixed, data dictionary pending measurement** (owner: zxk).
Base commit: `ac94c29`. Nothing in this file may be relaxed later without a
recorded decision.

## 1. Why this dataset is in the paper

C-MAPSS is saturated (see `docs/paper_b_baselines/BASELINE_PROTOCOL_AUDIT.md`
§6): the whole method space spans ~1.9 RMSE and the cycle feature carries most
of the signal. DS02 is a different regime — multi-flight, variable operating
conditions, degradation trajectories simulated under **real flight-condition
profiles** — and is used as a **cross-dataset check**, not as a new benchmark to
tune on.

**DS02 is simulation, not field data.** It must never be described as measured
failure data from real engines. The honest phrasing is "simulated run-to-failure
trajectories driven by recorded flight profiles".

## 2. Data dictionary

Status: **semantics fixed from the official documentation; shapes and dtypes
pending measurement** (`paper_b/ncmapss/inspect_ds02.py`, owner: zxk).

Source: Arias Chao, Kulkarni, Goebel & Fink, *Aircraft Engine Run-to-Failure
Dataset under Real Flight Conditions for Prognostics and Diagnostics*, **Data**
6(1):5, 2021, doi:10.3390/data6010005, Tables 3–8. Every semantic row below is
quoted from that source. The `measured` column is filled from the HDF5 file and
must agree with the documented meaning; a disagreement is a finding, not a
detail to smooth over.

### 2.1 Correction to the earlier draft of this file

The first draft of §2 said "`T` is the ambient/operating context" and §3 banned
`T` as "ambient temperature / scenario descriptor". **Both statements are
wrong.** In the N-CMAPSS HDF5 file:

- `T` holds the **model health parameters θ** (Table 7): the ten
  efficiency/flow modifiers the simulator uses to impose degradation. It is the
  ground-truth degradation state, not a temperature.
- The ambient/operating context is `W`, four scenario descriptors. The only
  temperature inside `W` is `T2`, *total temperature at the fan inlet* — a
  column of `W`, not the `T` array.

This is precisely the field-name trap this section exists to prevent: `T2` is a
column of `W`; `T` is the answer key. The ban on `T` stands, but for a much
stronger reason than the draft gave (§3.2).

### 2.2 HDF5 layout (Table 3)

| group | arrays |
|---|---|
| development | `W_dev`, `X_s_dev`, `X_v_dev`, `T_dev`, `Y_dev`, `A_dev` |
| test | `W_test`, `X_s_test`, `X_v_test`, `T_test`, `Y_test`, `A_test` |
| variable names | `W_var`, `X_s_var`, `X_v_var`, `T_var`, `A_var` |

The `*_var` arrays hold the column names, so the column order must be read from
them at runtime rather than assumed.

### 2.3 Fields

| array | columns | physical meaning | observable at deployment? | source |
|---|---|---|---|---|
| `W` | 4 | scenario descriptors: `alt` [ft], `Mach` [-], `TRA` [%], `T2` [°R] | **yes** — recorded flight conditions | Table 4 |
| `X_s` | 14 | measured sensors: `Wf` [pps], `Nf`/`Nc` [rpm], `T24`/`T30`/`T48`/`T50` [°R], `P15`/`P2`/`P21`/`P24`/`Ps30`/`P40`/`P50` [psia] | **yes** — the condition-monitoring signal | Table 5 |
| `X_v` | 14 | virtual sensors: `T40`, `P30`, `P45`, `W21`, `W22`, `W25`, `W31`, `W32`, `W48`, `W50`, `SmFan`, `SmLPC`, `SmHPC`, `phi` | **no** — simulator-internal quantities, not measured | Table 6 |
| `T` | 10 | health parameters θ: `fan_eff_mod`, `fan_flow_mod`, `LPC_eff_mod`, `LPC_flow_mod`, `HPC_eff_mod`, `HPC_flow_mod`, `HPT_eff_mod`, `HPT_flow_mod`, `LPT_eff_mod`, `LPT_flow_mod` | **no** — ground-truth degradation state | Table 7 |
| `Y` | 1 | RUL in **flight cycles** | **no** — the label | Table 3 |
| `A` | 4 | auxiliary: `unit`, `cycle` (flight cycle number), `Fc` (flight class), `hs` (health state) | partially — see §3.3 | Table 8 |

Sampling is 1 Hz within a flight; `cycle` counts flights. The two axes are not
interchangeable (§4).

### 2.4 Documentation inconsistencies to resolve at runtime

1. Table 3 states the file stores "17 variables", but Tables 4–8 enumerate
   4 + 14 + 14 + 10 + 1 + 4 = 47 columns. Count the actual columns and record
   which reading is right.
2. Table 7 row 10 gives the symbol `LPT_flow_mod` with the description "HPT
   flow modifier". The symbol is almost certainly correct and the description is
   a typo; confirm against `T_var` at runtime.
3. The paper does not state whether `Y` is constant within a flight cycle.
   Measure it: if `Y` varies inside a cycle, the flight-level label needs an
   explicit aggregation rule.
4. The paper does not state the dtype (float32/float64) or whether the arrays
   are chunked. Both matter for memory on a 6.5 M-row file.

## 3. Observability boundary (hard rule)

At inference time the model may see **only** what a deployed system sees. The
following are forbidden as model inputs under every variant:

- `T` — the ten health parameters θ, i.e. the ground-truth degradation state.
- `Y` (the RUL target) and anything derived from it, in any form.
- `X_v` — the fourteen virtual sensors, which are simulator-internal.
- Any future trajectory of `X_s` / `X_v`.
- `hs` (health state) and any other label-like auxiliary column.
- Any label computed from the **complete** flight, when the claim is a
  prediction made *before* that flight ends.

Violating any of these invalidates the run, not just the claim.

### 3.1 `X_v` — virtual sensors: forbidden

`X_v` is not a second sensor set. The CMAPSS model computes it internally
(`[x_s(t), x_v(t)] = F(w(t), θ(t))`, §2.1 of the dataset paper) and it is
released as a convenience for diagnostics and physics-informed work. Several of
its columns are close to the failure mechanism itself: `SmFan`, `SmLPC`,
`SmHPC` are the stall margins whose normalized deficit *defines* the health
index, and `phi` is a derived flow ratio.

Consequence: any model that consumes `X_v` is reading a simulator quantity that
a deployed system cannot measure. A DS02 result built on `X_v` is not a
prognostics result and may not be compared with C-MAPSS numbers.

Audit hook: the loader must raise if `X_v` is passed as a feature, and the
experiment config must not contain an `x_v` key at all. This is enforced in
`paper_b/ncmapss/data.py` by an explicit allow-list of feature arrays.

### 3.2 `T` — health parameters: forbidden, and the strongest leak in the file

`T` is the ten-dimensional θ vector that the generator drives from initial wear
through normal degradation to failure. `Y` is a deterministic function of the
same trajectory. Feeding `T` (or any monotone transform of it) is therefore
equivalent to feeding the label with extra steps, and it is strictly stronger
than `Y` because it also reveals the fault mode.

This is why the earlier draft's ban on `T` as "ambient temperature" was
misleading: the ban was right, the reason was wrong, and a reader could have
concluded that `T2` (a `W` column) was also banned. It is not: `T2` is a
recorded flight condition and is allowed with the rest of `W`.

Audit hook: `data.py` never loads `T_dev`/`T_test` into the feature path at all;
the arrays are only reachable through a `load_health_parameters()` helper that
is called from diagnostics scripts, never from the training path. The runtime
audit records `T` in the file inventory but asserts it is absent from every
feature matrix.

### 3.3 Auxiliary columns `A`

| column | status |
|---|---|
| `unit` | **never a feature.** Unit identity correlates with lifetime; as an input it is direct leakage. Used only for grouping and LOEO. |
| `cycle` | usable as an observable age counter (it is the flight index a deployed system knows). Its use must be declared in every table, exactly as in the C-MAPSS audit §6.3. |
| `Fc` (flight class) | **not a default feature.** It is assigned once per unit and correlates with flight length and lifetime. If used, it must be justified as operationally available and tested as an explicit ablation, never silently. |
| `hs` (health state) | **never a feature.** It is a label. |

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

C-MAPSS reports in this project are capped at 125. **We do not know which
convention the published C-MAPSS numbers use, and we may not assert that ours is
"the literature convention".** That inference was retracted in `D041`: a cap on
the *training label* does not constrain the *inference-time output*, so a model
trained on capped labels can extrapolate past 125, and a raw-label reading of the
published values is equally consistent. The only claim that survives is the
conditional arithmetic bound — *if* a model truncates its output at 125, its
raw-label RMSE is at least `B` (3.74 / 18.32 / 4.71 / 19.53 for FD001–FD004) —
and that premise has not been verified for any of the papers in the matrix.

Consequence for the paper: our capped numbers are comparable to a published
number **under the stated assumption** that the same convention was used. The
assumption is named in every table, never presented as established fact.

For DS02 **both label variants are stored and reported**: the raw RUL and the
label capped at 125, with every table stating which one it is showing and
whether the cap is in force. Neither variant replaces the other, and **capped
and raw results may never be placed in the same column, table, or sentence**.

The cap is a C-MAPSS convention carried over deliberately, not a property of
DS02: the raw variant is the stricter criterion and is reported for that reason.
Inventing a cap other than 125, or reporting only the variant that flatters the
method, is a protocol violation.

## 8. Prohibitions

- No tuning on the official test units.
- No re-decision of the MS / SA / HI switches based on DS02 outcomes (N3 is
  frozen by construction — see `BASELINE_RESULTS.md`).
- No comparison of a DS02 number against a C-MAPSS number without stating both
  protocols.
- No 1 Hz index used as a cycle, and no within-flight row count used as a unit.
