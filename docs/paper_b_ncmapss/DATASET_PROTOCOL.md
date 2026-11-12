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

### 4.1 Feature contract (frozen before any run)

Every rung consumes the **same** flight-level matrix; only the model differs.
This is the information-matching rule: if N2 beats N1, the difference may not be
explained by "it was given more".

Per flight `(unit, cycle)`, built from `X_s` and `W` only:

| block | definition | columns |
|---|---|---|
| sensor statistics | for each of the 14 `X_s` sensors: mean, std, OLS slope over that flight's 1 Hz samples | 42 |
| operating conditions | for each of the 4 `W` columns: mean, std | 8 |
| age | `cycle`, divided by a scale fitted on the training fold (95th percentile of its observed cycles) | 1 |
| **total** | | **51** |

Rules:

- The 51 columns are **identical for N1, N2 and N3**. N3 differs only in its input
  layer and encoder — never in what it is given. N0 receives the age column only.
- Normalization (z-score) is fitted on the training folds of the current LOEO
  fold and applied unchanged to the held-out unit. Never fitted on the held-out
  unit, never on a test unit.
- `X_v`, `T`, `Y` and `A.hs` are absent by construction; `unit` is a grouping key
  only.
- `A.Fc` (flight class) and flight duration are **not** among the 51. If either
  is ever used, it is a declared, separately labelled ablation — never a silent
  addition.
- Quantile- and phase-resolved statistics are likewise declared extensions, not
  part of this contract.

Sequence construction: one row per flight, ordered by `cycle`; the model sees the
last **70 flights**, right-aligned at the prediction flight, left zero-padded
with a validity mask — the same construction as C-MAPSS `span=70`
(`paper_b/training/data.py::_window`). Using the same span on both datasets is
what makes "same core structure" checkable rather than asserted.

The 70 is provisional until the flight counts are measured: if DS02 lifetimes make
it degenerate, the value is changed **once, before any run**, recorded here with
its justification, and used for every rung. It is not a knob to revisit after
seeing results.

Label: `Y` at the prediction flight, in both raw and capped-at-125 form (§7).

### 4.2 Training plan and conditioning (frozen)

The **schedule** is taken verbatim from `paper_b/configs/protocol.json`: lr 1e-3,
weight decay 1e-4, d_model 64, 2 layers, dropout 0.2, batch `min(512, N_train)`,
seeds 101 / 202 / 303. Only the input width changes (51 columns instead of the
C-MAPSS sensor block). `patience: 12` is present in that file but is **unused
here** — see the stopping rule below.

**Budget: 200 passes with `batch = min(64, N_train)`, and the target standardised
fold-locally.**
Measured from the file before any run: DS02 dev holds 446 flights over six units,
so a LOEO fold trains on 380 windows. With `batch = min(512, N)` that is one
gradient step per pass — 40 updates in total, a warm-up rather than a training
run.

Two corrections were needed, both found by diagnostics run before any cell was
produced, and both applied identically to all four rungs:

1. **Target standardisation.** `y` is standardised with the mean/std of the
   **training units of the current fold** and the prediction is mapped back to
   cycles before anything is written. Without it the head starts ~40 cycles away
   from its target and no rung converges: measured holdout RMSE on fold 1 was
   N0 15.23 / N1 21.71 against a linear-ridge reference of 5.19 on the same
   features. With it, N0 reaches 2.45 in the same 40 passes.
2. **Budget raised to 200 passes.** 200 passes × 6 updates = ~1200 parameter
   updates, the same order as C-MAPSS's ~1560. 40 passes (240 updates) leaves
   every rung under-trained; 800 passes starts memorising (N2 degrades from 6.81
   to 13.99). The pass budget is fixed, not chosen per rung.

| quantity | C-MAPSS | DS02 |
|---|---|---|
| passes over the data | 40 | 200 |
| batch | 512 | `min(64, N_train)` = 64 |
| updates per pass | ~39 | 6 |
| total updates | ~1560 | ~1200 |
| target | raw cycles | standardised fold-locally, inverted before writing |

`epochs = 200`, `batch = min(64, N_train)`. Applied identically to all four
rungs. This replaces `protocol.json`'s `epochs` and `batch` fields for DS02
only; the C-MAPSS line keeps its own schedule, and the difference is recorded
here so the "same core structure" claim is bounded exactly as stated below.

Because no early stopping is allowed (see below), the reported model is the one
at the end of this budget. The full training and holdout curves are therefore
reported alongside every number, and the holdout **minimum and the epoch at which
it occurs are reported as a diagnostic** — never used to pick the model, only to
show whether the fixed endpoint sits on a slope or at a floor. A fixed budget
hides over-fitting only if the curve is hidden with it.

The **label-construction** fields in the same file are **not** inherited. They are
C-MAPSS-specific and are re-defined here, once, and frozen:

| field | C-MAPSS value | DS02 |
|---|---|---|
| `rul_cap` | 125 | **inactive**: the longest DS02 engine is 89 flights (dev) / 76 (test), so `min(raw,125) == raw` everywhere. Both columns are still written; the capped column is reported as `inactive (no unit exceeds the cap)`. |
| `val_cuts` | `[0.4, 0.55, 0.7, 0.85, 0.95]` | not used — validation is a held-out engine, not a lifetime cut |
| `val_frac` | 0.2 | not used — LOEO replaces the 80/20 split |

Inheriting `rul_cap` silently would transplant the C-MAPSS label truncation onto
a dataset whose lifetime scale is different, and would redefine the raw criterion
by accident — exactly the confusion `D043` bounds.

**No early stopping.** Every rung trains for the fixed epoch budget and the final
epoch is the reported model. Early stopping on the held-out engine would mean the
engine used for the reported score also chose the stopping point; that is a
selection bias, not a training detail. Training and validation loss curves are
recorded for every run so that under-training is visible rather than inferred.

No rung gets a different schedule. If a rung fails to converge under this plan,
that is a reported finding, not a licence to retune — the same rule that keeps N3
frozen.

Scope of the "same core structure" claim, stated so it cannot be over-read: same
architecture, same optimizer, same hyper-parameters as the C-MAPSS line; the
**stopping rule deliberately differs** — C-MAPSS early-stops on its validation
split, DS02 trains a fixed budget and reports the final epoch, because here the
validation engine is also the held-out engine. A reader who assumes the training
procedure is identical end-to-end would otherwise be misled, and the difference
costs nothing: DS02 carries the "does the mechanism hold on another dataset"
claim, not a bit-for-bit re-run claim.

Operating-condition conditioning: the DSPT arms receive a cluster embedding, as
on C-MAPSS (`n_cond = 6`). For DS02 the clusters are k-means (k = 6) over the `W`
block, **fitted on the training folds of the current LOEO fold only**. This adds
no information that N1 lacks: the same `W` statistics are already columns 43–50
of the shared matrix, and the embedding is only a different encoding of them. Any
change that would give the DSPT arms a strictly richer input is a protocol
violation, not an architecture choice.

N3 reads the frozen C-MAPSS configuration from `paper_b/results/FREEZE_ABLATION_SEED101.json`
and `paper_b/configs/protocol.json`. It does not carry a hand-written copy of those
values on the DS02 side: a copied parameter is not evidence that anything was
frozen.

## 5. Splits (immutable)

- dev units: `[2, 5, 10, 16, 18, 20]`
- official test units: `[11, 14, 15]`

These are fixed. They may not be re-drawn, re-ordered, or augmented, and the
official test units may not be evaluated before the model and the analysis rule
are frozen.

## 6. Evaluation protocol

Implementation: `paper_b/ncmapss/evaluation.py` (`python -m
paper_b.ncmapss.evaluation --self-test` checks the rules below without any DS02
data on disk).

### 6.1 Unit of analysis

The unit is the **flight**, never the 1 Hz sample: one prediction and one true
RUL per `(unit, cycle)`, taken at the end of that flight (see §4). Averaging
over samples would weight a 3-hour flight 4.5x a 40-minute one while the
protocol requires equal weight per flight.

Three checks are enforced in code, each targeting one concrete way of getting
the unit wrong:

1. `(unit, cycle)` must be unique — a duplicate means 1 Hz rows reached the
   scorer.
2. `cycle` must be integer-valued — a fractional cycle is a timestamp.
3. `cycle` must be strictly increasing within each unit — a collapsed 1 Hz
   index fails here because its surviving spacing equals the flight length.

`cycle_provenance()` additionally records `first_cycle / last_cycle / span /
gaps` per unit as audit evidence: a genuine flight counter is contiguous
1..N, whereas a surviving sample index leaves large gaps.

### 6.2 LOEO folds and fit discipline

- LOEO over the six dev units: six folds, each holding out one unit entirely.
  This is the only cross-validation scheme used for development and selection.
- **Every fitted quantity** — scaler, operating-condition encoder, RUL
  normaliser, and any hyperparameter chosen from data — is fitted on the
  current training fold only. Fitting on all engines is a silent leak: the
  validation engine's range enters the normalisation.
- `assert_fold_discipline(fold, fit_units)` requires `fit_units` to equal the
  fold's five training units exactly and names the leaking engine otherwise.
- `assert_no_test_units()` rejects `[11, 14, 15]` anywhere on the dev side.

### 6.3 Metrics

All computed at flight level with `n` flights and `d = y_hat - y`.

| metric | definition | unit | criterion-sensitive | note |
|---|---|---|---|---|
| RMSE (pooled) | `sqrt(mean(d^2))` over all flights | cycles | yes | **supplementary**: reported beside the primary number, never instead of it |
| MAE | `mean(|d|)` | cycles | yes | read with RMSE to see tail dominance |
| NASA sum | `sum s_i`, `s_i = exp(-d/13)-1` if `d<0` else `exp(d/10)-1` | dimensionless | strongly | **not comparable across datasets**, see 6.7 |
| NASA mean | `sum s_i / n` | dimensionless | strongly | the only comparable form when point counts differ |
| Bias | `mean(d)` | cycles | yes | positive = late (dangerous side) |
| segment RMSE | per segment, two rules, see 6.4 | cycles | yes | early/mid/late shape |
| per-engine RMSE | within each unit, then equal weight per engine | cycles | yes | **primary**: D043 fixes per-engine equal weight as the headline number; also guards against one broken engine hiding in the mean |
| late_frac / severe_late | fraction of flights with `d > 0` / with `d > SEVERE_LATE_CYCLES` | fraction | yes | `SEVERE_LATE_CYCLES = 10` cycles is a named reporting convention, not a protocol threshold |
| seed aggregate | `mean +- sample sd (ddof=1)`, `n_seeds >= 3` | inherited | inherited | fewer than 3 seeds is refused |

**Primary vs supplementary.** The per-engine equal-weight RMSE is the headline
number (D043); the flight-pooled RMSE is reported beside it as a supplement.
They are not interchangeable: by Jensen the pooled value is at least the
per-engine mean, so quoting whichever happens to be smaller is a reporting
error. Every table states which column is which.

The NASA score is non-linear and amplifies the tail: a single `d = +40` term is
`exp(4) - 1 ~= 53.6`. It can therefore order models differently from RMSE; when
the two disagree, the text says so rather than quoting the flattering one.

### 6.4 Two segment rules, both reported

| rule | definition | tag | role |
|---|---|---|---|
| relative | `frac = RUL / L_unit`, where `L_unit` is the max RUL over that unit's **full label trajectory** supplied by the data layer; `late < 1/3`, `mid < 2/3`, `early >= 2/3` | `relative_thirds_primary` | **primary**: scale-free across units with different lifetimes |
| absolute | fixed cut points in cycles: `early >= 75`, `mid 25-75`, `late < 25` | `absolute_rul_supplementary` | **supplementary**: the rule the C-MAPSS tables use, needed to align early/mid/late *shapes* across datasets |

**The relative scale must come from the unit's lifetime, not from the frame
being scored.** Deriving it as the maximum RUL of the evaluated rows works on
the dev folds, where a unit contributes its whole trajectory, but it silently
breaks on the official test set: the frozen protocol scores one flight per test
unit, so that maximum equals the single row's own RUL, every fraction becomes
1.0, and the entire test set is labelled `early` with no error raised.
`evaluate()` therefore requires `unit_lifetime`; omitting it raises, and a
lifetime smaller than an observed RUL raises as well (that would mean the
lifetime belongs to a different unit).

Both are always produced, and every segment row carries `segment_rule`. Table
captions state which rule is primary. The absolute cut points are frozen in
`seg_boundaries` before any result is seen and may not be tuned afterwards. The
two rules may partition the flights differently; their results never share one
table.

### 6.5 Multi-seed aggregation

- At least three seeds; report `mean +- sample sd (ddof=1)`. Fewer than three
  is refused rather than estimated from two points.
- Aggregation rejects a mix of criteria (see 6.6).
- **The independent unit is the engine, not the fold-by-seed cell.** Six dev
  engines are six independent units; the 18 fold-by-seed numbers are not 18
  samples, and any interval taken across seeds uses `df = n_engines - 1`.
- Seed variation is training randomness, not evidence of significance. Method
  comparisons use same-seed pairing plus sign consistency.
- On the official test set there are three units and three seeds per unit:
  report each unit's error and the seed mean +- seed sd. There is no
  cross-engine dispersion to estimate, so no "cross-engine 95% CI" may be
  written.

### 6.6 Criterion hygiene

- Both label variants are always produced (`evaluate_both()`), never one alone.
- Every result row and every table carries `criterion` and `rul_cap_used`;
  `assert_single_criterion()` refuses a table that mixes them.
- Capped and raw may never share a column, a table, a ranking, or a sentence,
  and a DS02 number may never be placed in one ranking with a C-MAPSS number
  under the other criterion (see §7 and §8).

### 6.7 NASA sum is not comparable across datasets

C-MAPSS contributes one point per test engine (100 / 259 / 100 / 248), DS02 one
point per flight (a much larger count). The order-of-magnitude difference in
NASA sum between the two datasets is driven by the number of terms, not by
performance. Cross-dataset statements therefore use NASA **mean**, or compare
within-dataset orderings only.

### 6.8 Test gate

`assert_test_allowed(freeze_manifest, tag, seed)` requires a manifest with
`frozen: true` that lists the `(tag, seed)` pair, otherwise it exits. The
semantics match `paper_b/evaluation/evaluate_test.py`: the manifest declares
*which cells will be evaluated before the test set is opened*, so a manifest
written after the fact is a rewrite of the evaluation order and is never
accepted.

### 6.9 Artifacts

```
<out_dir>/
  summary.json           # both criteria, criterion/rul_cap_used flags, segment boundaries
  per_engine_capped.csv  # unit, n_flights, rmse, mae, nasa_sum, bias, criterion, rul_cap_used
  per_engine_raw.csv
  segments_capped.csv    # segment, segment_rule, n_flights, rmse, ..., criterion
  segments_raw.csv
```

Every CSV carries its own `criterion` / `rul_cap_used` (and `segment_rule` for
the segment tables), so a file lifted out of context still cannot be merged
into a table of the other criterion.

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
