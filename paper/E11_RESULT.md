# E11 — Result: five architectures do separate, and the thesis has to narrow

Run: 2026-10-30 23:45 – 2026-10-31 09:06 local, A40. 55 usable runs: three architectures
with 3 seeds × 5 folds, two with 1 seed × 5 folds (Amendment 1), plus the A and BE arms
from E10/E10b.
Pre-registration: `paper/E11_ARCH_PREREGISTRATION.md`, frozen at `1e63c800`, amended at
`b8eeb757` (single-seed TCN/Transformer, before any of their runs existed).
Analysis: `scripts/dsh_e11_arch_analyze.py`, `results/dsh/e11/E11_arch.json`.

## Verdict: **DISPERSED** — and all three pre-registered predictions failed

## 1. The table

FD002, 5-fold cross-validation, every engine held out exactly once. Values are the mean
over (fold, seed) runs of the per-run metric, so a 3-seed arm gets no ensemble advantage.
Intervals from seed 2027 in every arm (the only seed all seven share).

| arm | seeds | RMSE | MAE | NASA | late | severe |
|---|---:|---:|---:|---:|---:|---:|
| **A** (candidate) | 3 | **13.953** | 10.257 | **4.091** | 0.508 | **0.228** |
| tf (Transformer) † | 1 | 15.448 | **11.281** | 5.250 | 0.504 | 0.250 |
| gru (GRU+attn) | 3 | 15.971 | 12.161 | 4.975 | 0.504 | 0.268 |
| bilstm (BiLSTM) | 3 | 16.199 | 12.231 | 5.565 | 0.519 | 0.267 |
| tcn (dilated TCN) † | 1 | 19.820 | 15.482 | 8.560 | 0.538 | 0.308 |
| cnn1d (1-D CNN) | 3 | 19.968 | 15.722 | 8.041 | **0.477** | 0.274 |

† single seed (Amendment 1).

Against A, all five are worse with intervals clearing zero:

| contrast | ΔRMSE | 95% CI |
|---|---:|---|
| cnn1d − A | **+6.280** | [+4.236, +8.438] |
| tcn − A | **+6.842** | [+4.830, +8.848] |
| gru − A | **+3.499** | [+1.959, +5.096] |
| bilstm − A | **+3.168** | [+1.628, +4.758] |
| tf − A | **+2.444** | [+0.444, +4.418] |

**Spread across architectures: 6.01 RMSE (13.95 to 19.97).**

## 2. The pre-registered predictions, checked

| # | prediction (frozen before the runs) | outcome |
|---|---|---|
| 1 | max-min spread across architectures **1.0-4.0 RMSE** | **6.01 - failed** |
| 2 | **at least half** of the 10 pairwise contrasts have a CI crossing zero | **0 of 5 vs A cross zero - failed** |
| 3 | per-fold ranking unstable (fold-0 leader not leading on >=2 other folds) | **the fold-0 leader leads on 4 of 5 folds - failed** |

**All three predictions failed.** Under the rule frozen in section 4 of the pre-registration
that is reported as a failure and it weakens the paper's original thesis; it is not
re-interpreted after the fact.

The per-fold table makes the failure legible:

| arm | f0 | f1 | f2 | f3 | f4 |
|---|---:|---:|---:|---:|---:|
| A | 12.68 | 13.49 | 15.09 | 14.35 | 14.16 |
| BE | - | **13.18** | 16.18 | 15.92 | 14.74 |
| tf | 15.67 | 14.93 | 16.47 | 15.19 | 14.97 |
| gru | 17.72 | 14.42 | 15.37 | 15.80 | 16.54 |
| bilstm | 15.78 | 16.47 | 16.72 | 15.66 | 16.36 |
| tcn | 18.64 | 20.80 | 20.18 | 18.19 | 21.29 |
| cnn1d | 19.20 | 19.61 | 19.85 | 18.11 | 23.06 |

Per-fold ordering: f0 `A > BE > tf > bilstm > gru > tcn > cnn1d`; f1 `BE > A > gru > tf >
bilstm > cnn1d > tcn`; f2 `A > gru > BE > tf > bilstm > cnn1d > tcn`; f3 `A > tf > bilstm >
gru > BE > cnn1d > tcn`; f4 `A > BE > tf > bilstm > gru > tcn > cnn1d`.

## 3. What the failures actually show

The architectures do not form a continuum whose ordering is noise. They form **three
bands**, and the failure of prediction 3 is the clearest evidence for it: the instability
is not uniform across the table, it is confined to the middle.

| band | members | RMSE | within-band spread | rank stability |
|---|---|---:|---:|---|
| best | A | 13.95 | - | **1st on 4 of 5 folds** |
| middle | tf 15.45, gru 15.97, bilstm 16.20 | 15.45-16.20 | **0.75** | **all three orderings occur** |
| worst | tcn 19.82, cnn1d 19.97 | 19.82-19.97 | **0.15** | **bottom two on every fold** |

* **Between bands:** 2.4 to 6.8 RMSE, every contrast significant. Architecture choice
  matters, and the paper says so.
* **Within the middle band:** 0.75 RMSE spread, and the three members swap places across
  folds - `tf` leads twice, `gru` twice, `bilstm` never, with no fold-level pattern. That is
  0.75 against `u_c` = 0.61-1.23 (section 4.6): unresolvable.
* **Within the worst band:** 0.15 RMSE apart and in the bottom two on every fold. Also
  unresolvable, but consistently so.

So the correct claim is not "method differences are unresolvable". It is:

> **On C-MAPSS at 260 held-out engines, architecture differences above about 1.5 RMSE are
> resolvable and real; differences below about 1.2 RMSE are not.** The boundary is set by
> the uncertainty budget, and it falls *between* architecture families rather than inside
> them.

Prediction 3 failed in the letter - I expected the leader to change and it did not - but
the instability I predicted did occur, in the middle band rather than at the top. Both
facts are reported; the pre-registered verdict stands as DISPERSED.

## 4. The single-split bias, again

E10b showed the single split overstated BE − A by 2.4×. The architecture comparison
repeats the pattern in the same direction: on fold 0 alone, `gru` − A is **+2.10**, while
the five-fold estimate is **+3.50**. Fold 0 is systematically kind to the candidate,
because the candidate scores 12.68 there against 13.5–15.1 elsewhere (§4.3.2). Any
single-split ranking of these architectures would therefore understate every deficit.

## 5. Consequences for the manuscript

| item | before E11 | after E11 |
|---|---|---|
| thesis | "most method differences are unresolvable" | **"differences below ≈1.2 RMSE are unresolvable; above ≈1.5 they are real"** |
| §4.2 main table | MDE + E10/E10b calibration | **replaced by this table**; MDE becomes the explanation |
| A's role | an arm used to study the protocol | **also a measured result: best of six, by 2.4–6.8 RMSE** |
| `u_arch` in §4.6 | pending | **6.01 (spread) / 2.5 (SD across the six arms)** — the largest single component after the fold term |
| Track | Track E only | **Track E + a defensible performance table** |
| C23 | "single split turns a −7.7% advantage into a −0.9% non-result" | unchanged — E10 still holds; what narrows is the *general* claim, not the PE result |

The narrowing is not a retreat. It converts a sweeping claim that a reviewer could
falsify with one counter-example into a measured boundary with a number attached, and it
is the boundary — not the sweeping claim — that §5's guidelines operationalise.

## 6. Boundaries

* FD002 only; FD003/FD004 sealed; official test set not used (see
  `paper/TEST_SET_PROTOCOL_CHANGE.md`).
* TCN and Transformer carry one seed (Amendment 1); their rows are marked †. Their
  deficits (6.84 and 2.44) are 2.0–5.6× the seed component, so single-seeding does not
  explain them, but the paper does not use their exact ordering within a band for any
  claim.
* Baseline `d_model` is 88 (as defined in `src/baselines.py`) against 64 for the
  candidate; parameter counts are 41,858 (A), 85,801 (cnn1d), 144k (tf), 159k (tcn). The
  candidate is the smallest model in the table and the most accurate — reported as an
  observation, not as a capacity-matched conclusion.
* The candidate receives the cycle index and EMA features; the five baselines receive the
  sensor window only, as published. The ablation ladder (A 13.95 → BE 15.12 → baselines
  15.45–19.97) separates the feature contribution from the architecture contribution, and
  is reported in full.
