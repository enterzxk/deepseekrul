# FINAL AUDIT BUNDLE — C-MAPSS / N-CMAPSS RUL Negative & Boundary Results

**Date**: 2026-11-22
**Status**: Terminal Stop (Path 3B). No module passed the pre-registered gates.
**Test**: C-MAPSS test remains sealed (no re-run). N-CMAPSS test not read.

---

## 0. Executive summary

Three independent lines were tested:

1. **Paper B**: LTAG learned gate + RHSE-TSH sensor reconstruction on DSPT/B0.
   - LTAG frozen test 15/15, mean +0.391 RMSE.
   - But K1-Calib / Linear-Stack beat LTAG on 5-cut validation 4/4; LTAG only wins at end-of-life.
   - RHSE-TSH deleted after fair R5b/R5m baselines matched/beat it on bias/drift.
2. **Paper C (C-MAPSS clean restart)**: minimal GRU L50 `B_min` (FD001 10.630, FD004 12.142).
   - M1 CADF: FD001 +0.970, FD004 +0.115 (v1.0), +0.208 (v1.1 K0=5), +0.451/−0.061 (v1.2) → fail both-subset gate.
   - M2/M3 degrade both subsets.
   - 0/3 modules pass Gate 1; C-MAPSS frozen.
3. **Paper C (N-CMAPSS pivot)**: TCN L50 `B_min` (22.36–22.59 across DS02/05/06/07).
   - M1 per-condition: 4/4 worse than B_min.
   - M2-Lite per-step W FiLM: 4/4 worse.
   - Gate 2-N fail (0/4); terminal stop.

**Bottom line**: under honest minimal baselines, fair temporal baselines, and pre-registered gates,
none of the three proposed mechanisms produced a reproducible net gain. The strongest publishable
artifact is a **benchmark audit / negative short paper**, not a new SOTA method.

---

## 1. Paper B verified tables

### 1.1 Frozen B0 test (15/15)

| subset | seeds | K1 | LTAG | Δ(K1−LTAG) | wins |
|---|---:|---:|---:|---:|---:|
| FD001 | 6 | 13.421 | 12.995 | +0.426 | 6/6 |
| FD002 | 3 | 11.514 | 11.352 | +0.162 | 3/3 |
| FD003 | 3 | 12.979 | 12.323 | +0.656 | 3/3 |
| FD004 | 3 | 13.031 | 12.710 | +0.321 | 3/3 |
| mean | — | — | — | **+0.391** | **15/15** |

Original record +0.398; audit from JSON +0.391 (weighting/rounding).

### 1.2 Validation strong controls (observable features)

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| K1 | 12.094 | 11.527 | 12.793 | 13.608 |
| LTAG | 11.334 | 11.210 | 12.372 | 13.522 |
| K1-Calib | 10.931 | 10.970 | 12.129 | 12.839 |
| Linear-Stack | 10.150 | 10.604 | 12.033 | 13.080 |

K1-Calib / LS 4/4 beat LTAG; LS wins only 50–56% engines; FD001 median worse.

### 1.3 Cut-wise LS−LTAG (negative = LS better)

| subset | 0.40 | 0.55 | 0.70 | 0.85 | 0.95 |
|---|---:|---:|---:|---:|---:|
| FD001 | −4.170 | −0.481 | −0.302 | +0.301 | +0.267 |
| FD002 | −1.548 | −0.563 | −0.708 | −0.199 | +0.592 |
| FD003 | −1.239 | −0.646 | −0.120 | +0.213 | −0.276 |
| FD004 | −0.904 | −1.097 | −0.163 | −0.450 | +0.851 |

At 0.95 LTAG is best on FD002/FD004, second on FD001/FD003 (≤0.28 from best).

### 1.4 RHSE fair temporal baselines (3-seed mean recovery)

| subset | fault | R3 | R5b | R5m | R3−max |
|---|---|---:|---:|---:|---:|
| FD001 | bias | 0.799 | 0.793 | 0.821 | −0.022 |
| FD001 | drift | 0.856 | 0.868 | 0.925 | −0.069 |
| FD001 | stuck | 0.737 | −0.695 | −0.202 | +0.940 |
| FD003 | bias | 0.908 | 0.927 | 0.939 | −0.032 |
| FD003 | drift | 0.916 | 0.853 | 0.921 | −0.005 |
| FD003 | stuck | 0.605 | −0.179 | −0.288 | +0.783 |

Only stuck passes; RHSE-TSH removed.

### 1.5 End-to-end Phase 2 (clean engine-mean RMSE)

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| M0 | 9.957 | 10.863 | 10.486 | 11.783 |
| M0+Aug | 10.011 | 10.624 | 10.683 | 11.323 |
| M1 | 9.802 | 10.694 | 10.892 | 11.728 |
| M2 | 9.885 | 10.329 | 10.688 | 11.442 |

Fault recovery M1 ≤0.409; M2 negative on several cells; M0+Aug often better; Gate 2 fail.

---

## 2. Paper C C-MAPSS verified tables

### 2.1 B_min

| model | L | FD001 | FD004 |
|---|---:|---:|---:|
| GRU | 50 | 10.630 ± 0.912 | 12.142 ± 0.548 |
| GRU | 30 | 12.246 | 13.388 |
| TCN | 30 | 13.072 | 14.293 |
| TCN | 50 | 13.269 | 14.432 |

B_min = GRU L50 (51,073 params); B_min-Cap = GRU 3L (76,033 params).

### 2.2 M1 CADF

| arm | FD001 | FD004 | Gate |
|---|---:|---:|---|
| M1 v1.0 | 9.660 (+0.970) | 12.027 (+0.115) | fail |
| M1 v1.1 K0=5 | — | 11.934 (+0.208) | fail |
| M1 v1.2 win | 10.179 (+0.451) | 12.203 (−0.061) | fail |
| M1 v1.2 eng | 10.736 | 12.587 | fail |
| M1 v1.2 static | 10.723 | 12.651 | fail |

### 2.3 M2/M3

| arm | FD001 | FD004 |
|---|---:|---:|
| M2 full | 12.341 | 13.562 |
| M2 gate_only | 11.895 | 12.996 |
| M2 pool_only | 11.576 | 13.279 |
| M3 beta | 11.145 | 13.111 |
| M3 vanilla | 12.228 | 12.342 |

M3 PICP 0.952/0.889; MPIW 63.9/45.1.

**Gate 1: 0/3; C-MAPSS frozen.**

---

## 3. Paper C N-CMAPSS verified tables

### 3.1 B_min (TCN L50)

| subset | B_min |
|---|---:|
| DS02 | 22.587 |
| DS05 | 22.657 |
| DS06 | 21.710 |
| DS07 | 22.364 |

### 3.2 M1 / M2-Lite

| subset | B_min | M1 best | M2-Lite |
|---|---:|---:|---:|
| DS02 | 22.587 | 22.814 | 23.356 |
| DS05 | 22.657 | 23.088 | 23.422 |
| DS06 | 21.710 | 22.192 | 22.577 |
| DS07 | 22.364 | 22.760 | 23.315 |

Gate 2-N: 0/4; no revision because all arms are worse than B_min. Terminal stop.

---

## 4. Proposed short-paper / benchmark-audit outline

**Title**: *When Do RUL Modules Help? A Cut-Controlled Audit of C-MAPSS and N-CMAPSS
Aggregation, Calibration, and Sensor Reconstruction*

**Type**: IEEE TIM short paper / benchmark audit (negative + boundary).

1. **Introduction**
   - RUL papers report single averages; aggregation/calibration/sensor modules are often
     compared against weak baselines and mixed cuts.
   - Contributions: cut-controlled audit; leakage audit; fair temporal baselines; minimal
     strong baseline saturation; negative evidence for three module families.

2. **Cut-composition ranking reversal and `cycle_frac` leakage (Paper B)**
   - Validation 5-cut mean vs end-of-life test; LS vs LTAG crossover.
   - `cycle_frac = cut/true_lifetime` is not observable; removing it changes rankings.
   - Table: cut-wise LS−LTAG; test 15/15.

3. **Fair temporal baseline vs PCA reconstruction (Paper B)**
   - R5b/R5m use only observed `x_f`; R3 only wins on stuck.
   - Table: bias/drift/stuck × FD001/FD003.
   - Lesson: last-value baseline is a straw man; fair hold/median removes the advantage.

4. **Minimal GRU L50 saturation and single-vs-multi-condition absorption (Paper C)**
   - B_min GRU L50 10.63/12.14; M1 +0.97 on FD001 but −0.06 on FD004 (v1.2).
   - M2/M3 degrade.
   - N-CMAPSS: TCN L50 22.36–22.59; M1/M2 worse.
   - Lesson: strong minimal baseline + small 5-fold validation leaves no room for the modules.

5. **Recommendations**
   - Report cut-wise tables; separate engine-mean/pooled; use fair temporal baselines;
     include capacity controls; pre-register gates; never tune on test.

6. **Limitations**
   - C-MAPSS test historically observed; N-CMAPSS 6 units per subset → high variance.
   - Negative results are conditional on the tested modules and protocol, not a universal claim.

---

## 5. Evidence index

- `docs/PAPER_BOUNDARIES_20261119.md`
- `docs/PHASE1C_GATE_VERDICT.md`, `docs/PHASE2_GATE2_VERDICT.md`
- `docs/LTAGV2_*.md`
- `docs/P0_CLEAN_BASE.md`, `docs/P1_MODULE_SCREENING.md`, `docs/P2_NCMAPSS_SCREENING.md`
- `results/paper_c/*.json`
- `paper/main.pdf` (Paper B draft, not submission-ready)
