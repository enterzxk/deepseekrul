# Experimental Setup (arms & frozen criteria — v0.1)

> Pre-registered in `gates.json` before any arm was trained; the verdict
> script (`scripts/summarize_gate0.py`) reads the same file. This section can
> be written now because it describes design, not outcomes.

## 4.1 Arms (Gate 0: does the causal prior add value at all?)

| arm | definition | injection | params |
|-----|-----------|-----------|--------|
| M0 | PE backbone, unchanged | — | 42,306 |
| M1 | PE + concat(μ_T, σ_T) | zero-init linear → pooled context | +192 |
| M1p | PE + φ(tau), tau = t/μ_T, u = σ_T/μ_T | zero-init linear → pooled context | +640 |
| M1_shuf | M1 with μ_T permuted across engines (within fold) | same | +192 |
| M1_const | M1 with constant outer-train median lifetime | same | +192 |

Zero-init guarantees every arm starts exactly at M0 (asserted at runtime,
tolerance 1e-4), so any learned difference is attributable to the prior
signal, not to initialization noise.

φ(tau) = [sin(2π·tau/τ_k), cos(2π·tau/τ_k)] for τ_k ∈ {0.125, 0.25, 0.5, 1.0}
plus the uncertainty scalar u (9 inputs).

## 4.2 Comparison arms (Gate 1: is the backbone route even necessary?)

- **H1**: HGB directly predicts RUL from the same causal prefix features
  (5-seed ensemble, same outer folds, same scaler, same window set, weight
  1/n_windows per engine, prediction clipped at 0).
- **H2**: weighted average of PE and H1 predictions; the weight is chosen on
  inner CV *inside outer-train* (never on outer-val).
- **H3**: PE with the H1 RUL prediction appended as an input feature.

G1.1–G1.3 (frozen): M1 or M1p must beat each of H1/H2/H3 on FD002 with CI
excluding 0. Otherwise the conclusion becomes "the tabular estimator alone
suffices" and the paper re-positions to an analysis paper — no further
modules.

## 4.3 Mechanism modules (Gate 1 pass only; one small experiment each)

- **B (degradation-conditioned temporal attention)**:
  A = QKᵀ/√d + B_deg, B_deg = f(φ(tau), u) as head-wise bias, last layer
  zero-init, added params < 10%.
- **C (degradation-aware sensor mixer)**: H ← H + g(tau,u)·U(τ)V(τ)ᵀH,
  rank ≤ 8, zero-init gate.
- Controls: absolute-cycle-conditioned variant and shuffled-tau variant are
  mandatory; a module survives only if it beats M1p with CI excluding 0 AND
  beats its shuffled control.

## 4.4 Uncertainty gate (Gate 2 pass only)

G2.1 (frozen): Spearman correlation between u = σ_T/μ_T and |μ_T − T| on
outer-val engines must exceed 0.3 on FD002. If it fails, the uncertainty-
aware contribution is dropped from the paper entirely.

## 4.5 Verdict statistics

- Statistical unit: **engine** (never windows).
- DeltaRMSE(a−b) computed on the same engine resample; pooled across the 5
  outer folds (engines are fold-disjoint, so pooling is exact).
- 5,000 bootstrap replicates, 95% percentile CI; seeds 101/202/303 reported
  as mean ± std for final tables.
- Gate decisions use seed-101 Stage A; Stage B (3 seeds) confirms any pass
  before the number is frozen for the paper.

## 4.6 Compared methods (Table C, unified protocol)

| method | source | route into Table C |
|--------|--------|--------------------|
| PE (M0) | ours | native |
| IDA-PE | ours | native |
| CET-MSA | Springer JIM 2026, official code verified | run authors' code, then port to our folds/targets/scaler |
| BootBoGs-TS (LSTM) | Springer, official code verified | authors' protocol numbers → Table A only (tuning framework, not architecture) |
| LVDACNN | IEEE TIM 2026, no official code | faithful reimplementation from paper (guarded: only if reproduction quality is defensible) |
| AIDGN-RUL / MSMT-FDA | 2026, existence verified, no official code | Table A only ("as reported", protocol annotated) |

Published numbers from different protocols are labeled "as reported" and are
never ranked against ours.
