# Paper B defensible claims (working document)

Every claim below must point to a result file. Claims marked `[PENDING]` are
not yet supported.

## C1. The historical ablation is checkpoint-consistent but protocol-confounded

**Evidence:** `results/legacy_repro.csv`, `reports/ARCHIVE_INDEX.md`.

The archived 32-cell test table is arithmetically exact (max diff 1.8e-15 vs
the prediction CSVs) and the checkpoints reproduce the predictions with
corr 0.9935-0.9995 under a reconstructed evaluator. The table is therefore not
fabricated. It is nevertheless unusable as paper evidence because (i) it is a
single seed/fold, (ii) the committed evaluator cannot load the checkpoints,
(iii) early stopping used window-level validation, and (iv) the official test
sets were observed repeatedly during development.

## C2. The sign of the DSPT ablation depends on the protocol

`[PENDING]` Under the fixed protocol (engine-level validation, P0 fixes,
3 seeds), the FD001 validation ranking is FULL < base in RMSE, opposite to the
archived test table. The official test comparison after the freeze will decide
whether this is a real gain or a validation artefact.

## C3. Multi-scale convolution

`[PENDING]` The `multi3` control (three k=3 kernels + projection,
parameter-matched to full) distinguishes multi-scale from capacity/projection.

## C4. Cross-sensor attention

`[PENDING]` SA is tested against sensor-identity and condition-embedding
variants. Mechanism evidence: cross-condition sensor-correlation changes are
10-19x larger on FD002/FD004 (mean |Delta corr| 0.208/0.114 vs 0.015/0.015;
top pairs s8-s14 0.558, s13-s14 0.557 on FD002).

## C5. Health-indicator supervision

`[PENDING]` HI is tested against h_B source, no-monotone, and inverted-sign
variants. Mechanism evidence: HI is a decreasing health index (Spearman
-0.79..-0.92), locally noisy (~48% of adjacent steps against the trend), and
largely redundant with RUL (R^2 0.61-0.78).

## C6. Vanilla Transformer comparison

`[PENDING]` DSPT vs Vanilla at the same window, plus a feature-matched
control (same 3-channel input and cycle encoding). Parameter counts are
reported; no capacity-matched claim is made unless the feature-matched control
is used.

## C0. Positioning against the closest work (must be read before submission)

A web search on 2026-11-11 found three relevant lines:

1. **Protocol audits of C-MAPSS.** "Auditing Operating-Condition Normalization
   in C-MAPSS Remaining Useful Life Evaluation" (engrxiv preprint) and
   "Leakage-Controlled Grouped Validation and Interval-Safety Calibration for
   RUL/SOH Prediction" (MDPI Machines, 2026). They audit normalization and
   grouped validation. Our work must **not** claim to be the first C-MAPSS
   protocol audit; it differs by studying the *sign of an architectural
   ablation* and the mechanism behind it.
2. **Multi-sensor attention for RUL.** MSDA-Net (IEEE, 2026) reports a 3.3%
   dual-attention ablation gain on C-MAPSS. It does not report a condition-wise
   heterogeneity analysis or a condition-aware attention control.
3. **Multi-scale convolution + attention.** IEEE 2025 and Scientific Reports
   2025 papers use multi-scale temporal convolution; none report a
   parameter-matched single-scale control (`multi3`) or an autocorrelation
   argument for why the scales are redundant.

**Differentiation to verify by reading the papers:** our contribution is a
controlled, pre-registered ablation of one architecture with (i) engine-level
truncated validation, (ii) a mechanism probe for SA (cross-condition sensor
correlation change), HI (target noise/redundancy) and MS (scale redundancy),
and (iii) a reported sign flip between the historical and fixed protocols.
This is narrower than a benchmark audit and more mechanistic than a
single-seed architecture paper.

## C7. What is not claimed

- No claim that the official test sets are pristine.
- No claim of universal superiority across subsets.
- No claim based on a single seed.
- No claim that the historical numbers are fabricated.
