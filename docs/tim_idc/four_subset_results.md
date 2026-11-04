# Four-Subset Main Table + E22a-Scale Alignment

Date: 2026-11-04 (23:15). Seed 101, 5 outer folds pooled, engine-level
pooled RMSE (truth capped at 125), engine-clustered bootstrap CIs (5000
reps). Aggregation: `scripts/summarize_subsets.py` (analysis-only; the frozen
Gate 0 verdict files are untouched). JSONs:
`experiments/ida/gate0/fd34_summary.json`,
`experiments/ida/e22a_align/align_summary.json`.

## Table 1 — Four-subset five-arm main table (outer-val, caliber B)

| Subset (n units) | M0 | M1 | M1p | M1_shuf | M1_const | M1−M0 [CI] | shuf−M1 [CI] | const−M1 [CI] |
|---|---|---|---|---|---|---|---|---|
| FD001 (80) | 9.824 | 9.891 | 9.837 | — | — | +0.07 (six-seed mean; guardrail) | — | — |
| **FD002** (259) | 12.75 | **12.31** | 12.32 | 12.60 | 12.59 | 6/6 seeds sig, −0.28…−0.58 | sig 5/6 | sig 5/6 |
| FD003 (100) | 9.806 | **9.702** | 9.943 | 9.901 | 9.760 | −0.105 [−0.581, +0.359] ns | +0.199 ns | +0.059 ns |
| **FD004** (249) | 12.717 | **12.242** | 12.721 | 12.607 | 12.645 | **−0.475 [−0.749, −0.221] SIG** | **+0.365 [+0.166, +0.575] SIG** | **+0.403 [+0.177, +0.645] SIG** |

### Reading

1. **FD004 fully replicates the Gate 0 pattern, 3/3 contrasts significant**:
   prior injection helps (−0.475), shuffled and constant controls hurt.
   FD004 is the operationally hardest subset (6 conditions, 249 engines) —
   this is the strongest single-subset confirmation of the mechanism.
2. **FD003 is direction-consistent on all three contrasts but
   underpowered** (n = 100 units, same situation as the FD001 guardrail):
   −0.105 / +0.199 / +0.059, all CIs cross zero. No sign reversal anywhere.
3. **M1p vs M1 is subset-dependent**: M1p ≈ M1 on FD002/FD001, worse than M1
   on FD003 (9.943) and FD004 (12.721). The tau-phase encoding adds no
   significant benefit over raw (μ_T, σ_T) injection — consistent with
   Module B/C: the context channel saturates. M1 is the safer headline arm
   for the four-subset table; M1p stays as the seed-101 FD002 point-optimal
   variant.
4. Paper presentation: report the full table; claim significance only where
   CIs exclude 0 (FD002 six-seed G0.1, FD004 all three, FD001 as guardrail
   with the power caveat). FD003 described as direction-consistent, ns.

## Table 2 — E22a-scale alignment (big backbone: d88, 2 layers, 40/15, 141k params)

| Subset | M0big | M1big | M1pbig | M1big−M0big [CI] | M1pbig−M0big [CI] |
|---|---|---|---|---|---|
| FD002 (260) | 12.759 | 12.425 | 12.506 | **−0.334 [−0.647, −0.027] SIG** | −0.254 [−0.533, +0.011] ns |
| FD001 (100) | 10.094 | 9.846 | 9.946 | −0.247 [−0.580, +0.091] ns | −0.148 [−0.526, +0.255] ns |

### Reading

1. **The prior effect persists at E22a scale** (M1big vs M0big −0.334,
   significant on FD002) — the mechanism is not an artifact of the small
   frozen config.
2. **Capacity is not the bottleneck on outer-val**: M0big (12.759) ≈ small
   M0 (12.75); the big backbone does not beat the frozen small config. The
   gap to the historical E22a test numbers (FD002 11.51) is therefore a
   val-vs-test distribution difference (see `protocol_calibers.md`), not a
   capacity or architecture deficiency.
3. Consequence for the manuscript: the frozen small M1p/M1 remains the final
   method (smaller, faster, equal or better); the big arms are reported as a
   scale-robustness check. Official test will be unsealed once, with the
   frozen final model.
