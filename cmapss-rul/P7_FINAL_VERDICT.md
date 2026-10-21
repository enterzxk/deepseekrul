# P7 FINAL VERDICT - Surgical P7

Date: 2026-10-21

## Verdict

P7 fixed the two worst P6 failures:

- No more delta features: the FD002 collapse disappeared.
- BiLSTM-Attn is a better temporal backbone than the hand-written Conv-SSM.
- FD002 Full reaches 14.120 / 799.204, which **passes** the <=14.2 / <=1000 gate.

But P7 did **not** pass the FD001 gate and did **not** beat the external
baselines. Do not submit as Q2 SOTA.

## 1. FD001 five-seed (seeds 2027-2031)

| Variant | RMSE | NASA | MAE | Late | Mono | Params |
|---|---:|---:|---:|---:|---:|---:|
| Full (Hollow-A-GCN + BiLSTM-Attn + Mono) | 14.155 +/- 0.444 | 357.740 +/- 18.265 | 9.984 | 0.490 | 0.000 | 143,383 |
| w/o Mono Head | **13.587 +/- 0.546** | **313.152 +/- 38.795** | **9.576** | **0.448** | 0.339 | 139,158 |
| w/o A-GCN | 14.370 +/- 0.566 | 385.469 +/- 30.778 | 10.093 | 0.498 | 0.000 | 142,935 |
| Baseline BiLSTM | 14.959 +/- 0.758 | 362.637 +/- 31.798 | 10.801 | 0.456 | 0.000 | 144,701 |

Key paired tests vs Full:

| Comparison | Delta abs error | 95% CI | p |
|---|---:|---|---:|
| w/o Mono Head | -0.408 | [-0.796, -0.022] | 0.019 |
| w/o A-GCN | +0.108 | [-0.280, 0.512] | 0.690 |
| Baseline BiLSTM | +0.817 | [+0.286, +1.358] | 0.038 |

Interpretation:

- The structural mono head still gives exactly 0.000 violation.
- But removing it improves RMSE by 0.568 and NASA by 44.6, with p=0.019.
  **Mono is a safety/structural contribution, not an accuracy contribution.**
- A-GCN is not significant in P7.
- Full beats the BiLSTM baseline, but that baseline is 14.959; the bar is low.
- The best P7 model is w/o Mono at 13.587 / 313.2, still above the 12.5 gate.

## 2. FD002 one-seed validation/protocol check (seed 2027)

| Variant | RMSE | NASA | MAE | Params |
|---|---:|---:|---:|---:|
| Full (FiLM-BiLSTM + Hollow-A-GCN + Mono) | 14.120 | 799.204 | 9.742 | 143,383 |
| w/o A-GCN | 14.221 | 893.077 | 10.170 | 142,935 |
| w/o Mono Head | 14.321 | 1027.759 | 10.114 | 139,158 |
| Baseline BiLSTM | 13.881 | 832.390 | 10.165 | 144,701 |
| Baseline Transformer | **12.679** | **785.018** | **9.021** | 144,146 |

Interpretation:

- FD002 no longer collapses (P6 was 22.25 at epoch 13).
- Full passes the 14.2 / 1000 gate.
- **Transformer beats Full by 1.44 RMSE and BiLSTM beats it by 0.24.**
  The main model is not the best model on FD002.
- This is one seed only; it is a protocol check, not a 5-seed confirmation.

## 3. What was not completed

- FD001 TCN/Transformer: TCN took ~60 s/epoch on FD001 and blocked the
  5-seed six-way run; it was stopped. TCN/Transformer remain external
  baselines from the FD002 one-seed check (Transformer is strongest there).
- FD002 five seeds: not run after the one-seed check showed Full is not best.
- Zero-init FiLM is implemented and active only when condition windows are
  supplied; FD002 uses normalized op_1/op_2/op_3 windows.

## 4. Honest architecture verdict

| Claim | Evidence | Status |
|---|---|---|
| Remove delta features | FD002 22.25 -> 14.12 | PASS |
| BiLSTM-Attn > Conv-SSM | P6 15.168 -> P7 14.155 | PASS |
| FD002 <=14.2 / <=1000 | 14.120 / 799.204 | PASS (1 seed) |
| FD001 <=12.5 | 14.155 (Full), 13.587 (w/o Mono) | FAIL |
| Full beats BiLSTM | 14.155 vs 14.959 | PASS (weak baseline) |
| Full beats Transformer | 14.120 vs 12.679 on FD002 | FAIL |
| Mono improves accuracy | w/o Mono 13.587 vs Full 14.155 | FAIL |
| A-GCN significant | p=0.690 | FAIL |

## 5. Recommendation

The best P7 configuration is **w/o Mono Head (BiLSTM-Attn + Hollow-A-GCN +
Asymmetric Loss)**: 13.587 / 313.2. Use it as the new accuracy base. Keep the
structural mono head as a separate safety variant, not as the main model.

To reach Q2, the remaining gap is FD001 13.587 -> 12.5 and beating the
Transformer on FD002. That requires a new mechanism search, not another seed
rerun.

Do not claim "PA-STGRN beats SOTA" with these numbers.
