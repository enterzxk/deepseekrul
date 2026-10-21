# Q2 Upgrade Report - Phase 6

Date: 2026-10-21

## Executive verdict

Phase 6 implemented the requested structural mechanisms, but the pre-registered
Q2 gates were **not** all met. The honest result is:

- Structural monotonicity: **PASS** (test mono violation rate = 0.000).
- Zero-diagonal adaptive graph: **implemented**, but it **hurt** FD001 accuracy.
- 5-seed Wilcoxon/bootstrap statistics: **PASS** (on the frozen P3 results).
- External baselines: **implemented**; BiLSTM completed on 4 seeds and beats Full.
- FD002 SSM flip: **FAIL** (P6 FD002 validation collapsed; run stopped).
- FD001/FD002 SOTA gate: **FAIL** (P6 5-seed Full = 15.168 / 429.3;
  P3 Full = 13.42 / 327.1; BiLSTM = 14.145 / 332.6).

Do not submit this as a Q2 SOTA paper. The structural head solves the
monotonicity symptom, but not the accuracy problem.

## 1. Monotonic-by-construction head

Implemented a terminal-anchored monotonic head:

~~~text
y_T = unconstrained terminal RUL prediction
rate_t = softplus(MLP(H_t)) >= 0
y_t = y_T + sum_{k>t} rate_k, then clamp to [0, 125]
~~~

This guarantees y_t >= y_{t+1} by construction and keeps the terminal
prediction identical to the unconstrained head.

Measured on FD001 validation (seed 42, 50 epochs, patience 20):

~~~text
val RMSE             11.937
val NASA              8068.1
val MAE               8.049
val late_fraction     0.411
val mono_violation    0.000
params                149,385
~~~

FD001 P6 five-seed test (2027-2031, 9 variants):

| Variant | RMSE | NASA | Late | Mono |
|---|---:|---:|---:|---:|
| Full | 15.168 +/- 0.622 | 429.336 +/- 78.221 | 0.498 | 0.000 |
| w/o Mono Head | 15.738 +/- 1.546 | 571.820 +/- 367.610 | 0.562 | 0.443 |
| w/o A-GCN | 14.956 +/- 0.788 | 403.072 +/- 41.698 | 0.486 | 0.000 |
| Identity graph | 15.372 +/- 0.862 | 405.458 +/- 41.490 | 0.400 | 0.000 |
| w/o SSM (LSTM) | 14.854 +/- 0.457 | 403.428 +/- 40.523 | 0.454 | 0.000 |
| BiLSTM | 14.145 +/- 0.437 | 332.595 +/- 26.621 | 0.492 | 0.000 |

Conclusion: monotonicity is solved exactly (0.000) and the mono head
improves late fraction (0.498 vs 0.562), but Full is worse than the BiLSTM
baseline and worse than the P3 model. The structural head is not enough.

## 2. Hollow zero-diagonal graph

Implemented:

- S = ReLU(E1 E2^T) / sqrt(d_e)
- mask diagonal to -1e9, row-softmax => A_tilde with A_tilde_ii = 0
- A = (1 - alpha) I + alpha A_tilde, alpha = 0.35 fixed
- adjacency diagonal = 0.65, off-diagonal mass = 0.35
- separate cross-adjacency export for Fig. 3

This removes the "identity + MLP" objection, but the FD001 test result
degraded from 13.42 / 327.1 (P3, learnable alpha ~0.1) to 15.168 / 429.3
(P6 5-seed, hollow alpha 0.35 + delta input; seed 2027 alone was 15.85 / 517.6). The mechanism is honest but not useful
at this strength. A weaker alpha (0.10-0.20) or an annealed schedule must be
tested before claiming a gain.

## 3. FD002 condition-aware SSM

The requested condition-FiLM/CM-SSM was not completed. A P6 FD002 Full run
with the existing KMeans condition-aware normalization, hollow graph, delta
input and structural head reached val RMSE 22.25 at epoch 13, versus the P3
Full 14.955 / LSTM 14.363. The run was stopped because it was clearly outside
the gate. The earlier P3 conclusion stands: **the matched LSTM beats Full on
FD002**.

Status: **FAIL**. Do not claim SSM multi-condition superiority.

## 4. External baselines

Implemented in src/baselines.py and exposed through run_ablation.py:

| Baseline | Parameters |
|---|---:|
| BiLSTM | 144,701 |
| TCN | 145,237 |
| Transformer | 144,146 |

All three are parameter-matched to Full (149,385, +3%). The code path is
validated end-to-end. BiLSTM finished on 4 seeds (2027/2029/2030/2031):
RMSE 14.145 +/- 0.437, NASA 332.595 +/- 26.621. It beats Full
(15.168 / 429.336). TCN/Transformer were not completed because TCN needed
~76 s/epoch and the run was stopped.

## 5. Statistical testing

summarize_ablations.py now computes, for every ablation/baseline:

- paired per-engine absolute-error difference vs Full;
- two-sided Wilcoxon signed-rank p-value;
- 10,000-resample bootstrap 95% CI;
- significance flags at 0.05 and 0.01.

On the frozen P3 five-seed FD001 results:

| Ablation | Mean delta abs error | 95% CI | Wilcoxon p |
|---|---:|---|---:|
| w/o A-GCN | +0.633 | [0.057, 1.194] | 0.0268 |
| Identity graph | +0.864 | [0.325, 1.424] | 0.0060 |
| w/o SSM (LSTM) | +0.661 | [0.117, 1.214] | 0.0254 |
| w/o Physics | +0.475 | [0.042, 0.902] | 0.0452 |

On P6 (new seeds), no internal ablation is significant:

| Ablation | Mean delta abs error | 95% CI | Wilcoxon p |
|---|---:|---|---:|
| w/o A-GCN | -0.429 | [-1.101, 0.264] | 0.136 |
| Identity graph | +0.273 | [-0.389, 0.940] | 0.143 |
| w/o SSM (LSTM) | -0.592 | [-1.267, 0.105] | 0.068 |
| w/o Physics | +0.348 | [-0.454, 1.157] | 0.901 |
| BiLSTM | -0.918 | [-1.608, -0.232] | 0.011 |

The only significant P6 result is negative: BiLSTM has lower absolute error
than Full. The P3 positive effects did not replicate on the new seeds.

## 6. Performance gate

| Gate | Target | Achieved | Status |
|---|---:|---:|---|
| FD001 validation RMSE | <= 11.5 | 11.937 | FAIL |
| FD001 test RMSE | <= 12.5 | 13.423 (P3), 15.168 (P6) | FAIL |
| FD001 test NASA | <= 260 | 327.1 (P3), 429.3 (P6) | FAIL |
| FD002 test RMSE | <= 13.8 | 14.955 (P3) | FAIL |
| FD002 test NASA | <= 1000 | 1456.6 (P3) | FAIL |
| Mono violation | 0.000 | 0.000 | PASS |

## Final recommendation

The Phase 6 mechanisms are real, reproducible, and honestly evaluated. They
do not produce a Q2-grade SOTA story. The correct next step is a **P7
mechanism search**, not more writing:

1. Ablate hollow alpha in {0.10, 0.20, 0.35} and delta features on/off.
2. Keep the structural monotonic head; it is the only clean win.
3. For FD002, either implement actual operating-setting FiLM or drop the SSM
   claim and use the matched LSTM as the temporal backbone.
4. Only after FD001 <= 12.5 and FD002 <= 13.8 with 5 seeds and baselines should
   the paper be framed as Q2.
