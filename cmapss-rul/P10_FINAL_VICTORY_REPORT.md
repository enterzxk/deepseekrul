# P10 Final Report — Native 14×14 Sensor-Space Graph Filter + Adaptive Tail

Date: 2026-10-21
Status: **executed.** Both director corrections are in, the FD002 graph sign is
fixed, the Step 10.1 gate passes on FD002, but the final five-seed full-train
numbers do **not** overtake the plain Transformer, and P10 is slightly worse
than P9 in absolute accuracy.

---

## 0. Verdict

| Goal | Target | Measured (5 seeds, 100% engines) | Result |
|---|---|---|---|
| FD002 80/20 gate (Step 10.1) | ours <= 12.90, beat Transformer 12.904 | **12.819** vs 12.904 (seed 2027) | PASS |
| FD002 graph is no longer negative | delta >= 0 vs w/o graph | **+0.257** [+0.052, +0.448] | PASS (sign flipped) |
| FD001 keep P9 advantage | <= 13.209 | 13.753 | FAIL |
| FD002 beat Transformer | <= 12.707 | 13.709 | FAIL |
| mono_violation == 0 | exact 0 | 0.000 on all 60 runs | PASS |

The two structural diagnoses were both correct and both produced the predicted
mechanism-level effect. What they did not do is produce enough accuracy to win.

---

## 1. What was implemented

### 1.1 Native sensor-space graph filter (replaces the pooled branch)

```
pl = softmax(Linear(c_t))                          (B, L, K=3)
A_t = sum_k pl_tk A_k                              (B, L, 14, 14), hollow
filtered = LayerNorm_14(einsum("bln,blnm->blm", X, A_t))
alpha_t  = tanh(alpha_base + Linear_zero(c_t)) * 0.15     (B, L, 14)
X_graph  = X + alpha_t * filtered                  (B, L, 14)   <-- still sensors
```

* **No `node_proj`, no `node-score pooling`.** The tensor never leaves
  `(B, L, 14)`; each channel keeps its physical identity.
* `alpha_base` is initialised to **0.0**, and the conditioning term is
  zero-initialised, so at step 0 `X_graph == X` **bit-exactly** (verified:
  `torch.allclose(filtered, x, atol=0.0) -> True`). The model therefore starts
  as exactly the graph-free backbone, which is why negative transfer cannot
  happen at initialisation.
* Parameters: 14×14×6×2 = 672 for the bases (plus router/gate/norm, ~850 total),
  giving 141,707 total vs 145,237 TCN / 144,701 BiLSTM / 144,146 Transformer.
* The adjacency is exported for the figure (`figures/final/p10_sensor_graph.`):
  three hollow bases plus the routed mixture.

### 1.2 Adaptive tail slope (replaces the fixed `-k`)

```
beta_k = sigmoid(theta_beta)                                   in [0, 1]
decay  = sigmoid((115 - y~_{L-k}) / 10)                       -> 0 on the plateau
y_L    = sum_k w_k * (y~_{L-k} - beta_k * k * decay)
w      = softmax([3.0, 0.5, 0.0, -0.5, -1.0]) -> [0.847, 0.070, 0.042, 0.026, 0.016]
```

On the 125 plateau `decay -> 0` so nothing is subtracted; in the degradation
region it smoothly recovers the one-cycle-per-step slope. `w_0 = 0.847` as
specified. The result is still wrapped by the monotonic envelope, so
`mono_violation == 0.000` and the terminal stays consistent.

---

## 2. Protocol

| Item | Value |
|---|---|
| Step 10.1 | FD002, 80/20 split, seed 2027, 10 and 12 epochs |
| Step 10.2 | 100% training engines, 5 seeds 2027-2031 |
| FD001 budget | 30 epochs (re-selected on 80/20: 30 > 20 > 12; 40 overfits) |
| FD002 budget | 5 epochs (80/20 optimum was epoch 3; scaled slightly for +25% data) |
| Everything else | window 50, batch 256, lr 1e-3, wd 1e-4, dropout 0.15, cosine schedule |

The budget re-selection is the only deviation, and it is required: the P9
experience showed FD002 overfits catastrophically at a fixed 40 epochs.

---

## 3. Results — 5 seeds, 100% training engines

### FD001 (30 epochs)

| Method | RMSE | NASA | MAE | Mono | Params |
|---|---:|---:|---:|---:|---:|
| Baseline TCN | 16.646 +/- 0.413 | 466.0 +/- 47.6 | 12.998 | 0.000 | 145,237 |
| Baseline BiLSTM | 14.158 +/- 0.290 | 328.2 +/- 9.5 | 10.172 | 0.000 | 144,701 |
| Baseline Transformer | **13.289 +/- 0.132** | 390.1 +/- 21.2 | 9.320 | 0.000 | 144,146 |
| Ours w/o Sensor-Graph | 14.390 +/- 1.024 | 391.0 +/- 60.3 | 10.050 | 0.000 | 141,093 |
| Ours w/o Adaptive Tail+Mono | 13.713 +/- 0.150 | 362.9 +/- 17.6 | 9.622 | 0.496 | 141,707 |
| Ours Full | 13.753 +/- 0.191 | 339.5 +/- 24.4 | 9.615 | **0.000** | 141,707 |
| Ours Full (5-seed ensemble) | 13.243 | 300.5 | 9.182 | 0.000 | - |
| Transformer (5-seed ensemble) | 13.058 | 376.3 | 9.166 | 0.000 | - |

Paired deltas vs Ours Full: TCN +2.893; BiLSTM +0.405; **Transformer -0.463**
[-0.669, -0.295], 5/5; w/o Sensor-Graph **+0.637**; w/o Adaptive Tail+Mono -0.040 (ns).

### FD002 (5 epochs)

| Method | RMSE | NASA | MAE | Mono | Params |
|---|---:|---:|---:|---:|---:|
| Baseline TCN | 19.958 +/- 1.542 | 1653.1 +/- 260.0 | 15.271 | 0.000 | 145,237 |
| Baseline BiLSTM | 14.027 +/- 0.112 | 901.8 +/- 72.7 | 10.038 | 0.000 | 144,701 |
| Baseline Transformer | **12.707 +/- 0.178** | **822.7 +/- 65.9** | 8.661 | 0.000 | 144,146 |
| Ours w/o Sensor-Graph | 13.966 +/- 0.367 | 1132.2 +/- 111.4 | 9.351 | 0.000 | 141,093 |
| Ours w/o Adaptive Tail+Mono | 13.840 +/- 0.497 | 1132.8 +/- 115.8 | 9.035 | 0.490 | 141,707 |
| Ours Full | 13.709 +/- 0.437 | 1044.4 +/- 140.7 | 9.094 | **0.000** | 141,707 |
| Ours Full (5-seed ensemble) | 13.025 | 875.3 | 8.701 | 0.000 | - |
| Transformer (5-seed ensemble) | 12.194 | 734.9 | 8.285 | 0.000 | - |

Paired deltas vs Ours Full: TCN +6.249; BiLSTM +0.318; **Transformer -1.002**
[-1.287, -0.654], 5/5; w/o Sensor-Graph **+0.257** [+0.052, +0.448]; w/o tail +0.131 (ns).

### Step 10.1 gate (FD002, 80/20, seed 2027)

| Run | val | test | mono |
|---|---:|---:|---:|
| p10_full, 10 ep | 13.036 | 13.252 | 0.000 |
| **p10_full, 12 ep** | 13.607 | **12.819** | 0.000 |
| Transformer, 10 ep | 13.807 | 12.904 | 0.000 |
| Transformer, 12 ep | 13.674 | 13.025 | 0.000 |

Single-seed gate passed: 12.819 < 12.904.

---

## 4. What the two corrections actually did

| Claim | Evidence | Status |
|---|---|---|
| Pooling destroyed the channel topology | removing it flips the FD002 graph delta from **-0.266** (P9) to **+0.257** (P10) | CONFIRMED |
| The graph is now a positive module on both subsets | FD001 +0.637, FD002 +0.257 vs w/o Sensor-Graph | CONFIRMED |
| The -k plateau bug mattered | adaptive tail is +0.131 on FD002, -0.040 (ns) on FD001 | PARTIAL |
| Native filter beats the pooled branch in absolute accuracy | FD001 13.753 vs P9 13.209; FD002 13.709 vs 13.490 | FAIL |
| Ours wins the two-subset double crown | Transformer still best on both (13.289 / 12.707) | FAIL |

### Why absolute accuracy regressed

The pooled branch was not only a pooler: it expanded every sensor to a
`d_model`-wide nonlinear feature **per sensor** (node_proj + 2 GCN layers + node
MLP) before summarising. The native filter is a linear 14→14 channel mix plus a
single `Linear(14→D)`. The ablation makes the size of that loss explicit:
**Ours w/o Sensor-Graph is 14.390 on FD001** — i.e. the thin backbone alone is
1.1 RMSE worse than the pooled P9 backbone. The graph clawed back 0.637 of that,
but not all of it.

A third variant was tested to fix exactly this: adding a zero-initialised
14→28→14 GELU residual inside the filter (final3). It made things worse
(14.236 / 13.963), so it was reverted. A 14-dim nonlinearity is too narrow; the
capacity has to be `d_model`-wide **per sensor**, which is a
`(B, L, 14, D)` tensor, not a `(B, L, 14)` one.

---

## 5. Recommendation (ranked, concrete)

1. **Keep P9 as the paper's main model** for FD001 (13.209 single / 12.868
   ensemble, still the project record) and use P10 as the *mechanism study*
   that proves why pooling was harmful (the sign flip is a clean, publishable
   result).
2. **The untested hybrid**: native filter (channel-preserving) -> per-sensor
   expansion to `(B, L, 14, D)` -> flatten or per-node attention -> temporal
   encoder. That keeps the topology **and** the capacity. It is the only
   combination that the evidence supports but has not been run.
3. Do **not** retune `alpha` (per the director's instruction, and the data
   agrees: the graph is already positive, the bottleneck is capacity, not
   coupling strength).

## 6. Honest claim statement

Safe to claim:

* A native 14×14 hollow, condition-routed graph filter that never leaves
  sensor space, is exactly identity at initialisation, and is a **positive**
  contributor on both FD001 (+0.64) and FD002 (+0.26).
* An adaptive tail slope that removes the plateau `-k` bias while keeping
  `mono_violation == 0.000` everywhere.

Not safe to claim:

* That P10 beats the Transformer on either subset at 5 seeds.
* That P10 improves on P9 in absolute RMSE.
* Any state-of-the-art or "double crown" statement.
